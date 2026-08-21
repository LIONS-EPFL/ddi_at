from enum import Enum

import numpy as np
import torch
import torch.nn as nn
import wandb
import torchvision
from torchvision import transforms
from pytorch_lightning.profiler import PassThroughProfiler
import torchmetrics
import pytorch_lightning as pl
from torch.nn import functional as F
import matplotlib
from warnings import warn

from acat.danskinattack import DanskinAttack, max_ensemble_attack
from acat.jax_danskinattack import JaxDanskinAttack, jax_max_ensemble_attack
from torch.optim.lr_scheduler import ReduceLROnPlateau,LambdaLR,CyclicLR
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from acat.config import AttackArgs, Hpars
from acat.resnet import ResNet18, ResNet50


class Stage(Enum):
    train = 1
    val = 2
    test = 3

class LossType(Enum):
    clean = 1
    adv = 2

def overwrite_gradients(parameters, grad):
    for p, g in zip(parameters, grad):
        if p.grad is None:
            p.grad = torch.zeros_like(p)
        flattened_view = p.grad.view(-1)
        flattened_view[:] = g.view(-1)


NORMALIZERS = {
    'cifar10': ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    'mnist': ((0.1307,), (0.3081,))
}
CHANNELS = {
    'cifar10': 3,
    'mnist': 1,
}
def print_once(text,id=None):
    key=(text,id)
    if not key in print_once.done:
        print_once.done.add(key)
        print(text)
print_once.done=set()
class Model(pl.LightningModule):
    def __init__(self, 
        hparams, 
        profiler=None, 
        classes=None,
    ):
        super().__init__()

        # Allows us to do the gradient update manually
        self.automatic_optimization = False

        # Classes  
        if classes is None:
            raise ValueError("`classes` needs to be set with list of class names")
        self.classes = classes
        self.num_classes = len(classes)

        # profiling
        self.profiler = profiler or PassThroughProfiler()

        # make sure we have both hparam dict for tensorboard/ckpt and config for tab completion
        self.hparams.update(Hpars.to_dict(hparams))
        self.config = Hpars.to_cls(hparams)
        self.save_hyperparameters()

        # Model
        model_kwargs = dict(num_classes=self.num_classes, ch_in=CHANNELS[self.config.dataset],
                            batch_norm=self.config.batch_norm,celu=self.config.celu)
        if self.config.model == 'resnet18':
            model = ResNet18(**model_kwargs)
        elif self.config.model == 'resnet50':
            model = ResNet50(**model_kwargs)
        elif self.config.model == 'linear':
            dim = 3 * 32 * 32 # CIFAR10 dimensions
            model = nn.Sequential(
                nn.Flatten(),
                nn.Linear(dim, self.num_classes),
            )
        elif self.config.model == 'alex':
            model = torchvision.models.SqueezeNet(**model_kwargs)

        # Dataset specific normalization
        if self.config.normalize:
            normalizer = transforms.Normalize(*NORMALIZERS[self.config.dataset])
        else:
            normalizer = transforms.Normalize((0.0,0.0,0.0), (1.0,1.0,1.0))
        self.model = nn.Sequential(normalizer, model)

        # Logging (see https://pytorch-lightning.readthedocs.io/en/stable/extensions/metrics.html)
        self.metrics = {
            (Stage.train, LossType.adv): torchmetrics.Accuracy(),
            (Stage.val, LossType.adv): torchmetrics.Accuracy(),
            (Stage.test, LossType.adv): torchmetrics.Accuracy(),

            (Stage.train, LossType.clean): torchmetrics.Accuracy(),
            (Stage.val, LossType.clean): torchmetrics.Accuracy(),
            (Stage.test, LossType.clean): torchmetrics.Accuracy(),
        }

        self.val_adv_acc_argmax_epoch = torch.tensor(-1.0)
        self.val_adv_acc_max = torch.tensor(-1.0)

        # Register to have the right device type
        self.metrics_modules = nn.ModuleList([metric for metric in self.metrics.values()])

    def configure_optimizers(self):
        opt = self.config.model_opt.make(self.model)

        # Model scheduler
        if self.config.model_opt.scheduler == 'step':
            milestones = np.array(self.config.model_opt.scheduler_milestones)
            milestones *= self.config.epochs
            milestones = np.floor(milestones)

            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    opt, 
                    milestones,
                    gamma=self.config.model_opt.scheduler_factor)
            schedulers = [{
                'scheduler': scheduler,
                'name': 'model_lr',
                # 'interval':'epoch',
                # 'frequency': 1,
            }]

        elif self.config.model_opt.scheduler=="const-overt":
            def linear_decay(epoch):
                start_epoch=(self.config.epochs*0.5)
                if epoch>=start_epoch:
                    return 1.0/max(1,min(1,(epoch-start_epoch)*0.5))
                else:
                    return 1.0
            schedulers=[{'scheduler':LambdaLR(opt,linear_decay),'name':'model_lr'}]
        elif self.config.model_opt.scheduler=="linear":
            def linear_decay(epoch):
                return 1.0/max(1,epoch)
            schedulers=[{'scheduler':LambdaLR(opt,linear_decay),'name':'model_lr'}]
        elif self.config.model_opt.scheduler == "onecycle":
            lr=self.config.model_opt.lr
            base_lr=lr*1e-4
            max_lr=lr
            # rising and falling steps
            slope_steps=self.config.epochs*int(5e4*(1-self.config.val_split))//(self.config.batch_size*2)
            sched=CyclicLR(opt, step_size_up=slope_steps,base_lr=base_lr,max_lr=max_lr)
            schedulers = [{'scheduler': sched, 'name': 'model_lr'}]
        elif self.config.model_opt.scheduler=="plateaudrop":
            sched=ReduceLROnPlateau(opt,patience=2,factor=0.5)
            schedulers = [{'scheduler': sched, 'name': 'model_lr'}]
        else:
            schedulers = []

        return [opt], schedulers

    def forward(self, x):
        """Return logits
        """
        return self.model(x)

    def predict(self, logits):
        return logits.argmax(dim=-1)

    def loss(self, logits, y, reduction="mean"):
        return F.cross_entropy(logits, y, reduction=reduction)

    def attack(self, batch, stage=Stage.train):
        img, label = batch
        num_of_attacks=self.config.num_of_attacks

        
        if stage == Stage.train:
            attack_config: AttackArgs = self.config.attack_opt
        else:
            attack_config: AttackArgs = self.config.test_attack_opt

        loss = lambda logits, y: self.loss(logits, y, reduction="mean")
        attack = attack_config.make(self.model, loss, num_classes=self.num_classes)

        prev_training = bool(self.training)
        self.eval()

        with torch.enable_grad():
            adv_img = attack(img, label)
            if prev_training:
                self.train()
            return adv_img, label

    def training_step(self, batch, batch_idx):
        if self.config.danskinattack==True:
            print_once("Have Danskinattack")
            if self.config.jaxed==True:
                print_once("Have Danskinattack jaxed")
                danskinattack = JaxDanskinAttack(self.config.normalize_danskin)
            else:
                print_once("Have Danskinattack numpy")
                danskinattack = DanskinAttack(self.config.normalize_danskin)
            descent_direction = danskinattack(self, batch, num_of_attacks=self.config.num_of_attacks)
            obj=np.asarray(danskinattack.simplex_min.store)
            fig,ax=plt.subplots()
            ax.plot(obj)
            self.logger.experiment.log(dict(inner_objective=wandb.Image(fig)))
            plt.close("all")
            opt = self.optimizers()
            opt.zero_grad()
            self.update_gradients_manually(self.model.parameters(), descent_direction)
            self.log_step_norm(opt)
            opt.step()
        elif self.config.max_ensemble==True:
            print_once("Max_ensemble")
            opt = self.optimizers()
            opt.zero_grad()

            losses = max_ensemble_attack(self, batch, num_of_attacks=self.config.num_of_attacks)
            loss = losses.mean()
            
            self.log_loss(loss, stage=Stage.train, type=LossType.adv)
            self.manual_backward(loss)
            self.log_step_norm(opt)
            opt.step()
        else:
            print_once("Vanilla")
            opt = self.optimizers()
            opt.zero_grad()

            loss = self._compute_loss(
                batch, 
                batch_idx, 
                stage=Stage.train, 
                type=LossType.adv, 
                reduction="mean", 
                return_logits=False)

            self.manual_backward(loss)
            self.log_step_norm(opt)
            opt.step()

    def log_step_norm(self, opt):
        grad_sum = 0.0
        with torch.no_grad():
            for p in self.model.parameters():
                if p.grad is not None:
                    grad_sum += p.grad.norm().pow(2.0)
        grad_sum = torch.sqrt(grad_sum)
        lr=max(group['lr'] for group in opt.param_groups) if len(opt.param_groups)>0 else 0.0
        step_norm = grad_sum * lr
        self.log("applied_step_norm", step_norm)

    def update_gradients_manually(self, parameters, grad):
        overwrite_gradients(parameters, grad)

    def _compute_loss(self, batch, batch_idx, 
        stage=Stage.train, 
        type=LossType.clean, 
        reduction="mean", 
        return_logits=False
    ):
        """Used by training, testing and validation
        """
        # Attack
        if type == LossType.adv:
            batch = self.attack(batch, stage)

        # Compute loss
        x, y = batch
        logits = self(x)

        # Normal cross entropy
        loss = self.loss(logits, y, reduction=reduction)
        
        # Log loss
        y_hat = self.predict(logits)
        on_step = stage == Stage.train
        self.metrics[(stage, type)](y_hat, y)
        self.log(f'{self.stage_name(stage)}_{type.name}_acc', self.metrics[(stage, type)], 
                 on_step=on_step, on_epoch=True, prog_bar=True)
        self.log_loss(loss, stage, type)

        if return_logits:
            return loss, logits
        else:
            return loss

    def log_loss(self, loss, stage:Stage, type: LossType):
        self.log(f'{self.stage_name(stage)}_{type.name}_loss', loss.mean(), prog_bar=True)

    def on_train_epoch_end(self) -> None:
        # Call lr scheduler manually since we are in manual mode
        for lr_scheduler in self.trainer.lr_schedulers:
            if isinstance(lr_scheduler["scheduler"],ReduceLROnPlateau):
                pass# we do this in validate
                #lr_scheduler["scheduler"].step("val_adv_acc")
            else:
                lr_scheduler["scheduler"].step()

    def on_validation_epoch_end(self) -> None:
        if self.config.val_adv:
            # Store val stats for current best accuracy
            avg_prec = self.metrics[(Stage.val, LossType.adv)].compute().detach()
    
            if avg_prec >= self.val_adv_acc_max:
                self.val_adv_acc_argmax_epoch = self.current_epoch
                self.val_adv_acc_max = avg_prec
            for lr_scheduler in self.trainer.lr_schedulers:
                if isinstance(lr_scheduler["scheduler"],ReduceLROnPlateau):
                    lr_scheduler["scheduler"].step(-avg_prec) # we are in min mode by default
            
            self.log('val_adv_acc_argmax_epoch', float(self.val_adv_acc_argmax_epoch))
            self.log('val_adv_acc_max', self.val_adv_acc_max)        

    def validation_step(self, batch, batch_idx):
        if self.config.log_verbose and batch_idx == 0:
            adv_img, _ = self.attack(batch, stage=Stage.val)
            self._log_examples(adv_img, stage=Stage.val, type=LossType.adv, N=20)

        self._compute_loss(batch, batch_idx, stage=Stage.val, type=LossType.clean)
        if self.config.val_adv:
            self._compute_loss(batch, batch_idx, stage=Stage.val, type=LossType.adv)

    def test_step(self, batch, batch_idx):
        self._compute_loss(batch, batch_idx, stage=Stage.test, type=LossType.clean)
        if self.config.val_adv:
            self._compute_loss(batch, batch_idx, stage=Stage.test, type=LossType.adv)

    def _log_examples(self, img, stage: Stage, type: LossType, N=20):
        # For debugging possible augmentations and the attack
        img_sample = img[:N]
        grid = torchvision.utils.make_grid(img_sample)
        self._log_image(f'{self.stage_name(stage)}_{type.name}_img', grid)

    def _log_image(self, title, image):
        self.logger.experiment.log({title: wandb.Image(image)}, commit=False)

    def stage_name(self, stage: Stage):
        return stage.name
