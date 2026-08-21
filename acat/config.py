import os
from typing import Union, Optional

import torch
import attr
from torchattacks.attacks.tpgd import TPGD
from torchattacks.attacks.autoattack import AutoAttack
from torchattacks.attacks.ffgsm import FFGSM

from acat.pgd import PGD
from acat.defaultpgd import PGD as defaultPGD
# should be equivalent to the following except the init_dist
#from torchattacks.attacks.pgd import PGD as defaultPGD


class Base(object):
    @classmethod
    def to_dict(cls, x):
        if isinstance(x, dict):
            return x
        else:
            return attr.asdict(x)

    @classmethod
    def to_cls(cls, x):
        if isinstance(x, cls):
            return x
        else:
            return cls(**x)

    @classmethod
    def factory(cls, **kwargs):
        return cls()


@attr.s()
class OptArgs(Base):
    type = attr.ib(type=str, default="sgd", validator=attr.validators.in_({"sgd","adam"}))
    lr = attr.ib(type=float, default=0.1)
    momentum = attr.ib(type=float, default=0.9)
    weight_decay = attr.ib(type=float, default=5e-4)

    # Step scheduler
    scheduler = attr.ib(type=str, default='step', validator=attr.validators.in_({None, "step","plateaudrop","linear","onecycle","const-overt"}))
    scheduler_milestones = attr.ib(type=list, default=[0.33, 0.66])
    scheduler_factor = attr.ib(type=float, default=0.1)

    def make(self, model):
        # Configurate model optimizer
        if self.type == "sgd":
            opt = torch.optim.SGD(
                model.parameters(), 
                lr=self.lr,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
            )
        elif self.type == "adam":
            opt=torch.optim.Adam(
                model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay
            )
        else:
            raise ValueError("Invalid model opt", self)
        
        return opt


@attr.s()
class AttackArgs(Base):
    type=attr.ib(default="pgd",validator=attr.validators.in_({'pgd', 'defaultpgd', 'tpgd', 'autoattack','FAST'}))
    eps=attr.ib(default=0.031)
    lr=attr.ib(type=float, default=None)
    num_steps=attr.ib(default=7)
    use_best=attr.ib(default=True)
    fast_alpha=attr.ib(default=10.0/255.0,
                       type=float)
    init_dist=attr.ib(default="uniform",validator=attr.validators.in_({"uniform","ortho","epsilon-edge"}))
    one_t_start=attr.ib(default=None,type=Optional[int])

    def make(self, model, loss=None, num_classes=10):
        if self.type == "pgd":
            opt = PGD(
                model,
                loss=loss,
                eps=self.eps,
                alpha=self.lr,
                steps=self.num_steps,
                random_start=True,
                use_best=self.use_best,init_dist=self.init_dist)
        elif self.type == "defaultpgd":
            opt = defaultPGD(
                model,
                eps=self.eps,
                alpha=self.lr,
                steps=self.num_steps,
                random_start=True,init_dist=self.init_dist,one_t_start=self.one_t_start)
        elif self.type == "tpgd":
            opt = TPGD(
                model,
                eps=self.eps,
                alpha=self.lr,
                steps=self.num_steps)
        elif self.type == "FAST":
            opt=FFGSM(
                model,eps=self.eps,alpha=self.fast_alpha
            )
        elif self.type == "autoattack":
            opt = AutoAttack(model, norm='Linf', eps=self.eps, version='standard', n_classes=num_classes)
        else:
            raise ValueError("Invalid attack args", self)
        return opt

    def __attrs_post_init__(self):
        if self.lr is None:
            self.lr = 2.5 * self.eps / self.num_steps


ATTACK_DEFAULT = {'eps': 0.031, 'num_steps': 7, 'lr': 0.007843137}
TEST_ATTACK_DEFAULT = {'eps': 0.031, 'num_steps': 20}


@attr.s() #(auto_attribs=True)
class Hpars(Base):
    gpus = attr.ib(type=int, default=1, metadata=dict(help='number of gpus'))
    exp_dir = attr.ib(type=str, default='EXP', metadata=dict(help='directory of experiment'))
    project = attr.ib(type=str, default='acat')
    wandb_entity = attr.ib(type=str, default="ANON")
    exp_name = attr.ib(type=str, default='debug', metadata=dict(help='name of experiment'))
    ckpt_path = attr.ib(type=str, default=None)

    # See https://docs.wandb.ai/guides/track/advanced/resuming
    wandb_resume_id = attr.ib(type=str, default=None)

    val_adv = attr.ib(type=bool, default=True)
    val_split = attr.ib(type=Union[int, float], default=0.2)
    batch_size = attr.ib(type=int, default=128)    
    num_of_attacks = attr.ib(type=int, default=10)
    danskinattack = attr.ib(type=bool, default=False)
    normalize_danskin=attr.ib(type=bool,default=False)
    jaxed = attr.ib(type=bool, default=False)
    max_ensemble = attr.ib(type=bool, default=False)
    
    # epochs used for step-size scheduler while max_epochs can be overwritten to early stop training (e.g. for AT)
    epochs = attr.ib(type=int, default=200, metadata=dict(help='number of epochs'))
    max_epochs = attr.ib(type=int, default=None)

    # hpars for landscape plot
    attack_per_step = attr.ib(type=bool, default=False)
    max_batches = attr.ib(type=int, default=5)
 
    # Logging
    progress_bar_refresh_rate = attr.ib(type=int, default=20, metadata=dict(help='Refresh rate of progress bar'))
    val_rate = attr.ib(type=int, default=5)
    log_verbose = attr.ib(type=bool, default=False)
    profiler = attr.ib(type=str, default=None, validator=attr.validators.in_({None, 'simple', 'pytorch-mem'}))

    # For debugging
    limit_train_batches = attr.ib(type=float, default=1.0)
    limit_val_batches = attr.ib(type=float, default=1.0)
    limit_test_batches = attr.ib(type=float, default=1.0)
    fast_dev_run = attr.ib(type=int, default=False)  # pt-lightning config
    acumm_grad=attr.ib(type=int,default=1)
    patience=attr.ib(type=int,default=5)
    log_every=attr.ib(type=int,default=50)
 
    # Dataset and transformations
    dataset = attr.ib(type=str, default="cifar10", validator=attr.validators.in_({'cifar10',"cinic10"}))
    num_workers = attr.ib(type=int, default=5)
    augment = attr.ib(type=bool, default=False)
    normalize = attr.ib(type=bool, default=True)

    # Model configs
    model = attr.ib(type="str", default="resnet50")
    model_opt = attr.ib(type=OptArgs, default={}, converter=OptArgs.to_cls)
    attack_opt = attr.ib(type=AttackArgs, default=ATTACK_DEFAULT, converter=AttackArgs.to_cls)
    test_attack_opt = attr.ib(type=AttackArgs, default=TEST_ATTACK_DEFAULT, converter=AttackArgs.to_cls)
    batch_norm = attr.ib(default=True)
    celu = attr.ib(default=True)
    def __attrs_post_init__(self):
        self.out_dir = os.path.join(self.exp_dir, self.exp_name)

        if self.max_epochs is None:
            self.max_epochs = self.epochs
    
