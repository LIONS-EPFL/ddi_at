import os
import numpy as np
from acat.model import Model
from pathlib import Path
import torch
import wandb
from acat.config import Hpars
from acat.model import *
from acat.runner import get_train_transforms, base_dataset
from acat.danskinattack import max_ensemble_attack
from acat.jax_danskinattack import JaxDanskinAttack
from torchvision.datasets.cifar import CIFAR10
from pl_bolts.datamodules import CIFAR10DataModule, MNISTDataModule
from tqdm import tqdm
from sacred import Experiment, SETTINGS
import time
from pytorch_lightning import Trainer, seed_everything

SETTINGS['HOST_INFO']['INCLUDE_GPU_INFO'] = False


def store_direction_in_grad(batch, model, mode):
    if mode == 'danskin':
        danskinattack = JaxDanskinAttack()
        descent_direction = danskinattack(
            model, batch, num_of_attacks=model.config.num_of_attacks)
        for e in danskinattack.simplex_min.store:
            wandb.log({'inner_objective': e})
        model.update_gradients_manually(model.model.parameters(),
                                        descent_direction)

    elif mode == 'max_ensemble':

        losses = max_ensemble_attack(
            model, batch, num_of_attacks=model.config.num_of_attacks)
        loss = losses.mean()

        loss.backward()
    else:
        loss = model._compute_loss(batch,
                                   0,
                                   stage=Stage.train,
                                   type=LossType.adv,
                                   reduction="mean",
                                   return_logits=False)

        loss.backward()


def step(model, micro_lr):
    with torch.no_grad():
        for p in model.parameters():
            p.add_(-micro_lr * p.grad / torch.norm(p.grad))


def evaluate_along_grad(model, batch, micro_lr, K, attack=False):
    values = []
    seed_everything(0)
    for i in tqdm(range(K)):
        if attack:
            tqdm.write("ATTACKING")
            batch = model.attack(batch)
            tqdm.write("FINISHED ATTACK")
        x, y = batch
        logits = model(x)
        loss = model.loss(logits, y, reduction="mean")
        wandb.log({"loss" : loss.item(), "step_size": i*micro_lr})
        step(model, micro_lr)
    return values


ex = Experiment('acat')


@ex.automain
def run(h):
    # Configs
    hpars: Hpars = Hpars.to_cls(h)
    ckpt = hpars.ckpt_path
    attack = hpars.attack_per_step


    get_dataset = base_dataset[hpars.dataset]
    class_names = get_dataset(f"data/{hpars.dataset}",
                              train=True,
                              download=True).classes   

    model = Model(dict(), classes=class_names)
    seed_everything(0)
    if ckpt != None:
        model.load_from_checkpoint(ckpt)
    else:
        model = Model(
        hpars,
        classes=class_names,
    )
    model.model.cuda()
    device = next(model.model.parameters()).device
    print("--- DEVICE: ", device)

    if hpars.dataset == 'mnist':
        batch = torch.load('../tests/images.pt', map_location=device), torch.load('../tests/labels.pt', map_location=device)

    #print(next(model.model.parameters()))
    model.metrics = {m:model.metrics[m].to(device) for m in model.metrics}
    if hpars.danskinattack:
        mode = 'danskin'
    elif hpars.max_ensemble:
        mode = 'max_ensemble'
    else:
        mode = 'madry'

    wandb.init(project=hpars.project, name=hpars.exp_name+"/"+ hpars.dataset +"/"+ mode)

    wandb.config.update({"mode": mode, "attack_per_step": attack})


    K = 100
    micro_lr = hpars.model_opt.lr / (K)
    axis = [k * micro_lr for k in range(K)]
    

    batch = batch[0].to(device), batch[1].to(device)
    print("\t### MODE {} ###".format(mode))
    store_direction_in_grad(batch, model, mode)
    print("\t\t -- Computed Descent direction --")
    values = evaluate_along_grad(model, batch,
                                    micro_lr, K, attack)

    #revert back to checkpoint
    step(model, -hpars.model_opt.lr)
    model.zero_grad()
