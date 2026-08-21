import math
from math import e

import os
from pathlib import Path
from typing import Optional

import numpy as np
from pytorch_lightning.profiler import SimpleProfiler
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from torchvision.datasets.cifar import CIFAR10
from torchvision import transforms
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
#from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import WandbLogger
from pl_bolts.datamodules import CIFAR10DataModule
from pytorch_lightning.profiler import PyTorchProfiler
from pytorch_cinic.dataset import CINIC10
from pytorch_lightning import LightningDataModule

from acat.model import Model
from acat.config import Hpars


from sacred import Experiment, SETTINGS
SETTINGS['HOST_INFO']['INCLUDE_GPU_INFO'] = False

ex = Experiment('acat')


base_dataset = {
    'cifar10': CIFAR10,
    "cinic10":CINIC10
}

class CINIC10DataModule(LightningDataModule):
    def __init__(self,root=os.path.expanduser("~/.datasets/cinic10"),train_batch_size=128,val_batch_size=128,num_workers=None):
        self.root=root
        self.val_batch_size=val_batch_size
        self.train_batch_size = train_batch_size
        self.num_workers=num_workers

    def setup(self, stage: Optional[str] = None):
        train_augmentation = [transforms.ToTensor()]
        train_augmentation += [
            transforms.ColorJitter(.25, .25, .25),
            transforms.RandomRotation(2),
        ]
        val_aug = [transforms.ToTensor()]
        self.splits={p:CINIC10(self.root,partition=p,transform=transforms.Compose(aug),download=True) for p,aug in zip(["train","valid","test"],[train_augmentation,val_aug,val_aug])}
    def train_dataloader(self):
        return DataLoader(self.splits["train"],batch_size=self.train_batch_size,shuffle=True,num_workers=self.num_workers)
    def val_dataloader(self):
        return DataLoader(self.splits["valid"],batch_size=self.val_batch_size,shuffle=False,num_workers=self.num_workers)
    def test_dataloader(self):
        return DataLoader(self.splits["test"],batch_size=self.val_batch_size,shuffle=False,num_workers=self.num_workers)

def get_train_transforms(hpars):
    train_augmentation = []
    
    # Data
    if hpars.dataset in ["cifar10"]:
        image_size = 32
        train_augmentation += [ # base_data is already in tensor form, only need to augment
                transforms.RandomCrop(image_size, padding=4),
                transforms.RandomHorizontalFlip(),
            ]
    
    train_augmentation += [transforms.ToTensor()]
    train_augmentation += [
        transforms.ColorJitter(.25,.25,.25),
        transforms.RandomRotation(2),
    ]

    return transforms.Compose(train_augmentation)


@ex.config
def cfg():
    # Unfortunately `h` cannot prepopulated. Otherwise `__attrs_post_init__` of attrs does not work
    h = {} #Hpars.to_dict(Hpars())


@ex.automain
def run(h):
    # Configs
    hpars: Hpars = Hpars.to_cls(h)

    # Logging
    #logger = TensorBoardLogger(hpars.exp_dir, name=hpars.exp_name)
    wandb_logger = WandbLogger(project=hpars.project, entity=hpars.wandb_entity, name=hpars.exp_name, id=hpars.wandb_resume_id)

    # Save checkpoint using wandb id so we can restart while logging to the same experiment
    ckpt_dir = os.path.join(hpars.out_dir, wandb_logger.experiment.id)
    print("Saving checkpoints to", ckpt_dir)
    ckpt_best_avg = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename='ckpt_{epoch:02d}',
        save_top_k=1,
        monitor='val_adv_acc',
        mode="max",
        every_n_epochs=hpars.val_rate,
    )
    checkpoint_callbacks = [ckpt_best_avg]
    ckpt_cb_last = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename='checkpoint_{epoch:02d}',
        save_top_k=-1,
        save_last=True,
        monitor=None,
        every_n_epochs=40,
    )
    checkpoint_callbacks.append(ckpt_cb_last)

    lr_monitor = LearningRateMonitor()

    # TODO
    if hpars.dataset=="cifar10":
        dm = CIFAR10DataModule(
            data_dir=f"data/{hpars.dataset}",
            val_split=hpars.val_split,
            num_workers=hpars.num_workers,
            batch_size=hpars.batch_size,
            normalize=False, # Happens in the model AFTER attack
        )
        dm.train_transforms = get_train_transforms(hpars)
    elif hpars.dataset=="cinic10":
        dm=CINIC10DataModule(root=f"data/{hpars.dataset}",train_batch_size=hpars.batch_size,val_batch_size=hpars.batch_size,num_workers=hpars.num_workers)
    else:
        raise NotImplementedError(f"Unknown dataset{hpars.dataset}")

    # Get class labels and targets
    get_dataset = base_dataset[hpars.dataset]
    if hpars.dataset=="cifar10":
        class_names = get_dataset(f"data/{hpars.dataset}", train=True, download=True).classes
    else:
        class_names = get_dataset(f"data/{hpars.dataset}", download=True).classes

    # Profiler
    if hpars.profiler == "simple":
        profiler = SimpleProfiler()
    elif hpars.profiler == "pytorch-mem":
        profiler = PyTorchProfiler(
            profile_memory=True,
            sort_by_key="cuda_memory_usage",
        )
    else:
        profiler = None

    # Define models
    model = Model(
        hpars, 
        profiler=profiler,
        classes=class_names,
    )

    # Pytorch lightning Trainer
    trainer = Trainer(
        resume_from_checkpoint=hpars.ckpt_path,
        logger=wandb_logger,
        profiler=profiler,
        gpus=hpars.gpus, 
        max_epochs=hpars.max_epochs, 
        progress_bar_refresh_rate=hpars.progress_bar_refresh_rate, 
        check_val_every_n_epoch=hpars.val_rate,
        callbacks=checkpoint_callbacks + [lr_monitor],
        limit_train_batches=hpars.limit_train_batches,
        limit_val_batches=hpars.limit_val_batches,
        limit_test_batches=hpars.limit_test_batches,
        fast_dev_run=hpars.fast_dev_run,
        log_every_n_steps=min(hpars.log_every,int(50e3/hpars.batch_size)/2) # log at least 2x per epoch
    )

    trainer.fit(model, dm)
    print("Testing with checkpoint:", ckpt_best_avg.best_model_path)
    if len(ckpt_best_avg.best_model_path) != 0:
        trainer.test(datamodule=dm, ckpt_path=ckpt_best_avg.best_model_path)
        wandb_logger.experiment.summary["best_avg_acc_ckpt"] = ckpt_best_avg.best_model_path
