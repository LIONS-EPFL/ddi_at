import torch
import numpy as np
import wandb

from acat.danskinattack import DanskinAttack, flatten_tensors, max_ensemble_attack
from acat.model import overwrite_gradients
from acat.plotter import store_direction_in_grad, evaluate_along_grad, step


class DummyModel(torch.nn.Module):
    def __init__(self, input_size, num_classes, h_layers=False):
        super().__init__()
        self.num_classes = num_classes
        if not h_layers:
            self.model = torch.nn.Linear(input_size, self.num_classes)
        else:
            self.model = torch.nn.Sequential(torch.nn.Linear(input_size, 7),
                                        torch.nn.Linear(7, 9),
                                        torch.nn.Linear(9, self.num_classes))


    def forward(self, img):
        img = img.view(img.shape[0], -1)
        return self.model(img)

    def attack(self, batch):
        return batch

    def loss(self, logits, label, reduction="none"):
        if reduction=="mean":
            return logits.mean()
        return logits.mean(1)
    def log(self, *kwargs, **args):
        print("LOGGED")

    def update_gradients_manually(self, parameters, grad):
        overwrite_gradients(parameters, grad)
    def _compute_loss(self, batch, batch_idx, 
        stage, 
        type, 
        reduction="mean", 
        return_logits=False
    ):
        x, y = batch
        logits = self(x)

        # Normal cross entropy
        loss = self.loss(logits, y, reduction=reduction)
        return loss


def test_danskinattack_get_losses():
    batch_size = 128
    num_channels = 3
    img_dim = 32
    num_classes = 10
    num_of_attacks = 10
    input_size = num_channels * img_dim * img_dim

    img = torch.randn(batch_size, num_channels, img_dim, img_dim)
    labels = torch.randn(batch_size)
    batch = img, labels
    model = DummyModel(input_size, num_classes)

    danskinattack = DanskinAttack()
    losses = danskinattack.get_losses(model, batch, num_of_attacks=num_of_attacks)
    assert losses.shape == torch.Size((num_of_attacks,))


def test_danskinattack_get_gradients():
    batch_size = 128
    num_channels = 3
    img_dim = 32
    num_classes = 10
    num_of_attacks = 10
    input_size = num_channels * img_dim * img_dim
    num_params = input_size*num_classes + num_classes

    img = torch.randn(batch_size, num_channels, img_dim, img_dim)
    labels = torch.randn(batch_size)
    batch = img, labels
    model = DummyModel(input_size, num_classes)

    danskinattack = DanskinAttack()
    losses = danskinattack.get_losses(model, batch, num_of_attacks=num_of_attacks)
    M, get_subproblem_grad = danskinattack.get_gradients(model, losses)
    assert M.shape == torch.Size((num_of_attacks, num_params))


def test_danskinattack_solve_QP():
    num_of_attacks = 4
    M = [[1, 1], [1, 2], [2, 2], [2, 1]]
    MMT =1.0*np.array(M) @ np.array(M).T
    danskinattack = DanskinAttack()
    y = danskinattack.solve_QP(torch.tensor(MMT).double(), num_of_attacks=num_of_attacks)
    y = np.array(y)
    assert np.all(y == [1, 0, 0, 0])


def test_danskinattack():
    batch_size = 128
    num_channels = 3
    img_dim = 32
    num_classes = 10
    num_of_attacks = 10
    input_size = num_channels * img_dim * img_dim
    num_params = input_size*num_classes + num_classes

    img = torch.randn(batch_size, num_channels, img_dim, img_dim)
    labels = torch.randn(batch_size)
    batch = img, labels
    model = DummyModel(input_size, num_classes)

    danskinattack = DanskinAttack()
    d = danskinattack(model, batch, num_of_attacks=num_of_attacks)
    d = np.array(d)
    assert d.shape == (num_params,)


def test_overwrite_gradients():
    input_size = 13
    num_classes = 10
    batch_size = 16
    num_of_attacks = 3

    img = torch.randn(batch_size, input_size)
    labels = torch.randn(batch_size)
    batch = img, labels

    model = DummyModel(input_size, num_classes, h_layers=True)

    danskinattack = DanskinAttack()
    d = danskinattack(model, batch, num_of_attacks=num_of_attacks)

    #for testing purposes replace d by 1s, comment out
    d = torch.ones_like(d)

    overwrite_gradients(model.parameters(), d)
    
    for p in model.parameters():
        assert p.grad.shape == p.shape
        assert np.all(np.array(p.grad) == np.array(torch.ones_like(p.grad)))


def test_max_ensemble_attack():
    batch_size = 128
    num_channels = 3
    img_dim = 32
    num_classes = 10
    num_of_attacks = 10
    input_size = num_channels * img_dim * img_dim
    num_params = input_size*num_classes + num_classes

    img = torch.randn(batch_size, num_channels, img_dim, img_dim)
    labels = torch.randn(batch_size)
    batch = img, labels
    model = DummyModel(input_size, num_classes)

    losses = max_ensemble_attack(model, batch, num_of_attacks=num_of_attacks)
    losses = np.array(losses.detach())
    assert losses.shape == (batch_size,)


def test_danskin_plotter():
    batch_size = 128
    num_channels = 3
    img_dim = 32
    num_classes = 10
    num_of_attacks = 10
    input_size = num_channels * img_dim * img_dim
    num_params = input_size*num_classes + num_classes

    img = torch.randn(batch_size, num_channels, img_dim, img_dim)
    labels = torch.randn(batch_size)
    batch = img, labels
    model = DummyModel(input_size, num_classes)

    K = 100
    lr = 0.01
    micro_lr = lr / (K)
    axis = [k * micro_lr for k in range(K)]
    batch_idx = 0
    attack_batch = model.attack(batch)
    wandb.init(project="acat", name="loss-curve-test")
    mode = 'danskin'
    for mode in ['danskin', 'madry']:
        x, y = batch
        logits = model(x)
        loss = model.loss(logits, y, reduction="mean")
        print("MODE ----- {} -----".format(mode))
        print("\t--LOSS:  {}".format(loss.item()))
        store_direction_in_grad(batch, batch_idx, model, mode)
        print("\t-- Computed Descent direction --")
        values = evaluate_along_grad(model, attack_batch, batch_idx,
                                        micro_lr, K)
        step(model, -lr)
        model.zero_grad()
        curve = [[x, y] for x, y in zip(axis, values)]
        table = wandb.Table(data=curve, columns=["step", "loss"])
        wandb.log({
            "{0} - batch {1}".format(mode, batch_idx):
            wandb.plot.line(table,
                            "step",
                            "loss",
                            title="{0} loss curve - batch {1}".format(
                                mode, batch_idx))
        })