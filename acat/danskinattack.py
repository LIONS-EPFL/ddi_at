import torch
import numpy as np
import jax.numpy as jnp

from acat.descentdir import SimplexMin


def flatten_tensors(tensors):
    flattened_tensors = []
    for i, g in enumerate(tensors):
        flattened_tensors.append(g.view(-1))
    return torch.cat(flattened_tensors)


def max_ensemble_attack(model, batch, num_of_attacks=10):
    img, label = batch
    batch_size = img.shape[0]
    img = img.repeat(num_of_attacks, *(1,)*img.ndim)
    img = img.view(-1, *img.shape[2:])
    label = label.repeat(num_of_attacks, *(1,)*label.ndim)
    label = label.view(-1, *label.shape[2:])
    batch = img, label
    img, label = model.attack(batch)
    logits = model(img)
    losses = model.loss(logits, label, reduction="none")
    losses_shape = losses.shape[1:]
    losses = losses.view(num_of_attacks, batch_size, *losses_shape)
    losses, indices = torch.max(losses, dim=0)
    # logits_shape = logits.shape[1:]
    # logits = logits.view(num_of_attacks, batch_size, *logits_shape)
    return losses# , logits[indices]


class DanskinAttack(object):
    def __init__(self, normalize=True):
        self.normalize = normalize

    def __call__(self, model, batch, num_of_attacks=10):
        losses = self.get_losses(model, batch, num_of_attacks=num_of_attacks)
        M, MMT = self.get_gradients(model, losses)
        #MMT = MMT.cpu().detach().numpy()
        y = self.solve_QP(MMT, num_of_attacks=num_of_attacks)
        return self.get_descent_direction(y, M)
        
    def solve_QP(self, MMT, num_of_attacks=10):
        init = torch.ones_like(MMT[0])/num_of_attacks
        step_size = 1/torch.linalg.norm(MMT, ord=2)
        maxiter = 100
        simplex_min = SimplexMin(init, step_size, maxiter, lambda y: torch.matmul(MMT, y))
        y = simplex_min.optimize()
        return y

    def get_descent_direction(self, y, M):
        y = torch.tensor(np.array(y), device=M.device)
        direction =  M.transpose(0, 1).matmul(y)
        return direction/torch.linalg.norm(direction)

    def get_losses(self, model, batch, num_of_attacks=10):
        img, label = batch
        batch_size = img.shape[0]
        img = img.repeat(num_of_attacks, *(1,)*img.ndim)
        img = img.view(-1, *img.shape[2:])
        label = label.repeat(num_of_attacks, *(1,)*label.ndim)
        label = label.view(-1, *label.shape[2:])
        batch = img, label
        img, label = model.attack(batch)
        logits = model(img)
        losses = model.loss(logits, label, reduction="none")
        losses_shape = losses.shape[1:]
        losses = losses.view(num_of_attacks, batch_size, *losses_shape)
        losses = losses.mean(1)
        model.log('train_avg_adv_loss', losses.mean(), prog_bar=True)
        return losses

    def get_gradients(self, model, losses):
        M = []
        for loss in losses:
            grads = torch.autograd.grad(loss, model.model.parameters(), retain_graph=True)
            grads = flatten_tensors(grads)
            M.append(grads)
        M = torch.vstack(M)
        MMT = M.matmul(M.transpose(0, 1))
        return M, MMT
