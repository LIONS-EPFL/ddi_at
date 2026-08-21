import torch
import jax
import jax.dlpack
from functorch import make_functional_with_buffers, grad, vmap
from acat.jax_descentdir import SimplexMinJax


def gram(p):
    return torch.mm(p.view(p.shape[0], -1), p.view(p.shape[0], -1).transpose(0, 1))

mult = vmap(lambda a, b: a * b, in_dims=(0, 0))

def torch2jax(t):
    return jax.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(t))


def jax2torch(t):
    return torch.utils.dlpack.from_dlpack(jax.dlpack.to_dlpack(t))


def jax_max_ensemble_attack(model, batch, num_of_attacks=10):
    img, label = batch
    batch_size = img.shape[0]
    img = img.repeat(num_of_attacks, *(1,) * img.ndim)
    img = img.view(-1, *img.shape[2:])
    label = label.repeat(num_of_attacks, *(1,) * label.ndim)
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
    return losses  # , logits[indices]


class JaxDanskinAttack(object):
    def __init__(self, normalize=True):
        self.normalize = normalize
        self.simplex_min = None

    def __call__(self, model, batch, num_of_attacks=10):
        att_batch = self.get_attacks(model, batch, num_of_attacks=num_of_attacks)
        tuple_of_per_parameter_gradients = self.get_grad_per_attack(model, att_batch)

        
        MMT = torch.zeros(num_of_attacks, num_of_attacks, device=next(model.model.parameters()).device)
        for p in tuple_of_per_parameter_gradients:
            MMT.add_(gram(p))


        
        y = self.solve_QP(MMT)
        assert torch.is_tensor(y)
        return self.get_descent_direction(y, tuple_of_per_parameter_gradients)

    def solve_QP(self, MMT):
        assert len(MMT.shape) == 2

        num_of_attacks = MMT.shape[0]
        # from pytorch to jax
        MMT = torch2jax(MMT.detach())
        self.simplex_min = SimplexMinJax(MMT, num_of_attacks, maxiter=1000)
        y = self.simplex_min.optimize()
        # back to pytorch
        y = jax2torch(y)
        return y

    def get_descent_direction(self, y, M):
        per_param_direction = []
        nrm = 0
        
        for grads_p in M:
            grad = torch.sum(mult(y, grads_p), dim=0)
            nrm += torch.dot(grad.view(-1), grad.view(-1))
            per_param_direction.append(grad)
        if self.normalize:
            per_param_direction = [ grad / torch.sqrt(nrm) for grad in per_param_direction]
        return per_param_direction

    def get_attacks(self, model, batch, num_of_attacks=10):
        
        img, label = batch
        batch_size = img.shape[0]

        img = img.repeat(num_of_attacks, *(1,) * img.ndim)
        label = label.repeat(num_of_attacks, *(1,) * label.ndim)

        img = img.view(-1, *img.shape[2:])
        label = label.view(-1, *label.shape[2:])

        

        att_imgs, att_labels = model.attack((img, label))
        att_imgs = att_imgs.view(num_of_attacks, batch_size, *att_imgs.shape[1:])
        att_labels = att_labels.view(num_of_attacks, batch_size, *att_labels.shape[1:])

        
        return att_imgs, att_labels

    def get_grad_per_attack(self, model, att_batch):
        
        att_imgs, att_labels = att_batch
        model.model.eval()
        func, params, buffers = make_functional_with_buffers(model.model, disable_autograd_tracking=True)
        loss = lambda params, buffers, imgs, labels: model.loss(
            func(params, buffers, imgs), labels, reduction="mean"
        )
        
        per_parameter_grad_stack = vmap(grad(loss), in_dims=(None, None, 0, 0))(
            params, buffers, att_imgs, att_labels
        )

        

        model.model.train()
        return per_parameter_grad_stack
