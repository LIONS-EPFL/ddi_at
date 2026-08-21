import torch
import torch.nn as nn
from acat.utils import print_once

from torchattacks.attack import Attack
import numpy as np


class PGD(Attack):
    r"""
    PGD in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'
    [https://arxiv.org/abs/1706.06083]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 0.3)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of steps. (Default: 40)
        random_start (bool): using random initialization of delta. (Default: True)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.PGD(model, eps=8/255, alpha=1/255, steps=40, random_start=True)
        >>> adv_images = attack(images, labels)

    """
    def __init__(self, model, eps=0.3,
                 alpha=2/255, steps=40, random_start=True,init_dist="uniform",one_t_start=None):
        super().__init__("PGD", model)
        assert init_dist in {"uniform","ortho","epsilon-edge"}
        self.init_dist=init_dist
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self._supported_mode = ['default', 'targeted']
        self.one_t_start=None

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self._targeted:
            target_labels = self._get_target_label(images, labels)

        loss = nn.CrossEntropyLoss()

        adv_images = images.clone().detach()

        if self.random_start:
            print_once(f"Random start{self.random_start} with init {self.init_dist}")
            if self.init_dist =="ortho":
                # an attempt at diverse madry: orthogonal init, scaled to epsilon cube around image
                Bn=adv_images.shape[0]
                dim=np.prod(adv_images.shape[1:])
                ortogonal_possible=Bn<=dim
                # as long as C*H*W >=B*N_attack, we can just do the lazy thing
                if ortogonal_possible:
                    noise = torch.nn.init.orthogonal_(torch.empty_like(adv_images))
                else:
                    raise NotImplementedError("Due to the batch size we can't orthogonalize across the B*n_a times  H*W*C matrix, need to do it for each image individually which isn't done yet")
                # reshape to make sure the image is set
                # attempt to project to hypercube while retaining orthogonality:
                # 1. shifts don't affect orthogonality
                # 2. scales don't effect orthogonality
                # shift up to make sure we are in [0,infty]
                noip=noise.permute(1,0,2,3).reshape(noise.shape[1],-1)
                noise=noise-noip.min(-1).values.reshape(1,noise.shape[1],1,1)
                # update noip
                noip=noip-noip.min(-1,keepdim=True).values
                # scale down to make sure we are in [0,1]
                noise=noise/(1e-19+noip.max(-1).values.reshape(1,noise.shape[1],1,1))
                # scale and shift *again*, now to make sure we are in [-eps, eps] now
                noise=(noise-0.5)*2*self.eps
            elif self.init_dist =="epsilon-edge":
                noise=torch.sign(torch.empty_like(adv_images).uniform_(-1.0,1.0))*self.eps
            else:
                # Normal madry:
                # Starting at a uniformly random point
                noise=torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = adv_images + noise
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        alpha_start=self.alpha
        for step in range(self.steps):
            if self.one_t_start is not None:
                alpha=alpha_start/max(1,step-self.one_t_start+1)
            else:
                alpha=alpha_start
            adv_images.requires_grad = True
            outputs = self.model(adv_images)

            # Calculate loss
            if self._targeted:
                cost = -loss(outputs, target_labels)
            else:
                cost = loss(outputs, labels)

            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]


            adv_images = adv_images.detach() + alpha*grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images
