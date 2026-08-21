from jax import jit
import jax.numpy as jnp
import jax.lax as lax
from functools import partial
import torch


class SimplexMin():

    def __init__(self, y_init, step_size, maxiter, gradient):
        
        self.step_size = step_size
        self.maxiter = maxiter
        self.y = y_init
        self.y_prev = y_init
        self.dim = len(y_init)
        self.gradient = gradient

        self.store = []

    def step(self, k: int):
        with torch.no_grad():
            extrapolation_param = k/(k+3)
            extra = self.y + extrapolation_param * (self.y - self.y_prev)
        
            self.y_prev = self.y
            self.y = SimplexMin.project(extra - self.step_size*self.gradient(extra), 1, self.dim)
    
    def optimize(self):
        for i in range(self.maxiter):
            self.step(i)
        return self.y

    @torch.jit.script
    def project(v, radius: int, dim: int):
        mu, _ = torch.sort(v)
        cumul_sum = torch.divide(
            torch.flip(torch.cumsum(torch.flip(mu, [0]), dim=0), [0]) - radius, torch.arange(dim, 0, -1,device=mu.device))
        rho = torch.argmin(torch.where(mu > cumul_sum, torch.arange(dim,device=mu.device), dim))
        theta = cumul_sum[rho]
        return torch.maximum(v - theta, torch.zeros_like(v))
