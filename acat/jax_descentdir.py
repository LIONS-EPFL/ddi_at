import jax
from jax import jit
import jax.numpy as jnp
import jax.lax as lax
from functools import partial


class SimplexMinJax():

    def __init__(self, MMT,num_of_attacks, maxiter=100):

        # M is intended to be number of number of attacks,parametrs=> MMT is meant to be num_attack,num_attack
        #MMT = jnp.matmul(M,jnp.transpose(M))
        MMT = MMT
        gradient = jit(lambda y: jnp.dot(MMT,y))
        # create 1/num vector on whatever device the tensor lives on
        y_init = jax.device_put(jnp.ones(num_of_attacks),MMT.device_buffer.device())/num_of_attacks
        step_size = 1/jnp.linalg.norm(MMT, ord=2)

        self.step_size = step_size
        self.maxiter = maxiter
        self.y = y_init
        self.y_prev = y_init
        self.dim = len(y_init)
        self.gradient = gradient

        self.store = []

    def step(self, k):
        extrapolation_param = k/(k+3)
        extra = self.y + extrapolation_param * (self.y - self.y_prev)
        
        self.y_prev = self.y
        self.y = type(self).project(extra - self.step_size*self.gradient(extra), 1, self.dim)
    
    def optimize(self):
        for i in range(self.maxiter):
            self.step(i)
            self.store.append(jnp.dot(self.y, self.gradient(self.y)))
        return self.y

    @partial(jit, static_argnums=(1, 2))
    def project(v, radius, dim):
        mu = lax.sort(v)
        cumul_sum = jnp.divide(
            lax.cumsum(mu, reverse=True) - radius, jnp.arange(dim, 0, -1))
        rho = jnp.amin(jnp.where(mu > cumul_sum, jnp.arange(dim), dim))
        theta = cumul_sum[rho]
        return jnp.maximum(v - theta, 0)
