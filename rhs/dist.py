import numpyro
from jax import numpy as jnp

def lognormal_mode(self):
    return jnp.exp(self.loc - jnp.square(self.scale))

numpyro.distributions.LogNormal.mode = property(lognormal_mode)


def gamma_mode(self):
    return jnp.clip((self.concentration - 1) / self.rate, min=0)

numpyro.distributions.Gamma.mode = property(gamma_mode)

