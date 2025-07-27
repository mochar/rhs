from collections.abc import Callable

import numpyro
from jax import numpy as jnp
from jax.typing import ArrayLike


def lognormal_mode(self) -> ArrayLike:
    return jnp.exp(self.loc - jnp.square(self.scale))

numpyro.distributions.LogNormal.mode = property(lognormal_mode)


def gamma_mode(self) -> ArrayLike:
    return jnp.clip((self.concentration - 1) / self.rate, min=0)

numpyro.distributions.Gamma.mode = property(gamma_mode)

