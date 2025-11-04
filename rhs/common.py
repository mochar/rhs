from typing import Any
from collections import OrderedDict

import numpyro
import numpyro.distributions as dist
from numpyro.primitives import Message
import jax.numpy as jnp
from jax.typing import ArrayLike

type TraceType = OrderedDict[str, Message]

exp_transform = dist.transforms.ExpTransform()
pos_const = dist.constraints.softplus_positive

def normal_site(name, inits: dict[str, Any], log: bool = False) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
    loc = numpyro.param(f'locs.{name}', inits[f'locs.{name}'])
    assert loc is not None
    scale = numpyro.param(
        f'scales.{name}',
        inits[f'scales.{name}'],
        constraint=pos_const)
    assert scale is not None
    D = dist.LogNormal if log else dist.Normal
    return numpyro.sample(name, D(loc, scale)), loc, scale

def lognormal_site(name, inits: dict[str, Any]) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
    return normal_site(name, inits, log=True)

def to_reg_lambda(tau2, lambda2, c2):
    return jnp.sqrt(lambda2 * c2 / (c2 + tau2 * lambda2))

