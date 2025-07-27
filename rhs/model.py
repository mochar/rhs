from dataclasses import dataclass, field
from typing import Any
from collections.abc import Callable

import numpyro
import numpyro.distributions as dist
from numpyro import handlers
import jax.numpy as jnp
from jax.typing import ArrayLike

"""
reparam:
 - base
 - invg-invg
   - cen
   - dec1
   - dec2
   - dec
 - g-g
   - cen
   - dec
guide:
  - factorized
  - mvn
  - mxn
"""

exp_transform = dist.transforms.ExpTransform()
pos_const = dist.constraints.softplus_positive

def lognormal_site(name, loc_init=0., scale_init=.1) -> ArrayLike:
    loc = numpyro.param(f'locs.{name}', jnp.array(loc_init))
    scale = numpyro.param(
        f'scales.{name}',
        jnp.array(scale_init),
        constraint=pos_const)
    return numpyro.sample(name, dist.LogNormal(loc, scale))

def to_reg_lambda(tau2, lambda2, c2):
    return jnp.sqrt(lambda2 * c2 / (c2 + tau2 * lambda2))


@dataclass
class Configuration:
    X: ArrayLike
    Y: ArrayLike
    tau_scale: float
    c_df: float
    c_scale: float
    inits: dict[str, Any] = field(default_factory=dict)
    N: int = field(init=False)
    D: int = field(init=False)
    model: Callable = field(init=False)
    guide: Callable = field(init=False)

    def __post_init__(self):
        self.N, self.D = self.X.shape

        # Model
        def model():
            noise = numpyro.sample('noise', dist.Exponential(1.))
            c2_aux = numpyro.sample('c2_aux', dist.InverseGamma(self.c_df*0.5, self.c_df*0.5))
            c = self.c_scale * jnp.sqrt(c2_aux)

            tau = numpyro.sample('tau', dist.HalfCauchy(scale=self.tau_scale))
            with numpyro.plate('d', self.D):
                lambda_ = numpyro.sample('lambda', dist.HalfCauchy(scale=1.))
                lambda_reg = to_reg_lambda(tau**2, lambda_**2, c*c)
                
                coef_dec = numpyro.sample('coef_dec', dist.Normal(0., 1.))
                coef = numpyro.deterministic('coef', coef_dec * tau * lambda_reg)
                
            mean = self.X @ coef
            with numpyro.plate('n', self.N):
                numpyro.sample('y', dist.Normal(mean, noise), obs=self.Y)
        self.model = model

        # Guide
        def guide():
            lognormal_site('noise')
            lognormal_site('c2_aux')
            lognormal_site('tau')
                
            with numpyro.plate('d', self.D):
                lognormal_site('lambda')
                        
                coef_loc = numpyro.param('locs.coef', self.inits['locs.coef_dec'])
                coef_scale = numpyro.param('scales.coef_dec',
                                           self.inits['scales.coef_dec'],
              	                       constraint=pos_const)
                numpyro.sample('coef_dec', dist.Normal(coef_loc, coef_scale))
        self.guide = guide

        # Populate init values
        t = handlers.trace(handlers.seed(self.model, 0)).get_trace()
        for name, node in t.items():
            if node['type'] != 'sample':
                continue
            shape = node['fn'].shape()
            self.inits[f'locs.{name}'] = jnp.zeros(shape)
            self.inits[f'scales.{name}'] = jnp.full(shape, .1)
