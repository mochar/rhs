from dataclasses import dataclass, field
from typing import Any
from collections.abc import Callable

import numpyro
import numpyro.distributions as dist
from numpyro import handlers
import jax.numpy as jnp
from jax import random
from jax.typing import ArrayLike

from .common import to_reg_lambda, lognormal_site
from .reparam import Reparam
from .guide import GuideStructure


@dataclass
class Configuration:
    X: ArrayLike
    Y: ArrayLike
    reparam_tau: Reparam
    reparam_lambda: Reparam
    structure: GuideStructure
    coef_dec: bool
    tau_scale: float
    c_df: float
    c_scale: float
    noise_scale: float = 1.
    N: int = field(init=False)
    D: int = field(init=False)
    model: Callable = field(init=False)
    guide: Callable = field(init=False)
    inits: dict[str, Any] = field(default_factory=dict)
    coef_name: str = field(init=False)

    def __post_init__(self):
        self.N, self.D = self.X.shape
        self.coef_name = 'coef' + ('_dec' if self.coef_dec else '')

        # Model
        def model():
            noise = numpyro.sample('noise', dist.Exponential(self.noise_scale))
            c2_aux = numpyro.sample('c2_aux', dist.InverseGamma(self.c_df*0.5, self.c_df*0.5))
            c = numpyro.deterministic('c', self.c_scale * jnp.sqrt(c2_aux))

            if self.reparam_tau is None:
                tau = numpyro.sample('tau', dist.HalfCauchy(scale=self.tau_scale))
            else:
                tau = self.reparam_tau.model('tau', self.tau_scale)

            with numpyro.plate('d', self.D):
                if self.reparam_lambda is None:
                    lambda_ = numpyro.sample('lambda', dist.HalfCauchy(scale=1.))
                else:
                    lambda_ = self.reparam_lambda.model('lambda', 1.)
                    
                lambda_reg = numpyro.deterministic('lambda_reg', to_reg_lambda(tau**2, lambda_**2, c*c))

                if self.coef_dec:
                    coef_dec = numpyro.sample('coef_dec', dist.Normal(0., 1.))
                    coef = numpyro.deterministic('coef', coef_dec * tau * lambda_reg)
                else:
                    coef = numpyro.sample('coef', dist.Normal(0., tau * lambda_reg))

            mean = self.X @ coef
            with numpyro.plate('n', self.N):
                numpyro.sample('y', dist.Normal(mean, noise), obs=self.Y)
        self.model = model

        # SVI inits
        self.inits = self.make_inits()
        
        # Guide
        def guide():
            lognormal_site('noise', self.inits)
            c2_aux, *_ = lognormal_site('c2_aux', self.inits)
            c = self.c_scale * jnp.sqrt(c2_aux)

            if self.reparam_tau:
                tau, *_ = self.reparam_tau.guide('tau', self.tau_scale, self.inits)
            else:
                tau, *_ = lognormal_site('tau', self.inits)

            self.structure.guide(self.D, self.reparam_lambda, self.coef_dec, tau, c, self.inits)

        self.guide = guide

    @property
    def name(self):
        tau_reparam_name = self.reparam_tau.name if self.reparam_tau else 'base'
        lambda_reparam_name = self.reparam_lambda.name if self.reparam_lambda else 'base'
        return f'{self.structure.name}_{tau_reparam_name}_{lambda_reparam_name}'

    def make_inits(self) -> dict[str, Any]:
        t = handlers.trace(handlers.seed(self.model, 0)).get_trace()
        inits: dict[str, Any] = {}
        for name, node in t.items():
            if node['type'] != 'sample':
                continue
            shape = node['fn'].shape()
            inits[f'locs.{name}'] = jnp.zeros(shape)
            inits[f'scales.{name}'] = jnp.full(shape, .1)
            
        inits['chols.coef'] = dist.LKJCholesky(self.D).sample(random.key(0))
        return inits
    
