from dataclasses import dataclass, field
from typing import Any
from collections import OrderedDict
from functools import partial

import dill
import numpyro
import numpyro.distributions as dist
from numpyro.primitives import Message
from numpyro import handlers
import jax
import jax.numpy as jnp
from jax import random
from jax.typing import ArrayLike
import optax

from .dist import *


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

def model_root(c_df, c_scale):
    noise = numpyro.sample('noise', dist.Exponential(1.))
    c2_aux = numpyro.sample('c2_aux', dist.InverseGamma(c_df*0.5, c_df*0.5))
    c = c_scale * jnp.sqrt(c2_aux)
    return noise, c

def model_likelihood(X, Y, coef, noise):
    mean = X @ coef
    with numpyro.plate('n', Y.shape[0]):
        numpyro.sample('y', dist.Normal(mean, noise), obs=Y)


type TraceType = OrderedDict[str, Message]
        
@dataclass
class ModelBase:
    X: ArrayLike
    Y: ArrayLike
    tau_scale: float
    c_df: float
    c_scale: float
    inits: dict[str, Any] = field(default_factory=dict)
    D: int = field(init=False)

    def __post_init__(self):
        self.D = self.X.shape[1]
        
        t = handlers.trace(handlers.seed(self.model, 0)).get_trace()
        for name, node in t.items():
            if node['type'] != 'sample':
                continue
            shape = node['fn'].shape()
            self.inits[f'locs.{name}'] = jnp.zeros(shape)
            self.inits[f'scales.{name}'] = jnp.full(shape, .1)
            
    def model(self):
        pass

    def guide(self):
        pass

    def trace_model(self, seed=0) -> TraceType:
        return handlers.trace(handlers.seed(self.model, seed)).get_trace()

    def trace_guide(self, data, seed=0) -> TraceType:
        g = handlers.seed(self.guide, seed)
        g = handlers.substitute(g, data)
        trace = handlers.trace(g).get_trace()
        return trace

    def estimate(self, trace: TraceType, site: str):
        posterior = self.posterior(trace, site)
        match type(posterior):
            case dist.LogNormal:
                return posterior.mode
            case dist.Normal:
                return posterior.mean

    def posterior(self, trace: TraceType, site: str) -> dist.Distribution:
        estimate = partial(self.estimate, trace)
        posterior = partial(self.posterior, trace)
        
        match site:
            case 'c':
                c2_aux = posterior('c2_aux')
                loc = .5 * c2_aux.loc + jnp.log(self.c_scale)
                scale = .5 * c2_aux.scale
                return dist.LogNormal(loc, scale)
            case 'coef':
                tau = estimate('tau')
                lambda_reg = to_reg_lambda(tau**2, estimate('lambda')**2, estimate('c')**2)
                rhs_scale = tau * lambda_reg
                coef_dec = posterior('coef_dec')
                return dist.Normal(coef_dec.loc * rhs_scale, coef_dec.scale * rhs_scale)
            case _:
                return trace[site]['fn']


class ModelSimple(ModelBase):
    def model(self):
        noise, c = model_root(self.c_df, self.c_scale)
        
        tau = numpyro.sample('tau', dist.HalfCauchy(scale=self.tau_scale))
        with numpyro.plate('d', self.D):
            lambda_ = numpyro.sample('lambda', dist.HalfCauchy(scale=1.))
            lambda_reg = to_reg_lambda(tau**2, lambda_**2, c*c)
            
            coef_dec = numpyro.sample('coef_dec', dist.Normal(0., 1.))
            coef = numpyro.deterministic('coef', coef_dec * tau * lambda_reg)
        model_likelihood(self.X, self.Y, coef, noise)
    
    def guide(self):
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

class ModelII(ModelBase):
    pass

class ModelIG(ModelBase):
    pass
            
Model = ModelSimple | ModelII | ModelIG
            
class Trainer:
    def __init__(self, model: Model, seed: int = 0):
        self.model = model
        self.key = random.key(seed)
        self.optim = optax.adam(0.01)
        self.elbo = numpyro.infer.Trace_ELBO(10)
        self.svi = numpyro.infer.SVI(self.model.model, self.model.guide, self.optim, loss=self.elbo)
        self.svi_state = self.svi.init(self.subkey(), init_params=self.model.inits)
        self.gather()
        self.losses = []

    def subkey(self):
        self.key, subkey = random.split(self.key)
        return subkey

    def posterior(self, site: str) -> dist.Distribution:
        return self.model.posterior(self.trace, site)

    def estimate(self, site: str):
        return self.model.estimate(self.trace, site)
    
    def gather(self):
        self.params = self.svi.get_params(self.svi_state)
        self.trace = self.model.trace_guide(self.params)
        tracem = self.model.trace_model()
        self.estimates = {site: self.estimate(site)
                          for t in [self.trace, tracem]
                          for site, node in t.items()
                          if node['type'] in ['sample', 'deterministic'] and not node.get('is_observed', False)}
            
    def train(self, steps):
        try:
            res = self.svi.run(self.subkey(), steps, init_state=self.svi_state,
                               progress_bar=False)
            self.svi_state = res.state
            self.losses.extend(res.losses.tolist())
        finally:
            self.gather()

    def save(self, path):
        with open(path, 'wb') as f:
            dill.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            return dill.load(f)
