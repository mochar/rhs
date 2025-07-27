from collections import OrderedDict

import dill
import numpyro
import numpyro.distributions as dist
from numpyro.primitives import Message
from numpyro import handlers
import jax.numpy as jnp
from jax import random
from jax.typing import ArrayLike
import optax

from .dist import *
from .model import Configuration, to_reg_lambda

type TraceType = OrderedDict[str, Message]

            
class Trainer:
    def __init__(self, conf: Configuration, seed: int = 0):
        self.conf = conf
        self.key = random.key(seed)
        self.optim = optax.adam(0.01)
        self.elbo = numpyro.infer.Trace_ELBO(10)
        self.svi = numpyro.infer.SVI(self.conf.model, self.conf.guide,
                                     self.optim, loss=self.elbo)
        self.svi_state = self.svi.init(self.subkey(), init_params=self.conf.inits)
        self.gather()
        self.losses = []

    def subkey(self):
        self.key, subkey = random.split(self.key)
        return subkey

    def trace_model(self, seed=0) -> TraceType:
        return handlers.trace(handlers.seed(self.conf.model, seed)).get_trace()

    def prior(self, site: str) -> dist.Distribution:
        conf = self.conf
        trace = self.trace_model()

        match site:
            case 'c':
                c2_aux = trace['c2_aux']['fn']
                return dist.TransformedDistribution(
                    c2_aux, [dist.transforms.PowerTransform(.5),
                             dist.transforms.AffineTransform(0., conf.c_scale)])
            case _:
                return trace[site]['fn']
    
    def trace_guide(self, conditioned=True, seed=0) -> TraceType:
        g = handlers.seed(self.conf.guide, seed)
        if conditioned:
            g = handlers.substitute(g, self.params)
        trace = handlers.trace(g).get_trace()
        return trace

    def estimate(self, site: str) -> ArrayLike | None:
        posterior = self.posterior(site)
        match type(posterior):
            case dist.LogNormal:
                return posterior.mode
            case dist.Normal:
                return posterior.mean

    def posterior(self, site: str) -> dist.Distribution:
        conf = self.conf
        trace = self.trace_guide()
        
        match site:
            case 'c':
                c2_aux = self.posterior('c2_aux')
                loc = .5 * c2_aux.loc + jnp.log(conf.c_scale)
                scale = .5 * c2_aux.scale
                return dist.LogNormal(loc, scale)
            case 'coef':
                tau = self.estimate('tau')
                lambda_reg = to_reg_lambda(tau**2, self.estimate('lambda')**2, self.estimate('c')**2)
                rhs_scale = tau * lambda_reg
                coef_dec = self.posterior('coef_dec')
                return dist.Normal(coef_dec.loc * rhs_scale, coef_dec.scale * rhs_scale)
            case _:
                return trace[site]['fn']
        
    def gather(self):
        self.params = self.svi.get_params(self.svi_state)
        self.traceg = self.trace_guide()
        self.tracem = self.trace_model()
        self.estimates = {site: self.estimate(site)
                          for t in [self.traceg, self.tracem]
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
