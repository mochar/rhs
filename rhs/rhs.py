import re
import dataclasses
from dataclasses import dataclass, field

import dill
import numpyro
import numpyro.distributions as dist
from numpyro import handlers
from numpyro.infer import MCMC, HMC, NUTS
from numpyro.infer.mcmc import MCMCKernel
from numpyro.infer.svi import SVIState
import jax.numpy as jnp
from jax import random
from jax.typing import ArrayLike
import optax
import dacite

from .dist import *
from .common import TraceType
from .model import Configuration, to_reg_lambda


@dataclass
class TrainerMixin:
    conf: Configuration
    key: ArrayLike = field(default_factory=lambda: random.key(0))

    def subkey(self):
        self.key, subkey = random.split(self.key)
        return subkey

    def trace_model(self, seed=0) -> TraceType:
        return handlers.trace(handlers.seed(self.conf.model, seed)).get_trace()

    def save(self, path):
        with open(path, 'wb') as f:
            dill.dump(dataclasses.asdict(self), f)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            data = dill.load(f)
            trainer = dacite.from_dict(data_class=cls, data=data)
            trainer.gather()
            return trainer


@dataclass
class TrainerMCMC(TrainerMixin):
    num_warmup: int = 500
    num_samples: int = 500
    kernel: MCMCKernel = field(init=False)
    mcmc: MCMC = field(init=False)

    def __post_init__(self):
        # self.kernel = HMC(self.conf.model)
        self.kernel = NUTS(self.conf.model, init_strategy=numpyro.infer.init_to_sample)
        self.mcmc = MCMC(self.kernel, num_warmup=self.num_warmup, num_samples=self.num_samples)

    def train(self):
        self.mcmc.run(self.subkey())
        self.gather()

    def gather(self):
        self.tracem = self.trace_model()
        self.samples = self.mcmc.get_samples()
        self.estimates = {site: samples.mean(0) for site, samples in self.samples.items()}

@dataclass
class TrainerSVI(TrainerMixin):
    optim: optax.GradientTransformation = optax.adam(0.01)
    elbo: numpyro.infer.ELBO = numpyro.infer.Trace_ELBO(10)
    svi_state: SVIState | None = None
    losses: list[float] = field(default_factory=list)
    
    def __post_init__(self):
        self.svi = numpyro.infer.SVI(self.conf.model, self.conf.guide,
                                     self.optim, loss=self.elbo)
        if self.svi_state is None:
            self.svi_state = self.svi.init(self.subkey(), init_params=self.conf.inits)
        else:
            self.svi.init(random.key(0)) # build svi.constrain_fn
        self.gather()

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
            case 'coef' if self.conf.coef_dec:
                tau = self.estimate('tau')
                lambda_reg = to_reg_lambda(tau**2, self.estimate('lambda')**2, self.estimate('c')**2)
                rhs_scale = tau * lambda_reg
                coef_dec = self.posterior('coef_dec')
                return dist.Normal(coef_dec.loc * rhs_scale, coef_dec.scale * rhs_scale)
            case _ if self.conf.reparam and (match := re.match(r'^(tau|lambda)', site)):
                name = match[0]
                scale = self.conf.tau_scale if name == 'tau' else 1.
                return self.conf.reparam.posterior(trace, site, name, scale)
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

    def sample_posterior(self, num_samples=300, seed=0):
        pred = numpyro.infer.Predictive(
            self.conf.guide, params=self.params, num_samples=num_samples)
        return pred(random.key(seed))
