import re
from dataclasses import dataclass, field
from typing import Literal, Callable

import dill
import numpyro
import numpyro.distributions as dist
from numpyro import handlers
from numpyro.infer import MCMC, HMC, NUTS, Trace_ELBO, TraceMeanField_ELBO
from numpyro.infer.mcmc import MCMCKernel
from numpyro.infer.svi import SVIState
import jax.numpy as jnp
from jax import random
from jax.typing import ArrayLike
import optax

from .dist import *
from .common import TraceType
from .configuration import Configuration
from .common import to_reg_lambda
from .guide import GuideUnstructured, GuidePairCond, GuidePairMv, GuidePairCondCorr
from .utils import get_sample_params
from .elbo import MultiELBO


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
            # dill.dump(dataclasses.asdict(self), f)
            dill.dump(self, f)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            data = dill.load(f)
            # trainer = dacite.from_dict(data_class=cls, data=data)
            trainer = data
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
        self.kernel = NUTS(self.conf.model)
        self.mcmc = MCMC(self.kernel, num_warmup=self.num_warmup, num_samples=self.num_samples)

    def train(self):
        init_params = {s: jnp.full_like(n['value'], .01) for s, n in self.trace_model().items() if 'value' in n and n['type'] == 'sample'}
        self.mcmc.run(self.subkey())#, init_params=init_params)
        self.gather()

    def gather(self):
        self.tracem = self.trace_model()
        self.samples = self.mcmc.get_samples()
        self.diverging = self.mcmc._states['diverging'][0]
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
    
    def trace_guide(self, conditioned=True, seed=0, with_sites=True) -> TraceType:
        g = handlers.seed(self.conf.guide, seed)
        if conditioned:
            data = {**self.params}
            # if with_sites:
                # t = self.trace_guide(with_sites=False)
                # estimates = {site: self.estimate(site, t) for site in ['lambda']}
                # data.update(estimates)
            g = handlers.substitute(g, data)
        trace = handlers.trace(g).get_trace()
        return trace

    def estimate(self, site: str | dist.Distribution, trace: TraceType | None = None) -> ArrayLike | None:
        if isinstance(site, str):
            posterior = self.posterior(site, trace=trace)
        else:
            posterior = site
        match type(posterior):
            case dist.LogNormal:
                return posterior.mode
            case dist.Normal:
                return posterior.mean

    def posterior(self, site: str, trace: TraceType | None = None) -> dist.Distribution | None:
        conf = self.conf
        if trace is None:
            trace = self.trace_guide()

        match site:
            case 'c':
                c2_aux = self.posterior('c2_aux')
                loc = .5 * c2_aux.loc + jnp.log(conf.c_scale)
                scale = .5 * c2_aux.scale
                return dist.LogNormal(loc, scale)
            case 'coef':
                lambda_post = self.posterior('lambda')
                lambda_ = self.estimate(lambda_post)

                match self.conf.structure:
                    case GuideUnstructured():
                        coef = trace[self.conf.coef_name]["fn"]
                    case GuidePairCond() | GuidePairMv() | GuidePairCondCorr():
                        coef_marginal = dist.Normal(
                            self.params[f"locs.{self.conf.coef_name}"],
                            self.params[f"scales.{self.conf.coef_name}"],
                        )
                        corr = self.params["corrs.lambda_coef"]
                        loc, scale = GuidePairCond._posterior_coef(
                            # None,
                            lambda_,
                            # lambda_post.mean,
                            lambda_post.loc,
                            lambda_post.scale,
                            coef_marginal.loc,
                            coef_marginal.scale,
                            corr,
                        )
                        coef = dist.Normal(loc, scale)

                if not self.conf.coef_dec:
                    return coef
                
                tau = self.estimate('tau')
                lambda_reg = to_reg_lambda(tau**2, lambda_**2, self.estimate('c')**2)
                rhs_scale = tau * lambda_reg
                return dist.Normal(coef.loc * rhs_scale, coef.scale * rhs_scale)
            case 'lambda' if isinstance(self.conf.structure, GuidePairMv):
                return dist.LogNormal(
                    self.params['locs.lambda'],
                    self.params['scales.lambda'])
            case _ if self.conf.reparam_tau and site.startswith('tau'):
                scale = self.conf.tau_scale
                return self.conf.reparam_tau.posterior(trace, site, 'tau', scale)
            case _ if self.conf.reparam_lambda and site.startswith('lambda'):
                scale = 1.
                return self.conf.reparam_lambda.posterior(trace, site, 'lambda', scale)
            case _:
                return trace[site].get('fn')
        
    def gather(self):
        self.params = self.svi.get_params(self.svi_state)
        self.traceg = self.trace_guide()
        self.tracem = self.trace_model()
        self.estimates = {site: self.estimate(site)
                          for t in [self.traceg, self.tracem]
                          for site, node in t.items()
                          if node['type'] in ['sample', 'deterministic'] and not node.get('is_observed', False) and site not in ['lambda_reg', 'c']}
            
    def train(self, steps):
        try:
            res = self.svi.run(self.subkey(), steps, init_state=self.svi_state,
                               progress_bar=False)
            self.svi_state = res.state
            self.losses.extend(res.losses.tolist())
        finally:
            self.gather()

    def sample_posterior(self, num_samples=300, method: Literal['trace', 'posterior'] = 'trace', seed=0):
        match method:
            case 'trace':
                pred = numpyro.infer.Predictive(
                    self.conf.guide, params=self.params, num_samples=num_samples,
                    exclude_deterministic=False)
                return pred(random.key(seed))
            case 'posterior':
                key = random.key(seed)
                samples = {site: self.posterior(site).sample((key:=random.split(key)[1]),
                                                             (num_samples,))
                           for site, node in self.trace_guide().items()
                           if node['type'] in ['sample', 'deterministic']}
                return samples

    @staticmethod
    def build_specialized_elbo(
        conf: Configuration,
        num_particles: int = 1,
        reparam_only: bool = True
    ) -> MultiELBO | Trace_ELBO:
        """ELBO where half-cauchy priors tau and lambda use the `TraceMeanField`
        ELBO.

        This is useful because when the half-cauchy prior is decomposed into
        gamma, and inverse-gamma distributions, the KL-divergence to the
        lognormal posterior is closed-form, which `TraceMeanField` can take
        advantage of.
        """
        # TODO Handle reparam tau and lambda cases seperately
        if reparam_only and conf.reparam_tau is None and conf.reparam_lambda is None:
            return Trace_ELBO(num_particles)
        
        sample_params = get_sample_params(conf.guide)
        half_cauchy_sites = []
        for site, params in sample_params.items():
            if site.split('_')[0] in ['lambda', 'tau']:
                half_cauchy_sites.extend([site, *params])
        half_cauchy_sites = tuple(half_cauchy_sites)
        
        elbos = {
            half_cauchy_sites: TraceMeanField_ELBO(num_particles),
            None: Trace_ELBO(num_particles)
        }
        multi_elbo = MultiELBO.build(elbos, conf.model, conf.guide)
        return multi_elbo
