from dataclasses import dataclass, field
from typing import Any
from collections.abc import Callable

import numpyro
import numpyro.distributions as dist
from numpyro import handlers
import jax.numpy as jnp
from jax import random
from jax.typing import ArrayLike

from .common import TraceType


# * Utils
exp_transform = dist.transforms.ExpTransform()
pos_const = dist.constraints.softplus_positive

def normal_site(name, inits: dict[str, Any], log: bool = False) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
    loc = numpyro.param(f'locs.{name}', inits[f'locs.{name}'])
    scale = numpyro.param(
        f'scales.{name}',
        inits[f'scales.{name}'],
        constraint=pos_const)
    D = dist.LogNormal if log else dist.Normal
    return numpyro.sample(name, D(loc, scale)), loc, scale

def lognormal_site(name, inits: dict[str, Any]) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
    return normal_site(name, inits, log=True)

def to_reg_lambda(tau2, lambda2, c2):
    return jnp.sqrt(lambda2 * c2 / (c2 + tau2 * lambda2))

# * Reparam

# ** InverseGamma - InverseGamma
@dataclass
class ReparamII:
    dec1: bool
    dec2: bool

    @property
    def name(self) -> str:
        if self.dec1 and self.dec2:
            return 'ii:dec'
        if self.dec1:
            return 'ii:dec1'
        if self.dec2:
            return 'ii:dec2'
        return 'ii:cen'

    def model(self, name: str, scale) -> ArrayLike:
        if self.dec1:
            aux1_dec = numpyro.sample(f'{name}_aux1_dec', dist.InverseGamma(0.5, 1.))
            aux1 = numpyro.deterministic(f'{name}_aux1', aux1_dec / (scale*scale))
        else:
            aux1 = numpyro.sample(f'{name}_aux1', dist.InverseGamma(0.5, 1. / (scale*scale)))

        if self.dec2:
            aux2_dec = numpyro.sample(f'{name}_aux2_dec', dist.InverseGamma(0.5, 1.))
            aux2 = numpyro.deterministic(f'{name}_aux2', aux2_dec/aux1)
        else:
            aux2 = numpyro.sample(f'{name}_aux2', dist.InverseGamma(0.5, 1./aux1))

        if self.dec1 and self.dec2:
            return numpyro.deterministic(name, scale * jnp.sqrt(aux2_dec / aux1_dec))
        else:
            return numpyro.deterministic(name, jnp.sqrt(aux2))

    def guide(self, name: str, scale, inits: dict[str, Any]) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
        aux1n = f'{name}_aux1'
        aux2n = f'{name}_aux2'

        if self.dec1:
            aux1_dec, aux1_dec_loc, aux1_dec_scale = lognormal_site(f'{aux1n}_dec', inits)
            aux1 = numpyro.deterministic(aux1n, aux1_dec / (scale*scale))
            aux1_loc, aux1_scale = self._posterior_aux1(aux1_dec_loc, aux1_dec_scale, scale)
        else:
            aux1, aux1_loc, aux1_scale = lognormal_site(aux1n, inits)

        if self.dec2:
            aux2_dec, aux2_dec_loc, aux2_dec_scale = lognormal_site(f'{aux2n}_dec', inits)
            aux2 = numpyro.deterministic(aux2n, aux2_dec / aux1)
            aux2_loc, aux2_scale = self._posterior_aux2(
                aux1_loc, aux1_scale, aux2_dec_loc, aux2_dec_scale)
        else:
            aux2, aux2_loc, aux2_scale = lognormal_site(aux2n, inits)

        loc, scale = self._posterior_site(aux2_loc, aux2_scale)
        site = numpyro.deterministic(name, jnp.sqrt(aux2))
        return site, loc, scale

    def _posterior_aux1(self, dec_loc, dec_scale, scale) -> tuple[ArrayLike, ArrayLike]:
        s = -2 * jnp.log(scale)
        return dec_loc + s, dec_scale

    def _posterior_aux2(self, aux1_loc, aux1_scale, dec_loc, dec_scale) -> tuple[ArrayLike, ArrayLike]:
        loc = dec_loc - aux1_loc
        scale = jnp.hypot(dec_scale, aux1_scale)
        return loc, scale
        
    def _posterior_site(self, aux2_loc, aux2_scale) -> tuple[ArrayLike, ArrayLike]:
        loc = 0.5 * aux2_loc
        scale = jnp.sqrt(0.25 * aux2_scale**2)
        return loc, scale

    def posterior(self, trace: TraceType, site: str, name: str, scale) -> dist.Distribution:
        """
        Param `name` is the name of the half-cauchy prior site.
        Param `site` is the name of the site of the requested posterior.
        """
        if site.endswith('aux1') and self.dec1:
            dec = trace[f'{site}_dec']['fn']
            loc, scale = self._posterior_aux1(dec.loc, dec.scale, scale)
            return dist.LogNormal(loc, scale)
        elif site.endswith('aux2') and self.dec2:
            ## aux2_dec / aux1
            # 1. Uses mean instead of dividing the two logs
            """
            aux1 = self.posterior(trace, f'{name}_aux1', name, scale)
            dec = t[f'{name}_aux2_dec']['fn']
            s = 1. / aux1.mean
            return LogNormal(dec.loc + jnp.log(s), dec.scale)
            """
            # 2. Divide two lognormals using aux1
            # """
            aux1 = self.posterior(trace, f'{name}_aux1', name, scale)
            dec = trace[f'{name}_aux2_dec']['fn']
            loc, scale = self._posterior_aux2(aux1.loc, aux1.scale, dec.loc, dec.scale)
            return dist.LogNormal(loc, scale)
            # """
        elif site == name:
            # 1. Indirectly
            d = self.posterior(trace, f'{name}_aux2', name, scale)
            loc, scale = self._posterior_site(d.loc, d.scale)
            return dist.LogNormal(loc, scale)

            # 2. Directly
            """
            aux1_dec = trace[f'{name}_aux1_dec']['fn']
            aux2_dec = trace[f'{name}_aux2_dec']['fn']
            loc = jnp.log(scale) + (aux2_dec.loc - aux1_dec.loc) / 2
            scale = 0.5 * jnp.hypot(aux1_dec.scale, aux2_dec.scale)
            return dist.LogNormal(loc, scale)
            """
        else:
            return trace[site]['fn']

    
# ** InverseGamma - Gamma

def lognormal_from_sqrtmul(aux1_loc, aux1_scale, aux2_loc, aux2_scale, reparam_scale=None):
    """
    If aux1 ~ LN() and aux2 ~ LN(), returns loc and scale of sqrt(aux1*aux2) ~ LN().x
    """
    if reparam_scale is not None:
        aux2_loc += jnp.log(reparam_scale*reparam_scale)
    loc = .5 * (aux1_loc + aux2_loc)
    # ``jnp.hypot`` is a more numerically stable way of computing
    # ``jnp.sqrt(x1 ** 2 + x2 **2)``.
    # scale = .5 * jnp.sqrt(aux1_scale**2 + aux2_scale**2)
    scale = .5 * jnp.hypot(aux1_scale, aux2_scale)
    return loc, scale

@dataclass
class ReparamIG:
    dec: bool

    @property
    def name(self) -> str:
        return 'ig:dec' if self.dec else 'ig:cen'

    def model(self, name: str, scale) -> ArrayLike:
        if self.dec:
            aux1 = numpyro.sample(f'{name}_aux1', dist.InverseGamma(0.5, 1.))
            aux2_dec = numpyro.sample(f'{name}_aux2_dec', dist.Gamma(0.5, 1.)) 
            aux2 = numpyro.deterministic(f'{name}_aux2', aux2_dec * scale * scale)
            return numpyro.deterministic(name, jnp.sqrt(aux1 * aux2_dec) * scale)
        else:
            aux1 = numpyro.sample(f'{name}_aux1', dist.InverseGamma(0.5, 1.))
            aux2 = numpyro.sample(f'{name}_aux2', dist.Gamma(0.5, 1./(scale*scale)))
            return numpyro.deterministic(name, jnp.sqrt(aux1 * aux2))

    def guide(self, name: str, scale, inits: dict[str, Any]) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
        aux1, aux1_loc, aux1_scale = lognormal_site(f'{name}_aux1', inits)

        if self.dec:
            aux2_dec, aux2_dec_loc, aux2_dec_scale = lognormal_site(f'{name}_aux2_dec', inits)
            aux2 = numpyro.deterministic(f'{name}_aux2', aux2_dec * scale * scale)
            aux2_loc, aux2_scale = self._posterior_aux2(
                aux2_dec_loc, aux2_dec_scale, scale)
            site = numpyro.deterministic(name, jnp.sqrt(aux1 * aux2_dec) * scale)

        else:
            aux2, aux2_loc, aux2_scale = lognormal_site(f'{name}_aux2', inits)
            site = numpyro.deterministic(name, jnp.sqrt(aux1 * aux2))

        loc, scale = self._posterior_site(aux1_loc, aux1_scale, aux2_loc, aux2_scale)
        return site, loc, scale

    def _posterior_aux2(self, dec_loc, dec_scale, scale) -> tuple[ArrayLike, ArrayLike]:
        return dec_loc + jnp.log(scale*scale), dec_scale

    def _posterior_site(self, aux1_loc, aux1_scale, aux2_loc, aux2_scale) -> tuple[ArrayLike, ArrayLike]:
        return lognormal_from_sqrtmul(aux1_loc, aux1_scale, aux2_loc, aux2_scale)
        
    def posterior(self, trace: TraceType, site: str, name: str, scale) -> dist.Distribution:
        """
        Param `name` is the name of the half-cauchy prior site.
        Param `site` is the name of the site of the requested posterior.
        """
        if site.endswith('aux2') and self.dec:
            dec = trace[f'{site}_dec']['fn']
            loc, scale = self._posterior_aux2(dec.loc, dec.scale, scale)
            return dist.LogNormal(loc, scale)
        elif site == name:
            aux1 = self.posterior(trace, f'{name}_aux1', name, scale)
            aux2 = self.posterior(trace, f'{name}_aux2', name, scale)
            loc, scale = self._posterior_site(aux1.loc, aux1.scale, aux2.loc, aux2.scale)
            return dist.LogNormal(loc, scale)
        else:
            return trace[site]['fn']


# ** Type
Reparam = None | ReparamII | ReparamIG

# * Structured guide

# ** Unstructured
@dataclass
class GuideUnstructured:
    """
    Mean field.
    """
    name = 'unstructured'
    
    def guide(self, D: int, reparam: Reparam, coef_decentered: bool, tau: ArrayLike, c: ArrayLike, inits: dict[str, Any]):
        with numpyro.plate('d', D):
            if reparam:
                lambda_, *_ = reparam.guide('lambda', 1., inits)
            else:
                lambda_, *_ = lognormal_site('lambda', inits)

            lambda_reg = to_reg_lambda(tau**2, lambda_**2, c*c)

            if coef_decentered:
                coef_dec, *_ = normal_site('coef_dec', inits)
                coef = numpyro.deterministic('coef', coef_dec * tau * lambda_reg)
            else:
                coef, *_ = normal_site('coef', inits)
        
# ** Paired Mv Normal
@dataclass
class GuidePairMv:
    """
    Model each (lambda_d, coef_d) pair as a multivariate normal.
    This requires lambda to be a sample site, so it cannot be used with a reparam.
    """
    name = 'pairmv'
        
    def guide(self, D: int, reparam: Reparam, coef_decentered: bool, tau: ArrayLike, c: ArrayLike, inits: dict[str, Any]):
        assert reparam is None
        # TODO implement
        with numpyro.plate('d', D):
            lambda_loc = numpyro.param('locs.lambda', inits['locs.lambda'])
            lambda_scale = numpyro.param(
                'scales.lambda',
                inits['scales.lambda'],
                constraint=pos_const)

            coef_name = 'coef' + ('_dec' if coef_decentered else '')
            coef_loc = numpyro.param(f'locs.{coef_name}', inits[f'locs.{coef_name}'])
            coef_scale = numpyro.param(
                f'scales.{coef_name}',
                inits[f'scales.{coef_name}'],
                constraint=pos_const)

            lambda_coef_corr = numpyro.param(
                'corrs.lambda_coef', 
                inits.get('corrs.lambda_coef', jnp.full(D, 0.0)),
                constraint=dist.constraints.interval(-1., 1.))
            
            covs = lambda_coef_corr * lambda_scale * coef_scale
            cov_matrix = jnp.stack(
                [
                    jnp.stack([lambda_scale**2, covs], axis=-1),
                    jnp.stack([covs, coef_scale**2], axis=-1),
                ],
                axis=-2,
            )
            loglambda_coef = numpyro.sample(
                'loglambda_coef',
                dist.MultivariateNormal(jnp.stack([lambda_loc, coef_loc], axis=-1), cov_matrix),
                infer={'is_auxiliary': True})

            lambda_ = numpyro.sample('lambda', dist.Delta(jnp.exp(loglambda_coef[:, 0])))
            lambda_reg = to_reg_lambda(tau**2, lambda_**2, c*c)

            coef_dist = dist.Delta(loglambda_coef[:, 1])
            if coef_decentered:
                coef_dec = numpyro.sample('coef_dec', coef_dist)
                coef = numpyro.deterministic('coef', coef_dec * tau * lambda_reg)
            else:
                coef_ = numpyro.sample('coef', coef_dist)

# ** Paired Conditional
@dataclass
class GuidePairCond:
    """
    Model each (lambda_d, coef_d) pair as a multivariate normal, but using seperate sites
    using p(lambda_d, coef_d)=p(coef_d|lambda_d)p(lambda_d).
    This does not require lambda to be a sample site, as long as a lambda values can be
    passed into the conditional, so it cannot be used with a reparam.
    """
    name = 'paircond'

    @staticmethod
    def _posterior_coef(lambda_, lambda_loc, lambda_scale, coef_loc, coef_scale, corr) -> tuple[ArrayLike, ArrayLike]:
        if lambda_ is None: # lambda_ = lambda_loc
            loc = coef_loc
        else:
            loc = coef_loc + corr * coef_scale / lambda_scale * (jnp.log(lambda_) - lambda_loc)
        scale = jnp.sqrt(1 - corr**2) * coef_scale
        return loc, scale
    
    def guide(self, D: int, reparam: Reparam, coef_decentered: bool, tau: ArrayLike, c: ArrayLike, inits: dict[str, Any]):
        with numpyro.plate('d', D):
            if reparam:
                lambda_, lambda_loc, lambda_scale = reparam.guide('lambda', 1., inits)
            else:
                lambda_, lambda_loc, lambda_scale = lognormal_site('lambda', inits)

            lambda_reg = to_reg_lambda(tau**2, lambda_**2, c*c)

            # If i use lambda here, utils.get_sample_params raises error
            lambda_coef_corr = numpyro.param(
                'corrs.lambda_coef', 
                # lambda _: inits.get('corrs.lambda_coef', jnp.zeros(lambda_.shape)),
                inits.get('corrs.lambda_coef', jnp.zeros(D)),
                constraint=dist.constraints.interval(-1., 1.))

            coef_name = 'coef' + ('_dec' if coef_decentered else '')
            coef_loc = numpyro.param(f'locs.{coef_name}', inits[f'locs.{coef_name}'])
            coef_scale = numpyro.param(
                f'scales.{coef_name}',
                inits[f'scales.{coef_name}'],
                constraint=pos_const)
    
            coef_loc_cond, coef_scale_cond = self._posterior_coef(
                lambda_, lambda_loc, lambda_scale, coef_loc, coef_scale, lambda_coef_corr)
            coef_dist = dist.Normal(coef_loc_cond, coef_scale_cond)
            
            if coef_decentered:
                coef_dec = numpyro.sample('coef_dec', coef_dist)
                coef = numpyro.deterministic('coef', coef_dec * tau * lambda_reg)
            else:
                coef_ = numpyro.sample('coef', coef_dist)

# ** Paired Conditional Correlated
@dataclass
class GuidePairCondCorr(GuidePairCond):
    """
    Same as GuidePairCond, but model the dependencies of the resulting coefs as a
    multivariate normal.
    """
    low_rank_factor: int | None = None

    @property
    def name(self):
        suffix = ''
        if self.low_rank_factor:
            suffix = f'_{self.low_rank_factor}'
        return f'paircondcorr{suffix}'
    
    def guide(self, D: int, reparam: Reparam, coef_decentered: bool, tau: ArrayLike, c: ArrayLike, inits: dict[str, Any]):
        coef_name = 'coef' + ('_dec' if coef_decentered else '')

        with numpyro.plate('d', D):
            if reparam:
                lambda_, lambda_loc, lambda_scale = reparam.guide('lambda', 1., inits)
            else:
                lambda_, lambda_loc, lambda_scale = lognormal_site('lambda', inits)

            lambda_reg = to_reg_lambda(tau**2, lambda_**2, c*c)

            lambda_coef_corr = numpyro.param(
                'corrs.lambda_coef', 
                inits.get('corrs.lambda_coef', jnp.zeros(lambda_.shape)),
                constraint=dist.constraints.interval(-1., 1.))

            coef_loc = numpyro.param(f'locs.{coef_name}', inits[f'locs.{coef_name}'])
            coef_scale = numpyro.param(
                f'scales.{coef_name}',
                inits[f'scales.{coef_name}'],
                constraint=pos_const)
    
        coef_loc_cond, coef_scale_cond = self._posterior_coef(lambda_, lambda_loc, lambda_scale, coef_loc, coef_scale, lambda_coef_corr)

        if self.low_rank_factor is None:
            coef_chol_corr = numpyro.param(
                'chols.coef',
                inits['chols.coef'],
                constraint=dist.constraints.corr_cholesky)
            coef_chol_cov = jnp.diag(coef_scale_cond) @ coef_chol_corr @ jnp.diag(coef_scale_cond).T
            coef_joint = numpyro.sample(
                'coef_joint',
                dist.MultivariateNormal(coef_loc_cond, scale_tril=coef_chol_cov),
                infer={'is_auxiliary': True})
        else:
            coef_cov_factor = numpyro.param(
                'coef_factor',
                inits.get('coef_factor', jnp.ones((D, self.low_rank_factor))*0.))
            coef_cov_diag = jnp.square(coef_scale_cond) - jnp.square(coef_cov_factor).sum(-1)
            coef_cov_diag = jnp.clip(coef_cov_diag, min=1e-6)
            # coef_cov_diag = jnp.square(coef_scale_cond - coef_cov_factor.sum(-1))
            coef_joint = numpyro.sample(
                'coef_joint',
                dist.LowRankMultivariateNormal(coef_loc_cond, coef_cov_factor, coef_cov_diag),
                infer={'is_auxiliary': True})

        with numpyro.plate('d', D):
            if coef_decentered:
                coef_dec = numpyro.sample('coef_dec', dist.Delta(coef_joint))
                coef = numpyro.deterministic('coef', coef_dec * tau * lambda_reg)
            else:
                coef_ = numpyro.sample('coef', dist.Delta(coef_joint))


# ** Full Matrix
@dataclass
class GuideFullMatrix:
    """
    Model p(lambda, coef) over all dimensions as a matrix normal.
    """
    name = 'fullmatrix'

    # TODO implement
    def guide(self, D: int, reparam: Reparam, coef_decentered: bool, tau: ArrayLike, c: ArrayLike, inits: dict[str, Any]):
        lambda_loc = numpyro.param('locs.lambda', inits['locs.lambda'])

        coef_name = 'coef' + ('_dec' if coef_decentered else '')
        coef_loc = numpyro.param(f'locs.{coef_name}', inits[f'locs.{coef_name}'])

        

        with numpyro.plate('d', D):
            if reparam:
                lambda_, lambda_loc, lambda_scale = reparam.guide('lambda', 1., inits)
            else:
                lambda_, lambda_loc, lambda_scale = lognormal_site('lambda', inits)

            lambda_reg = to_reg_lambda(tau**2, lambda_**2, c*c)

            lambda_coef_corr = numpyro.param(
                'corrs.lambda_coef',
                inits.get('corrs.lambda_coef', jnp.zeros(lambda_.shape)),
                constraint=dist.constraints.interval(-1., 1.))

            coef_name = 'coef' + ('_dec' if coef_decentered else '')
            coef_loc = numpyro.param(f'locs.{coef_name}', inits[f'locs.{coef_name}'])
            coef_scale = numpyro.param(
                f'scales.{coef_name}',
                inits[f'scales.{coef_name}'],
                constraint=pos_const)
    
            coef_loc_cond, coef_scale_cond = self._posterior_coef(lambda_, lambda_loc, lambda_scale, coef_loc, coef_scale, lambda_coef_corr)
            coef_dist = dist.Normal(coef_loc_cond, coef_scale_cond)
            
            if coef_decentered:
                coef_dec = numpyro.sample('coef_dec', coef_dist)
                coef = numpyro.deterministic('coef', coef_dec * tau * lambda_reg)
            else:
                coef_ = numpyro.sample('coef', coef_dist)


# ** Type
GuideStructure = GuideUnstructured | GuidePairCond | GuidePairMv | GuideFullMatrix | GuidePairCondCorr

# * Config
            
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
