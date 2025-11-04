from dataclasses import dataclass
from typing import Any

import numpyro
import numpyro.distributions as dist
import jax.numpy as jnp
from jax.typing import ArrayLike

from .common import normal_site, lognormal_site, pos_const, to_reg_lambda
from .reparam import Reparam


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
        

@dataclass
class GuidePairMv:
    """
    Paired multivariate normal guide.
    
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

@dataclass
class GuidePairCond:
    """
    Paired conditional guide.
    
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


@dataclass
class GuidePairCondCorr(GuidePairCond):
    """
    Paired conditional correlated guide.
    
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


@dataclass
class GuideFullMatrix:
    """
    Full matrix guide.
    
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


# Type
GuideStructure = GuideUnstructured | GuidePairCond | GuidePairMv | GuideFullMatrix | GuidePairCondCorr


            
