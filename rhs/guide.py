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
class GuideCorr:
    """
    Model correlation between coefficients as multivariate normal.
    """
    low_rank_factor: int | None = None

    @property
    def name(self):
        suffix = ''
        if self.low_rank_factor:
            suffix = f'_{self.low_rank_factor}'
        return f'corr{suffix}'
    
    def guide(self, D: int, reparam: Reparam, coef_decentered: bool, tau: ArrayLike, c: ArrayLike, inits: dict[str, Any]):
        coef_name = 'coef' + ('_dec' if coef_decentered else '')

        with numpyro.plate('d', D):
            if reparam:
                lambda_, lambda_loc, lambda_scale = reparam.guide('lambda', 1., inits)
            else:
                lambda_, lambda_loc, lambda_scale = lognormal_site('lambda', inits)

            lambda_reg = to_reg_lambda(tau**2, lambda_**2, c*c)

            coef_loc = numpyro.param(f'locs.{coef_name}', inits[f'locs.{coef_name}'])
            coef_scale = numpyro.param(
                f'scales.{coef_name}', inits[f'scales.{coef_name}'], constraint=pos_const)

        # Multivariate normal over D-dim coefficients: q(coef|lambda)
        if self.low_rank_factor is None:
            # Cholesky of the correlation matrix
            coef_chol_corr = numpyro.param(
                'chols.coef',
                inits['chols.coef'],
                constraint=dist.constraints.corr_cholesky)
            # Cholesky of the covariance matrix
            coef_chol_cov = jnp.diag(coef_scale) @ coef_chol_corr
            coef_joint = numpyro.sample(
                'coef_joint',
                dist.MultivariateNormal(coef_loc, scale_tril=coef_chol_cov),
                infer={'is_auxiliary': True})
        else:
            # Model q(coef|lambda) as a low-rank multivariate normal.
            # 𝛴 = FF^T + diag(d), where
            # - F is DxM is the low-rank factor, M < D
            # - d is D vector that represents the diagonal component
            # In our case, variances (ie diagonals of 𝛴), v, are already known.
            # We can use it to find d: d = v - sum_j(F_j^2)
            coef_cov_factor = numpyro.param(
                'coef_factor',
                inits.get('coef_factor', jnp.ones((D, self.low_rank_factor))*0.))
            coef_cov_diag = jnp.square(coef_scale) - jnp.square(coef_cov_factor).sum(-1)
            coef_cov_diag = jnp.clip(coef_cov_diag, min=1e-6)
            # coef_cov_diag = jnp.square(coef_scale_cond - coef_cov_factor.sum(-1))
            coef_joint = numpyro.sample(
                'coef_joint',
                dist.LowRankMultivariateNormal(coef_loc, coef_cov_factor, coef_cov_diag),
                infer={'is_auxiliary': True})

        with numpyro.plate('d', D):
            if coef_decentered:
                coef_dec = numpyro.sample('coef_dec', dist.Delta(coef_joint))
                coef = numpyro.deterministic('coef', coef_dec * tau * lambda_reg)
            else:
                coef_ = numpyro.sample('coef', dist.Delta(coef_joint))


@dataclass
class GuidePairMv:
    """
    Paired multivariate normal guide.
    
    Model each (lambda_d, coef_d) pair as a multivariate normal.
    This requires coef and lambda to be a modeled jointly, so it cannot be used with a reparam.
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
    using conditional rule: p(lambda_d, coef_d)=p(coef_d|lambda_d)p(lambda_d).
    In this way, lambda can be sampled independently, allowing reparam to be used.
    """
    name = 'paircond'

    @staticmethod
    def _posterior_coef(lambda_, lambda_loc, lambda_scale, coef_loc, coef_scale, corr) -> tuple[ArrayLike, ArrayLike]:
        """
        Return params of p(coef|lambda) ~ N(), using the conditional rule of
        multivariate normal distributions.
        """
        if lambda_ is None: # lambda_ = lambda_loc
            loc = coef_loc
        else:
            loc = coef_loc + corr * coef_scale / lambda_scale * (jnp.log(lambda_) - lambda_loc)
        scale = jnp.sqrt(1 - corr**2) * coef_scale
        return loc, scale
    
    def guide(self, D: int, reparam: Reparam, coef_decentered: bool, tau: ArrayLike, c: ArrayLike, inits: dict[str, Any]):
        with numpyro.plate('d', D):
            
            # Marginal of lambda, q(lambda)
            if reparam:
                lambda_, lambda_loc, lambda_scale = reparam.guide('lambda', 1., inits)
            else:
                lambda_, lambda_loc, lambda_scale = lognormal_site('lambda', inits)

            lambda_reg = to_reg_lambda(tau**2, lambda_**2, c*c)

            # D-dim correlation vector of each (coef, lambda) pair
            lambda_coef_corr = numpyro.param(
                'corrs.lambda_coef', 
                # TODO If i use lambda here, utils.get_sample_params raises error
                # lambda _: inits.get('corrs.lambda_coef', jnp.zeros(lambda_.shape)),
                inits.get('corrs.lambda_coef', jnp.zeros(D)),
                constraint=dist.constraints.interval(-1., 1.))

            # Marginal of coef, q(coef)
            coef_name = 'coef' + ('_dec' if coef_decentered else '')
            coef_loc = numpyro.param(f'locs.{coef_name}', inits[f'locs.{coef_name}'])
            coef_scale = numpyro.param(
                f'scales.{coef_name}',
                inits[f'scales.{coef_name}'],
                constraint=pos_const)

            # Given the marginals q(lambda) and q(coef), we can get the
            # conditional q(coef|lambda) using the conditional rule of
            # multivariate normals.
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

        # Multivariate normal over D-dim coefficients: q(coef|lambda)
        if self.low_rank_factor is None:
            # Cholesky of the correlation matrix
            coef_chol_corr = numpyro.param(
                'chols.coef',
                inits['chols.coef'],
                constraint=dist.constraints.corr_cholesky)
            # Cholesky of the covariance matrix
            coef_chol_cov = jnp.diag(coef_scale_cond) @ coef_chol_corr
            # coef_chol_cov = jnp.diag(coef_scale_cond) @ coef_chol_corr @ jnp.diag(coef_scale_cond).T
            coef_joint = numpyro.sample(
                'coef_joint',
                dist.MultivariateNormal(coef_loc_cond, scale_tril=coef_chol_cov),
                infer={'is_auxiliary': True})
        else:
            # Model q(coef|lambda) as a low-rank multivariate normal.
            # 𝛴 = FF^T + diag(d), where
            # - F is DxM is the low-rank factor, M < D
            # - d is D vector that represents the diagonal component
            # In our case, variances (ie diagonals of 𝛴), v, are already known.
            # We can use it to find d: d = v - sum_j(F_j^2)
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
GuideStructure = GuideUnstructured | GuideCorr | GuidePairCond | GuidePairMv | GuideFullMatrix | GuidePairCondCorr


            
