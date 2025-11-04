from dataclasses import dataclass
from typing import Any

import numpyro
import numpyro.distributions as dist
import jax.numpy as jnp
from jax.typing import ArrayLike

from .common import TraceType, lognormal_site


@dataclass
class ReparamII:
    """
    InverseGamma-InverseGamma reparam.
    """
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
    """
    InverseGamma-Gamma reparam.
    """
    dec: bool

    @property
    def name(self) -> str:
        return 'ig:dec' if self.dec else 'ig:cen'

    def model(self, name: str, scale) -> ArrayLike:
        if self.dec:
            aux1 = numpyro.sample(f'{name}_aux1', dist.InverseGamma(0.5, 1.))
            aux2_dec = numpyro.sample(f"{name}_aux2_dec", dist.Gamma(0.5, 1.0)) 
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

# Type
Reparam = None | ReparamII | ReparamIG

