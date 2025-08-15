from typing import Callable, Self

import jax.numpy as jnp
from numpyro.primitives import Message, Messenger
from numpyro.infer.elbo import ELBO
from numpyro import handlers

from .utils import get_sample_params


class MultiELBO(ELBO):
    def __init__(self, elbos: dict[tuple, ELBO]):
        self.elbos = elbos

    @staticmethod
    def build(
        elbos: dict[tuple | None, ELBO],
        model: Callable,
        guide: Callable,
        append_params: bool = True
    ) -> Self:
        """Convenience method to build the `elbos` parameter for the class.
        
        The `elbos` dict of the method maps site names to ELBO instance. When
        `append_params` is True, the site names only have to contain the sample
        site names - the corresponding params will be appended
        automatically. Furthermore, one of the keys may be set to None, in which
        case it will be filled in with the remaining sites.
        """
        none_idx = [sites for sites in elbos if sites is None]
        assert len(none_idx) <= 1

        # In case the site names only contain sample sites without params,
        # append the params to the list.
        if append_params:
            sample_params = get_sample_params(guide)
            for sites in list(elbos.keys()): # Make sure to copy
                if sites is None:
                    continue
                all_sites = []
                for site in sites:
                    all_sites.append(site)
                    all_sites.extend(sample_params.get(site, []))
                elbo = elbos.pop(sites)
                elbos[tuple(set(all_sites))] = elbo
        
        # Just return if no sites are unspecified.
        if len(none_idx) == 0:
            return MultiELBO(elbos)

        # Infer remaining sites
        given_sites = tuple(site for sites in elbos for site in (sites or []))
        sites = tuple(set([
            site
            for m in (model, guide)
            for site in handlers.trace(handlers.seed(m, 0)).get_trace()
            if site not in given_sites
        ]))
        elbo = elbos.pop(None)
        elbos[sites] = elbo
        return MultiELBO(elbos)

    def loss_with_mutable_state(
        self,
        rng_key,
        param_map,
        model,
        guide,
        *args,
        **kwargs,
    ):
        """Wraps the original methods but masks the sites that are not included
        in each elbo's specification.
        """
        loss_data = {'loss': jnp.array(0.0), 'mutable_state': None}
        for sites, elbo in self.elbos.items():
            elbo_model = site_mask(model, sites, reverse=True)
            elbo_guide = site_mask(guide, sites, reverse=True)
            loss = elbo.loss_with_mutable_state(
                rng_key, param_map, elbo_model, elbo_guide,
                *args, **kwargs)['loss']
            loss_data['loss'] += loss

        return loss_data


class site_mask(Messenger):
    """
    This messenger masks out samples sites by name.
    """
    def __init__(
        self,
        fn: Callable,
        sites: list[str] | tuple[str],
        reverse: bool = False # Mask all other sites instead
    ) -> None:
        self.sites = sites
        self.reverse = reverse
        super().__init__(fn)

    def process_message(self, msg: Message) -> None:
        is_in = msg['name'] in self.sites
        if is_in if self.reverse else not is_in:
            return

        if msg["type"] != "sample":
            if msg["type"] == "inspect":
                msg["mask"] = (
                    False if msg["mask"] is None else (False & msg["mask"])
                )
            return

        msg["fn"] = msg["fn"].mask(False)
