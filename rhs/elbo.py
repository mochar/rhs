from typing import Callable, Self
from pprint import pprint

import jax.numpy as jnp
from jax import random
from numpyro.primitives import Message, Messenger
from numpyro.infer.elbo import ELBO, compute_log_probs
from numpyro import handlers


class MultiELBO(ELBO):
    def __init__(self, elbos: dict[tuple, ELBO]):
        self.elbos = elbos

    @staticmethod
    def build(
        elbos: dict[tuple | None, ELBO],
        model: Callable,
        guide: Callable
    ) -> Self:
        none_idx = [sites for sites in elbos if sites is None]
        assert len(none_idx) <= 1
        if len(none_idx) == 0:
            return MultiELBO(elbos)
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
        # To reuse the trace samples in the next elbos, need to recreate
        # the trace that is generated in the elbo.
        _, guide_seed = random.split(rng_key)
        seeded_guide = handlers.seed(guide, guide_seed)
        guide_log_probs, guide_trace = compute_log_probs(
            seeded_guide, args, kwargs, param_map
        )
        
        loss_data = {'loss': jnp.array(0.0), 'mutable_state': None}
        guide_samples = {} # Accumulated site -> sample
        for i, (sites, elbo) in enumerate(self.elbos.items()):
            elbo_model = model
            elbo_guide = guide

            # Substitute the guide samples
            if i > 0:
                elbo_guide = handlers.substitute(elbo_guide, guide_samples)
            for site in sites:
                guide_samples[site] = guide_trace[site]['value']
            
            # Calculate loss
            elbo_model = site_mask(elbo_model, sites, reverse=True)
            elbo_guide = site_mask(elbo_guide, sites, reverse=True)
            loss = elbo.loss_with_mutable_state(
                rng_key, param_map, elbo_model, elbo_guide,
                *args, **kwargs)['loss']
            loss_data['loss'] += loss

            pprint('Guide:\n', handlers.trace(handlers.seed(elbo_guide)).get_trace())

            
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
        print('Masking: ', msg['name'])

        if msg["type"] != "sample":
            if msg["type"] == "inspect":
                msg["mask"] = (
                    True if msg["mask"] is None else (True & msg["mask"])
                )
            return

        msg["fn"] = msg["fn"].mask(True)
