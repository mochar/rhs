import numpyro
import numpyro.distributions as dist
from numpyro.infer.autoguide import AutoNormal
from numpyro import handlers
import optax
from jax import random
from jax import numpy as jnp

from rhs.elbo import MultiELBO, site_mask
from rhs.utils import get_sample_params


def model():
    a = numpyro.sample('a', dist.Normal())
    b = numpyro.sample('b', dist.Normal(a))
    c = numpyro.sample('c', dist.Normal(b))


def test_sample_params():
    guide = AutoNormal(model)
    sample_params = get_sample_params(guide)
    for s in ['a', 'b', 'c']:
        for t in ['loc', 'scale']:
            assert f'{s}_auto_{t}' in sample_params[s]
    
class TestSiteMask:
    masked_sites = ['a', 'b']

    def test_mask(self):
        model_masked = site_mask(model, self.masked_sites)
        trace = handlers.trace(handlers.seed(model_masked, 0)).get_trace()
        for site in self.masked_sites:
            assert isinstance(trace[site]['fn'], dist.MaskedDistribution)

    def test_mask_reverse(self):
        model_masked = site_mask(model, self.masked_sites, reverse=True)
        trace = handlers.trace(handlers.seed(model_masked, 0)).get_trace()
        for site in self.masked_sites:
            assert not isinstance(trace[site]['fn'], dist.MaskedDistribution)


class TestMultiELBO():
    def test_append_params(self):
        elbos = {
            ('a', 'b'): numpyro.infer.Trace_ELBO(),
            ('c',): numpyro.infer.Trace_ELBO()
        }
        elbo_multi = MultiELBO.build(elbos, model, AutoNormal(model), append_params=True)
        sites = list(elbo_multi.elbos.keys())
        assert set(sites[0]) == {'a', 'a_auto_loc', 'a_auto_scale',
                                 'b', 'b_auto_loc', 'b_auto_scale'}
        assert set(sites[1]) == {'c', 'c_auto_loc', 'c_auto_scale'}
        
    def test_loss_same(self):
        guide = AutoNormal(model)

        # TODO This gives me weird error if i pass guide in directly
        # https://github.com/pyro-ppl/numpyro/issues/2062
        sample_params = get_sample_params(AutoNormal(model))
        some_sites = ('a', 'b', *sample_params['a'], *sample_params['b'])
        elbos = {
            some_sites: numpyro.infer.Trace_ELBO(),
            None: numpyro.infer.Trace_ELBO()
        }
        elbo_multi = MultiELBO.build(elbos, model, guide, append_params=False)
        assert set(list(elbo_multi.elbos.keys())[1]) == set(['c', 'c_auto_loc', 'c_auto_scale'])

        key = random.key(0)
        optim = optax.adam(.1)

        # Normal elbo
        elbo = numpyro.infer.Trace_ELBO()
        svi = numpyro.infer.SVI(model, guide, optim, elbo)
        res = svi.run(key, num_steps=1, progress_bar=False)

        # Multi elbo
        svi_multi = numpyro.infer.SVI(model, guide, optim, elbo_multi)
        res_multi = svi_multi.run(key, num_steps=1, progress_bar=False)

        assert jnp.allclose(res.losses[0], res_multi.losses[0])
