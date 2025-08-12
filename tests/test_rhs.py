import numpyro
import numpyro.distributions as dist
from numpyro import handlers
import optax
from jax import random

from rhs.elbo import MultiELBO


class TestMultiELBO():
    def test_loss_same(self):
        def model():
            a = numpyro.sample('a', dist.Normal())
            b = numpyro.sample('b', dist.Normal(a))
            c = numpyro.sample('c', dist.Normal(b))

        guide = numpyro.infer.autoguide.AutoNormal(model)
        print(handlers.trace(handlers.seed(guide, 0)).get_trace().keys())
            
        elbos = {
            ('a', 'a_auto_loc' , 'a_auto_scale', 'b', 'b_auto_loc', 'b_auto_scale'): numpyro.infer.Trace_ELBO(),
            None: numpyro.infer.Trace_ELBO()
        }
        elbo_multi = MultiELBO.build(elbos, model, guide)
        assert set(list(elbo_multi.elbos.keys())[1]) == set(['c', 'c_auto_loc', 'c_auto_scale'])
        ## Array(11.460761, dtype=float32) == Array(22.921522, dtype=float32)


        key = random.key(0)
        optim = optax.adam(.1)

        # Normal elbo
        elbo = numpyro.infer.Trace_ELBO()
        svi = numpyro.infer.SVI(model, guide, optim, elbo)
        res = svi.run(key, num_steps=1)

        # Multi elbo
        svi_multi = numpyro.infer.SVI(model, guide, optim, elbo_multi)
        res_multi = svi_multi.run(key, num_steps=1)

        assert res.losses[0] == res_multi.losses[0]
