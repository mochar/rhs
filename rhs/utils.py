from typing import Callable

from numpyro.infer.inspect import get_model_relations


def get_sample_params(model: Callable) -> dict[str, list[str]]:
    """Return param names associated with each sample site.
    
    Warning: dont use model after calling this function, leads to error for some
    reason.
    """
    relations = get_model_relations(model)
    return relations['sample_param']
