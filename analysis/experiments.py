from dataclasses import dataclass
from functools import partial

import numpy as np
import numpyro

import rhs


REPARAMS = {
    'base': None,
    **{reparam.name: reparam
       for reparam in [
               rhs.ReparamIG(dec=False),
               rhs.ReparamIG(dec=True),
               rhs.ReparamII(dec1=False, dec2=False),
               rhs.ReparamII(dec1=True, dec2=False),
               rhs.ReparamII(dec1=False, dec2=True),
               rhs.ReparamII(dec1=True, dec2=True)
    ]}
}
# All combinations of ii and ig reparam, and tau and lambda
REPARAM_COMBS = [
    f'{reparam_tau_name}_{reparam_lambda_name}'
    for reparam_type in ['ii', 'ig']
    for reparam_tau_name in [r for r in REPARAMS if r.startswith(reparam_type)]
    for reparam_lambda_name in [r for r in REPARAMS if r.startswith(reparam_type)]    
]
STRUCTURES = {
    structure.name: structure
    for structure in [
            rhs.GuideUnstructured(),
            rhs.GuidePairCond(),
            # rhs.GuidePairCondCorr(),
            # rhs.GuidePairCondCorr(3)
    ]
}
ELBOS = {
    'standard': lambda conf, particles: numpyro.infer.Trace_ELBO(particles),
    'meanfield': lambda conf, particles: numpyro.infer.TraceMeanField_ELBO(particles),
    'mix': lambda conf, particles: rhs.TrainerSVI.build_specialized_elbo(conf, particles)
}

@dataclass(kw_only=True)
class Experiment:
    name: str
    seeds: list[int]
    datasets: dict[str, rhs.datasets.TestDataset]
    reparams: list[str] # = REPARAM_COMBS
    c_df: float = 3.
    c_scale: float = 3.
    noise_scale: float = 1.

    mcmc_experiments = {}
    svi_experiments = {}
    
    def __post_init__(self):
        if isinstance(self, MCMCExperiment):
            self.mcmc_experiments[self.name] = self
        else:
            self.svi_experiments[self.name] = self        

@dataclass(kw_only=True)
class MCMCExperiment(Experiment):
    num_warmup: int = 500
    num_samples: int = 500

@dataclass(kw_only=True)
class SVIExperiment(Experiment):
    structures: list[str]
    elbos: list[str]
    iters: int = 30_000
    num_particles: int = 10


# * Reparam MCMC

MCMCExperiment(
    name = 'mcmc_reparam',
    num_warmup = 500,
    num_samples = 500,
    seeds = list(range(20)),
    datasets = {
        'uncorrelated': rhs.datasets.TestDataset(
            D_per=30, B_k=np.array([3., 3., 3.]), correlated=False, x_std=.5),
        'correlated': rhs.datasets.TestDataset(
            D_per=30, B_k=np.array([3., 3., 3.]), correlated=.9, x_std=.5)
    },
    reparams = [
        'ig:dec_ig:dec',
        'ig:cen_ig:dec',
        'ig:dec_ig:cen',
        'ig:cen_ig:cen',
        
        'ii:dec_ii:dec',
        'ii:dec_ii:dec2',
        'ii:dec2_ii:cen',
    ],
)

# * best ELBO

SVIExperiment(
    name = 'svi_elbo',
    seeds = list(range(10)),
    num_particles = 10,
    iters = 30_000,
    datasets = {
        'uncorrelated': rhs.datasets.TestDataset(
            D_per=30, B_k=np.array([3., 3., 3.]), correlated=False, x_std=.5),
        'correlated': rhs.datasets.TestDataset(
            D_per=30, B_k=np.array([3., 3., 3.]), correlated=.8, x_std=.5)
    },
    # reparams = REPARAM_COMBS,
    reparams = ['ig:dec_ig:dec'],
    structures = list(STRUCTURES.keys()),
    elbos = list(ELBOS.keys())
)

SVIExperiment(
    name = 'svi_elbo_smol',
    seeds = list(range(10)),
    num_particles = 10,
    iters = 30_000,
    datasets = {
        'uncorrelated': rhs.datasets.TestDataset(
            D_per=5, B_k=np.array([3., 3., 3.]), correlated=False, x_std=.5),
        'correlated': rhs.datasets.TestDataset(
            D_per=5, B_k=np.array([3., 3., 3.]), correlated=.8, x_std=.5)
    },
    # reparams = REPARAM_COMBS,
    reparams = ['ig:dec_ig:dec'],
    structures = list(STRUCTURES.keys()),
    elbos = list(ELBOS.keys())
)
