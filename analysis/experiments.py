from dataclasses import dataclass, field
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
            rhs.GuideCorr(),
            rhs.GuideCorr(3),
            rhs.GuidePairCond(),
            rhs.GuidePairCondCorr(),
            rhs.GuidePairCondCorr(3)
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
    datasets: dict[str, rhs.datasets.Dataset]
    reparams: list[str] # = REPARAM_COMBS
    coef_decs: list[bool] = field(default_factory=lambda: [True])
    tau_scales: list[float] = field(default_factory=lambda: [0]) # 0 = determined by dataset
    c_scales: list[float] = field(default_factory=lambda: [3.])
    c_df: float = 3.
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

# * Shared data

ds_params = dict(
    B_k=np.array([3.0, 3.0, 3.0]),
    x_std=0.1)
D_per_large = 30
D_per_small = 3

ds_sparse_small = rhs.datasets.TestDataset(
    correlated=False, D_per=D_per_small, **ds_params)
ds_correlated_small = rhs.datasets.TestDataset(
    correlated=0.9, D_per=D_per_small, **ds_params)
ds_sparse_large = rhs.datasets.TestDataset(
    correlated=False, D_per=D_per_large, **ds_params)
ds_correlated_large = rhs.datasets.TestDataset(
    correlated=0.9, D_per=D_per_large, **ds_params)

# * best reparam

params = dict(
    seeds=list(range(5)),
    datasets={
        # "sparse": ds_sparse_large,
        "correlated": ds_correlated_large
    },
    reparams=[
        "base_base",
        
        "ig:dec_ig:dec",
        "ig:dec_ig:cen",
        
        "ig:cen_ig:dec",
        "ig:cen_ig:cen",
        
        "ii:dec_ii:dec",
        "ii:dec_ii:dec2",
        "ii:dec2_ii:cen",
    ],
)

# MCMCExperiment(
#     name = 'mcmc_reparam',
#     num_warmup = 500,
#     num_samples = 500,
#     **params
# )

# SVIExperiment(
#     name = 'svi_reparam',
#     num_particles = 10,
#     iters = 30_000,
#     structures=['unstructured'],
#     elbos=['meanfield'],
#     **params)

# * Posterior correlation

params = dict(
    seeds=[0],
    datasets={
        "large": ds_correlated_large,
        "small": ds_correlated_small
    },
    reparams=['ii:dec_ii:dec2'],
    coef_decs=[True],
    tau_scales=[0, 0.001],
    c_scales=[3., 10.]
)

MCMCExperiment(
    name = 'mcmc_post',
    num_warmup = 1000,
    num_samples = 1000,
    **params)

SVIExperiment(
    name = 'svi_post',
    num_particles = 10,
    iters = 30_000,
    structures = list(STRUCTURES.keys()),
    # elbos=['meanfield'],
    elbos = list(ELBOS.keys()),
    **params)

# * particles

for n in [1, 3, 5, 10, 30]:
    SVIExperiment(
        name = f'svi_particles_{n}',
        num_particles = n,
        iters = 30_000,
        structures = ['unstructured'],
        elbos = list(ELBOS.keys()),
        seeds=[0, 1, 2, 3],
        datasets={
            "large": ds_correlated_large,
        },
        reparams=['ii:dec_ii:dec2'],
        coef_decs=[True],
        tau_scales=[0],
        c_scales=[3.])
    
# * best ELBO

# SVIExperiment(
#     name = 'svi_elbo',
#     seeds = list(range(5)),
#     num_particles = 10,
#     iters = 30_000,
#     datasets={
#         # "sparse": ds_sparse,
#         "correlated": ds_correlated
#     },
#     # reparams = REPARAM_COMBS,
#     reparams = ['ig:dec_ig:dec', 'ii:dec_ii:dec2'],
#     structures = list(STRUCTURES.keys()),
#     elbos = list(ELBOS.keys())
# )

# SVIExperiment(
#     name = 'svi_elbo_smol',
#     seeds = list(range(5)),
#     num_particles = 10,
#     datasets={
#         # "sparse": ds_sparse,
#         "correlated": ds_correlated
#     },
#     iters = 30_000,
#     # reparams = REPARAM_COMBS,
#     reparams = ['ig:dec_ig:dec', 'ii:dec_ii:dec2'],
#     structures = list(STRUCTURES.keys()),
#     elbos = list(ELBOS.keys())
# )

