from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpyro
from numpyro import handlers
from numpyro.distributions import Normal
from jax import numpy as jnp

from . import TrainerSVI


class Dataset:
    pass


@dataclass
class SimDataset(Dataset):
    N: int
    D_per: int
    correlation: float
    K: int
    seed: int = 0
    X: ArrayLike = field(init=False)
    Y: ArrayLike = field(init=False)
    tau_scale: float = field(init=False)

    def __post_init__(self):
        self.D = self.D_per * self.K

        def m():
            with numpyro.plate('N', self.N):
                with numpyro.plate('K', self.K):
                    f = numpyro.sample('f', Normal(0, 1))
                    with numpyro.plate('D', self.D_per):
                        numpyro.sample('x', Normal(jnp.sqrt(self.correlation) * f,
                                                   jnp.sqrt(1-self.correlation)))
                mu = f.sum() / jnp.sqrt(self.K)
                numpyro.sample('y', Normal(mu, 1))
        trace = handlers.trace(handlers.seed(m, self.seed)).get_trace()
        # self.X = trace['x']['value'].reshape(self.N, -1) # (D_per, K, N) -> (N, D_per*K)
        self.X = trace['x']['value'].T.reshape(self.N, -1) # (D_per, K, N) -> (N, D_per*K)
        self.Y = trace['y']['value']
        
        # Calculate scale of tau according to Piironen & Vehtari (2017)
        self.tau_scale = self.K / ((self.D-self.K)*np.sqrt(self.N))


@dataclass
class TestDataset(Dataset):
    N: int = 100
    D_per: int = 10
    x_std: float = 1.0
    correlated: bool | float = True
    group_std: float = 1.
    B_k: NDArray = field(default_factory=lambda: np.array([2.0, -1.0, 1.])) #*4
    K: int = field(init=False)
    B: ArrayLike = field(init=False)
    B_k_std: NDArray = field(init=False)
    X_k: ArrayLike = field(init=False)
    X: ArrayLike = field(init=False)
    Y: ArrayLike = field(init=False)
    Y_std: ArrayLike = field(init=False)
    tau_scale: float = field(init=False)

    def __post_init__(self):
        self.K = len(self.B_k)
        self.D = self.D_per * self.K

        if self.correlated is not False: 
            self.X = np.empty((self.N, self.D))
            self.B = np.empty(self.D)

            for k in range(self.K):
                cor = self.correlated if isinstance(self.correlated, float) else .9
                corr_matrix = np.full((self.D_per, self.D_per), cor)
                np.fill_diagonal(corr_matrix, 1.0)
                L = np.linalg.cholesky(corr_matrix)
                # Z = np.random.normal(scale=self.group_std, size=(self.N, self.D_per))
                Z = np.random.normal(scale=self.x_std, size=(self.N, self.D_per))
                i = k*self.D_per
                self.X[:, i:i+self.D_per] = Z @ L.T
                self.B[i:i+self.D_per] = self.B_k[k] #/ self.D_per

            self.X_k = self.X[:, np.arange(0, self.D, self.D_per)]
        else:
            # The latent K signals that correlate with outcome
            self.X_k = np.random.normal(size=(self.N, self.K)) * self.x_std
            
            # When uncorrelated, each feature is again randomly drawn from a
            # gaussian, but one for each group is replaced by the latent signal
            # of the corresponding group.
            self.X = np.random.normal(size=(self.N, self.D)) * self.x_std
            for k in range(self.K):
                self.X[:, k*self.D_per] = self.X_k[:, k]

            self.B = np.zeros(self.D)
            self.B[np.arange(0, self.D, self.D_per)] = self.B_k

        # Compute Y using uncorrupted, "true" X, before adding noise
        self.Y = self.X @ self.B

        # Standardize X
        # self.X = np.random.normal(self.X, self.x_std)
        self.X = (self.X-self.X.mean(0)) / self.X.std(0)
        
        # Standardize Y
        self.Y = np.random.normal(self.Y, .1)
        self.Y_std = self.Y.std()
        self.Y = (self.Y-self.Y.mean()) / self.Y.std()

        # Standardized B_k: account for X scaling
        self.B_k_std = self.B_k / self.Y_std

        # Calculate scale of tau according to Piironen & Vehtari (2017)
        self.tau_scale = self.K / ((self.D-self.K)*np.sqrt(self.N))

    def plot_coeffs(self, trainer: TrainerSVI, with_samples: bool = False, annotate: bool = True, axs=None):
        coefs = trainer.estimates['coef']
        lambd = trainer.estimates['lambda']
        if axs is None:
            fig, axs = plt.subplots(nrows=2, sharex=True)

        if with_samples:
            samples = trainer.sample_posterior(10)
            for i, site in enumerate(['coef', 'lambda']):
                sns.scatterplot(data=pd.DataFrame(samples[site]).melt(),
                                x='variable', y='value', ax=axs[i],
                                alpha=.5, c='steelblue', lw=0, marker='*')
        
        axs[0].scatter(range(self.D), coefs)
        axs[0].set(title='Coefficient')
        axs[1].scatter(range(self.D), lambd)
        axs[1].set(title='Lambda')

        if annotate:
            for ax in axs:
                # Shade regions by group
                ax.axvspan(-.5, self.D_per-.5, alpha=0.1, color='k')
                ax.axvspan(self.D-self.D_per-.5, self.D-.5, alpha=0.1, color='k')
            
                # Zero line
                ax.axhline(0., c='k', alpha=.5, lw=1, ls='--')


            # Draw true coefficient line
            for i in range(self.K):
                axs[0].plot([self.D_per*i, self.D_per*(i+1)], [self.B_k[i]/self.Y_std]*2, c='r')
                    
        # if trainer.guide is not guide_factorized:
            # if trainer.guide is guide_mvn:
                # corrs = ((L:=trainer.params['lambda_coef_cholcorr']) @ L.mT)[:, 1, 0]
            # elif trainer.guide is guide_mvn_cond:
                # corrs = trainer.params['lambda_coef_corr']
            # fig, ax = plt.subplots(figsize=(6.3, 2.2))
            # ax.scatter(range(self.D), corrs)
            # ax.axvspan(-.5, dataset.D_per-.5, alpha=0.1, color='k')
            # ax.axvspan(D-dataset.D_per-.5, dataset.D-.5, alpha=0.1, color='k')
            # ax.axhline(0, c='k', alpha=.5, zorder=0)
            # ax.set(title='Corr')
            
