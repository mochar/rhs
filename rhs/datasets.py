from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray
import matplotlib.pyplot as plt

from . import TrainerSVI


@dataclass
class TestDataset:
    N: int = 100
    K: int = 3
    D_per: int = 10
    x_std: float = 1.0
    correlated: bool = True
    B_k: NDArray = field(default_factory=lambda: np.array([2.0, -1.0, 1.])) #*4
    B_k_std: NDArray = field(init=False)
    X: ArrayLike = field(init=False)
    Y: ArrayLike = field(init=False)
    Y_std: ArrayLike = field(init=False)

    def __post_init__(self):
        self.D = self.D_per * self.K
        C = np.arange(self.K).repeat((self.D_per))
        
        #V = np.eye(K)*3.
        #cor = 0.999
        #scales = 2.
        #cov = cor * 2*scales
        #U = np.full((D_per, D_per), cov)
        #np.fill_diagonal(U, scales**2)
        #X = multivariate_normal(cov=np.kron(V, U)).rvs(N)
        #X = (X-X.mean(0)) / X.std(0)

        X_k = np.random.normal(size=(self.N, self.K)) * self.x_std
        if self.correlated:
            M = np.eye(self.K).repeat(self.D_per, 0).T
            self.X = np.random.normal(X_k@M, .1)
        else:
            self.X = np.random.normal(size=(self.N, self.D)) * self.x_std
            for k in range(self.K):
                self.X[:, k*self.D_per] = X_k[:, k]
        self.X = (self.X-self.X.mean(0)) / self.X.std(0)
        
        self.Y = X_k @ self.B_k
        self.Y = np.random.normal(self.Y, 1.)
        self.Y_std = self.Y.std()
        self.Y = (self.Y-self.Y.mean()) / self.Y.std()
        self.B_k_std = self.B_k/self.Y_std

    def plot_coeffs(self, trainer: TrainerSVI):
        coefs = trainer.estimates['coef']
        lambd = trainer.estimates['lambda']
        fig, axs = plt.subplots(nrows=2, sharex=True)
        axs[0].scatter(range(self.D), coefs)
        axs[0].set(title='Coefficient')
        axs[1].scatter(range(self.D), lambd)
        axs[1].set(title='Lambda')
        for ax in axs:
            ax.axvspan(-.5, self.D_per-.5, alpha=0.1, color='k')
            ax.axvspan(self.D-self.D_per-.5, self.D-.5, alpha=0.1, color='k')
            
        for i in range(3):
            axs[0].plot([self.D_per*i, self.D_per*(i+1)], [self.B_k[i]/self.Y_std]*2, c='r')
        
        # if trainer.guide is not guide_factorized:
            # if trainer.guide is guide_mvn:
                # corrs = ((L:=trainer.params['coef_lambda_cholcorr']) @ L.mT)[:, 1, 0]
            # elif trainer.guide is guide_mvn_cond:
                # corrs = trainer.params['coef_lambda_corr']
            # fig, ax = plt.subplots(figsize=(6.3, 2.2))
            # ax.scatter(range(self.D), corrs)
            # ax.axvspan(-.5, dataset.D_per-.5, alpha=0.1, color='k')
            # ax.axvspan(D-dataset.D_per-.5, dataset.D-.5, alpha=0.1, color='k')
            # ax.axhline(0, c='k', alpha=.5, zorder=0)
            # ax.set(title='Corr')
            
