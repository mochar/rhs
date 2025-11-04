Sparsifying priors such as the horseshoe are normally used when there are many features, but only some are predictive. What happens when most features are predictive, but also redundant? Ideally we'd like a sparse solution; a minimal number of features have a non-zero coefficient.<sup><a href="#citeproc_bib_item_1">1</a></sup>

To explore this I implement regularized horseshoe priors in numpyro and train it with variational inference and MCMC with different variations of the model:

-   **Guide structure:** Model correlations between coefficient and its local sparsity parameter
-   **Reparamerization:** Factorize half-cauchy prior into gamma and inverse-gamma distributions

The resutls are discussed on my website.


# Reparameterizations

The global and feature-local sparsity parameters are modeled as half-Cauchy distributions in the prior. Standard exponential family variational approximations struggle to capture the thick Cauchy tails, and using Cauchy approximating family leads to high variance gradients. This can challenge inference and is therefore proposed to be factorized into Gamma and inverse-Gamma distributions.

In variational inference, the advantage is that the KL-divergence between a (inverse) gamma and a log-normal distribution is closed-form. The log-normal distribution can therefore be used as a variational posterior leading to lower variance gradients. In Pyro, we can use `MeanFieldELBO` to use the closed-form solution.


## InverseGamma-Gamma

The square of half-Cauchy $\mathcal{C}^+$ is equal in distribution to a product of Gamma and inverse-Gamma.<sup><a href="#citeproc_bib_item_2">2</a></sup><sup><a href="#citeproc_bib_item_3">3</a></sup>

If $ z \sim \mathcal{C}^+(k) $, then

$ \sqrt{z} = \alpha\beta $, where

$ \alpha \sim \mathcal{G}(\frac{1}{2},k^2), \beta \sim \mathcal{IG}(\frac{1}{2},1) $


## InverseGamma-InverseGamma

The square of half-Cauchy $\mathcal{C}^+$ is a result of sampling successively from two inverse-Gamma distributions.<sup><a href="#citeproc_bib_item_4">4</a></sup><sup><a href="#citeproc_bib_item_5">5</a></sup>

If $ z \sim \mathcal{C}^+(k) $, then

\begin{align*}
a &\sim \text{InvGamma}(\frac{1}{2}, \frac{1}{k^2}) \\
z^2 &\sim \text{InvGamma}(\frac{1}{2}, \frac{1}{a})
\end{align*}


# Guide structures

There exists a tight coupling between the coefficient and its horseshoe determined variance. Mean-field variational posterior will fail to capture this covariance.<sup><a href="#citeproc_bib_item_3">3</a></sup> Therefore different structured guide have been implemented:

-   **Paired multivariate normal:** Model each coeffcient-lambda pair as a multivariate normal.

\begin{align*}
(\hat{\lambda}_d, \theta_d) \sim \text{MvNormal}()
\end{align*}

Note that $\hat{\lambda} = \ln \lambda$ as $\lambda$ is positive.

-   **Paired conditional multivariate normal:** Same but in [conditional form](https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Conditional_distributions).

\begin{align*}
\lambda_d &\sim \text{LogNormal}() \\ 
\theta_d | \lambda_d &\sim \text{Normal}()
\end{align*}

Since $\lambda$ is now univariate, we can calculate its KL-divergence directly to its prior using the gamma reparameterization.

-   **Paired correlated:** ???
-   **Full:** Model the entire coefficient-lambda pair matrix with a matrix normal.


# References

<style>.csl-left-margin{float: left; padding-right: 0em;}
 .csl-right-inline{margin: 0 0 0 1em;}</style><div class="csl-bib-body">
  <div class="csl-entry"><a id="citeproc_bib_item_1"></a>
    <div class="csl-left-margin">1.</div><div class="csl-right-inline">Piironen, J., Paasiniemi, M. &#38; Vehtari, A. <a href="https://doi.org/10.1214/20-ejs1711">Projective inference in high-dimensional problems: Prediction and feature selection</a>. <i>Electronic journal of statistics</i> <b>14</b>, (2020).</div>
  </div>
  <div class="csl-entry"><a id="citeproc_bib_item_2"></a>
    <div class="csl-left-margin">2.</div><div class="csl-right-inline">Oh, C., Adamczewski, K. &#38; Park, M. Radial and directional posteriors for bayesian neural networks. at <a href="https://doi.org/10.48550/arXiv.1902.02603">https://doi.org/10.48550/arXiv.1902.02603</a> (2019).</div>
  </div>
  <div class="csl-entry"><a id="citeproc_bib_item_3"></a>
    <div class="csl-left-margin">3.</div><div class="csl-right-inline">Louizos, C., Ullrich, K. &#38; Welling, M. Bayesian compression for deep learning. at <a href="https://doi.org/10.48550/arXiv.1705.08665">https://doi.org/10.48550/arXiv.1705.08665</a> (2017).</div>
  </div>
  <div class="csl-entry"><a id="citeproc_bib_item_4"></a>
    <div class="csl-left-margin">4.</div><div class="csl-right-inline">Neville, S. E., Ormerod, J. T. &#38; Wand, M. P. <a href="https://doi.org/10.1214/14-EJS910">Mean field variational bayes for continuous sparse signal shrinkage: Pitfalls and remedies</a>. <i>Electronic journal of statistics</i> <b>8</b>, (2014).</div>
  </div>
  <div class="csl-entry"><a id="citeproc_bib_item_5"></a>
    <div class="csl-left-margin">5.</div><div class="csl-right-inline">Ghosh, S., Yao, J. &#38; Doshi-Velez, F. <a href="http://jmlr.org/papers/v20/19-236.html">Model selection in bayesian neural networks via horseshoe priors</a>. <i>Journal of machine learning research</i> <b>20</b>, 1–46 (2019).</div>
  </div>
</div>
