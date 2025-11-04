
# Table of Contents

1.  [Regularized horseshoe prior experiments](#org191a8ae)
    1.  [Guide structures](#org78b697b)
    2.  [Reparameterizations](#org8606913)
        1.  [InverseGamma-Gamma](#org7c74475)
        2.  [InverseGamma-InverseGamma](#org8db47e1)
    3.  [References](#orgb02c8bf)



<a id="org191a8ae"></a>

# Regularized horseshoe prior experiments

I implement regularized horseshoe priors in numpyro and train it with variational inference and MCMC. The main objective is to compare variations of the model:

-   **Guide structure:** Model correlations between coefficient and its local sparsity parameter
-   **Reparamerization:** Factorize half-cauchy prior into gamma and inverse-gamma distributions


<a id="org78b697b"></a>

## Guide structures


<a id="org8606913"></a>

## Reparameterizations

The global and feature-local sparsity parameters are modeled as Half-Cauchy distributions in the prior. This can challenge inference and is therefore proposed to be factorized into two seperate distributions.


<a id="org7c74475"></a>

### InverseGamma-Gamma


<a id="org8db47e1"></a>

### InverseGamma-InverseGamma

<sup><a href="#citeproc_bib_item_1">1</a></sup>


<a id="orgb02c8bf"></a>

## References

<style>.csl-left-margin{float: left; padding-right: 0em;}
 .csl-right-inline{margin: 0 0 0 1em;}</style><div class="csl-bib-body">
  <div class="csl-entry"><a id="citeproc_bib_item_1"></a>
    <div class="csl-left-margin">1.</div><div class="csl-right-inline">Oh, C., Adamczewski, K. &#38; Park, M. Radial and directional posteriors for bayesian neural networks. at <a href="https://doi.org/10.48550/arXiv.1902.02603">https://doi.org/10.48550/arXiv.1902.02603</a> (2019).</div>
  </div>
</div>

