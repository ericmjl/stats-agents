# R2D2 for Generalized Linear Mixed Models: Summary and PyMC Implementation

## Overview

The R2D2 prior for generalized linear mixed models (GLMMs) extends the original R2D2 shrinkage framework to non-Gaussian responses and group-specific effects (traditionally called "random effects"). This work by Yanchenko, Bondell, and Reich maintains the intuitive interpretation of placing priors on model fit (R²) while handling the complexities of GLMMs through approximation methods.

## Key Extensions from Linear R2D2

### Generalized Linear Mixed Model Framework

The model structure is:
```
η_i = β₀ + X_i β + Σₖ u_{k,g_{ik}}
Y_i | η_i, θ ~ f(y | η_i, θ)
```

Where:
- `η_i` is the linear predictor
- `β` are population-level effects: `βⱼ | φⱼ, W ~ Normal(0, φⱼW)`
- `u_k` are group-specific effects: `u_k | φ_{p+k}, W ~ Normal(0, φ_{p+k}W I_{L_k})`
- `φ` allocates variance across components: `Σⱼ φⱼ = 1`
- `W > 0` controls overall variance of the linear predictor

### Variance Decomposition R² (VaDeR)

For GLMMs, R² is defined as:
```
R² = Var{μ(η)} / [Var{μ(η)} + E{σ²(η)}]
```

Where:
- `μ(η) = E(Y|η)` is the conditional mean function
- `σ²(η) = Var(Y|η)` is the conditional variance function
- `η | β₀, W ~ Normal(β₀, W)` (assumed)

### Key Challenge: Non-Closed Form Solutions

Unlike linear regression where `R² = W/(W+σ²)` gives `W ~ BetaPrime(a,b)`, GLMMs require:
1. **Exact solutions** (few special cases)
2. **Linear approximation** (computationally simple)
3. **Quasi-Monte Carlo** (more accurate approximation)
4. **Generalized Beta Prime approximation** (recommended for practice)

## Mathematical Framework

### Special Cases with Exact Solutions

**Gaussian Linear Regression:**
- `μ(η) = η`, `σ²(η) = σ²`
- `R² = W/(W + σ²)`
- `W ~ BetaPrime(a,b)` when `R² ~ Beta(a,b)`

**Poisson Regression:**
- `μ(η) = σ²(η) = e^η`
- `R² = (e^W - 1)/(e^W - 1 + e^{-β₀-W/2})`
- Requires numerical methods for prior on W

### Approximation Methods

**1. Linear Approximation (Delta Method) - Equation (2.1) in Yanchenko et al.:**
```
Var{μ(η)} ≈ {μ'(β₀)}² W
E{σ²(η)} ≈ σ²(β₀)
R² ≈ W/(W + s²(β₀))
```
Where `s²(β₀) = σ²(β₀)/{μ'(β₀)}²`

**Key Advantage**: This reduces the complex GLMM case back to the simple linear regression form, allowing us to use `W ~ BetaPrime(a,b) × s²(β₀)` when `R² ~ Beta(a,b)`.

#### Linear Approximation Examples

**Gaussian Linear Model:**
- `μ(η) = η`, `σ²(η) = σ²`
- `μ'(η) = 1`, so `s²(β₀) = σ²/1² = σ²`
- **Result**: `W ~ BetaPrime(a,b) × σ²` (exact, not approximation)

**Poisson Regression:**
- `μ(η) = σ²(η) = e^η`
- `μ'(η) = e^η`, so `s²(β₀) = e^{β₀}/e^{2β₀} = e^{-β₀}`
- **Result**: `W ~ BetaPrime(a,b) × e^{-β₀}`

**Logistic Regression:**
- `μ(η) = e^η/(1+e^η)`, `σ²(η) = μ(η)[1-μ(η)]`
- `μ'(η) = e^η/(1+e^η)²`
- `s²(β₀) = [μ(β₀)(1-μ(β₀))]/[μ'(β₀)]² = μ(β₀)(1-μ(β₀))/[μ'(β₀)]²`
- **Result**: `W ~ BetaPrime(a,b) × s²(β₀)` where `s²(β₀)` depends on β₀

**2. Generalized Beta Prime (GBP) Distribution:**
If `V ~ BetaPrime(a,b)` then `W = d V^{1/c} ~ GBP(a,b,c,d)` with density:
```
π(w) = c(w/d)^{ac-1} [1+(w/d)^c]^{-a-b} / [d B(a,b)]
```

Properties:
- Behavior at origin controlled by `ac`
- Tail behavior controlled by `bc`
- Reduces to BetaPrime when `c = d = 1`
- Half-t distribution when `a = 1/2, b = ν/2, c = 2, d = √(νσ²)`

## PyMC Implementation

```python
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "pymc",
#   "numpy",
#   "pandas",
#   "scipy",
#   "pytensor",
# ]
# ///

import pymc as pm
import numpy as np
import pandas as pd
import pytensor.tensor as pt
from scipy.special import gamma as gamma_fn

def generalized_beta_prime_logp(value, a, b, c, d):
    """Log-probability for Generalized Beta Prime distribution."""
    w_scaled = value / d
    w_scaled_c = pt.power(w_scaled, c)

    logp = (
        pt.log(c)
        + (a * c - 1) * pt.log(w_scaled)
        - (a + b) * pt.log(1 + w_scaled_c)
        - pt.log(d)
        - pt.log(gamma_fn(a)) - pt.log(gamma_fn(b)) + pt.log(gamma_fn(a + b))
    )
    return logp

class GeneralizedBetaPrime(pm.Continuous):
    """Generalized Beta Prime distribution for PyMC."""

    def __init__(self, a, b, c=1, d=1, **kwargs):
        self.a = pt.as_tensor_variable(a)
        self.b = pt.as_tensor_variable(b)
        self.c = pt.as_tensor_variable(c)
        self.d = pt.as_tensor_variable(d)
        super().__init__(**kwargs)

    def logp(self, value):
        return generalized_beta_prime_logp(value, self.a, self.b, self.c, self.d)

def r2d2_glmm_linear_approximation(X, y, groups=None, family='gaussian',
                                   a=1, b=1, xi0=0.5, beta0_prior_sd=10):
    """
    R2D2 GLMM with linear approximation method.

    Parameters:
    -----------
    X : array-like, shape (n, p)
        Fixed effects design matrix
    y : array-like, shape (n,)
        Response vector
    groups : array-like, shape (n,), optional
        Group indicators for group-specific effects
    family : str
        'gaussian', 'poisson', 'binomial'
    a, b : float
        Beta parameters for R² ~ Beta(a, b)
    xi0 : float
        Dirichlet concentration parameter
    beta0_prior_sd : float
        Standard deviation for intercept prior
    """
    n, p = X.shape
    has_group_effects = groups is not None

    if has_group_effects:
        n_groups = len(np.unique(groups))
        n_components = p + 1  # population effects + group-specific effects
    else:
        n_groups = 0
        n_components = p

    # Define coordinates for model dimensions
    coords = {
        'predictors': [f'x_{i}' for i in range(p)],
        'obs': range(n),
    }

    if has_group_effects:
        coords['groups'] = range(n_groups)
        coords['components'] = [f'population_{i}' for i in range(p)] + ['group_effects']
    else:
        coords['components'] = [f'predictor_{i}' for i in range(p)]

    with pm.Model(coords=coords) as model:
        # Intercept
        beta0 = pm.Normal("beta0", mu=0, sigma=beta0_prior_sd)

        # Variance allocation (Dirichlet)
        phi = pm.Dirichlet("phi", a=np.full(n_components, xi0), dims="components")

        # Linear approximation parameter s²(β₀)
        if family == 'gaussian':
            # For Gaussian: μ'(η) = 1, σ²(η) = σ²
            sigma_sq = pm.InverseGamma("sigma_sq", alpha=1, beta=1)
            s_sq = sigma_sq  # s²(β₀) = σ²/1² = σ²

        elif family == 'poisson':
            # For Poisson: μ'(η) = e^η, σ²(η) = e^η
            # s²(β₀) = e^β₀ / (e^β₀)² = e^{-β₀}
            s_sq = pm.Deterministic("s_sq", pt.exp(-beta0))

        elif family == 'binomial':
            # For binomial (logit): μ'(η) = e^η/(1+e^η)², σ²(η) = μ(η)[1-μ(η)]
            # Approximation at β₀
            exp_beta0 = pt.exp(beta0)
            mu_beta0 = exp_beta0 / (1 + exp_beta0)
            mu_prime_beta0 = exp_beta0 / pt.power(1 + exp_beta0, 2)
            s_sq = pm.Deterministic("s_sq", mu_beta0 * (1 - mu_beta0) / pt.power(mu_prime_beta0, 2))

        # Global variance W ~ GBP(a, b, 1, s²(β₀))
        W = GeneralizedBetaPrime("W", a=a, b=b, c=1, d=s_sq)

        # Population-level effects
        if has_group_effects:
            population_var = phi[:-1] * W  # All but last component
            beta = pm.Normal("beta", mu=0, sigma=pt.sqrt(population_var), dims="predictors")
        else:
            population_var = phi * W  # Each component gets own variance
            beta = pm.Normal("beta", mu=0, sigma=pt.sqrt(population_var), dims="predictors")

        # Group-specific effects (if present)
        if has_group_effects:
            group_var = phi[-1] * W  # Last component
            group_intercepts = pm.Normal("group_intercepts", mu=0, sigma=pt.sqrt(group_var), dims="groups")

            # Linear predictor with group-specific intercepts
            eta = beta0 + pm.math.dot(X, beta) + group_intercepts[groups]
        else:
            eta = beta0 + pm.math.dot(X, beta)

        # Likelihood
        if family == 'gaussian':
            sigma = pm.Deterministic("sigma", pt.sqrt(sigma_sq))
            likelihood = pm.Normal("y", mu=eta, sigma=sigma, observed=y, dims="obs")

        elif family == 'poisson':
            mu = pm.Deterministic("mu", pt.exp(eta), dims="obs")
            likelihood = pm.Poisson("y", mu=mu, observed=y, dims="obs")

        elif family == 'binomial':
            # Assumes y is in {0,1} or needs to be converted
            p_success = pm.Deterministic("p", pm.math.sigmoid(eta), dims="obs")
            likelihood = pm.Bernoulli("y", p=p_success, observed=y, dims="obs")

        # Derived R² (approximate)
        if family == 'gaussian':
            # R² ≈ W/(W + σ²)
            r_squared = pm.Deterministic("r_squared", W / (W + sigma_sq))
        else:
            # R² ≈ W/(W + s²(β₀))
            r_squared = pm.Deterministic("r_squared", W / (W + s_sq))

    return model

def r2d2_poisson_exact(X, y, groups=None, a=1, b=1, xi0=0.5):
    """
    R2D2 for Poisson regression with exact R² calculation.
    This is more complex but provides the true R² relationship.
    """
    n, p = X.shape
    has_group_effects = groups is not None
    n_components = p + 1 if has_group_effects else p

    # Define coordinates
    coords = {
        'predictors': [f'x_{i}' for i in range(p)],
        'obs': range(n),
        'components': [f'predictor_{i}' for i in range(p)]
    }

    if has_group_effects:
        n_groups = len(np.unique(groups))
        coords['groups'] = range(n_groups)
        coords['components'] = [f'population_{i}' for i in range(p)] + ['group_effects']

    with pm.Model(coords=coords) as model:
        # Intercept
        beta0 = pm.Normal("beta0", mu=0, sigma=10)

        # Variance allocation
        phi = pm.Dirichlet("phi", a=np.full(n_components, xi0), dims="components")

        # For Poisson, we need custom prior on W that gives R² ~ Beta(a,b)
        # This requires numerical methods - here we use a rough approximation
        # In practice, would use QMC or GBP fitting

        # Approximate with log-normal on W (not exact but illustrative)
        log_W = pm.Normal("log_W", mu=0, sigma=2)
        W = pm.Deterministic("W", pt.exp(log_W))

        # Population-level effects
        if has_group_effects:
            population_var = phi[:-1] * W
            beta = pm.Normal("beta", mu=0, sigma=pt.sqrt(population_var), dims="predictors")

            # Group-specific effects
            group_var = phi[-1] * W
            group_intercepts = pm.Normal("group_intercepts", mu=0, sigma=pt.sqrt(group_var), dims="groups")
            eta = beta0 + pm.math.dot(X, beta) + group_intercepts[groups]
        else:
            population_var = phi * W
            beta = pm.Normal("beta", mu=0, sigma=pt.sqrt(population_var), dims="predictors")
            eta = beta0 + pm.math.dot(X, beta)

        # Poisson likelihood
        mu = pm.Deterministic("mu", pt.exp(eta), dims="obs")
        likelihood = pm.Poisson("y", mu=mu, observed=y, dims="obs")

        # Exact R² for Poisson (requires η ~ Normal(β₀, W))
        # R² = (e^W - 1) / (e^W - 1 + e^{-β₀ - W/2})
        exp_W = pt.exp(W)
        numerator = exp_W - 1
        denominator = numerator + pt.exp(-beta0 - W/2)
        r_squared = pm.Deterministic("r_squared", numerator / denominator)

        # Custom potential to encourage R² ~ Beta(a,b)
        # This is approximate - exact implementation would require transform
        r2_logp = (a-1)*pt.log(r_squared) + (b-1)*pt.log(1-r_squared)
        pm.Potential("r2_prior", r2_logp)

    return model

def r2d2_glmm_beta_approximation(X, y, groups=None, family='gaussian',
                                 a=1, b=1, xi0=0.5, beta0_prior_sd=10):
    """
    R2D2 GLMM using simple Beta approximation (linear/delta method).

    This implements the linear approximation from Section 2.2.1 of Yanchenko et al.,
    which reduces the complex GLMM case to the familiar linear regression form:
    R² ≈ W/(W + s²(β₀)) where s²(β₀) = σ²(β₀)/[μ'(β₀)]²

    This allows us to use W ~ BetaPrime(a,b) × s²(β₀) when R² ~ Beta(a,b).

    Parameters:
    -----------
    X : array-like, shape (n, p)
        Fixed effects design matrix
    y : array-like, shape (n,)
        Response vector
    groups : array-like, shape (n,), optional
        Group indicators for group-specific effects
    family : str
        'gaussian', 'poisson', 'binomial'
    a, b : float
        Beta parameters for R² ~ Beta(a, b)
    xi0 : float
        Dirichlet concentration parameter
    beta0_prior_sd : float
        Standard deviation for intercept prior
    """
    n, p = X.shape
    has_group_effects = groups is not None

    if has_group_effects:
        n_groups = len(np.unique(groups))
        n_components = p + 1  # population effects + group-specific effects
    else:
        n_groups = 0
        n_components = p

    # Define coordinates for model dimensions
    coords = {
        'predictors': [f'x_{i}' for i in range(p)],
        'obs': range(n),
    }

    if has_group_effects:
        coords['groups'] = range(n_groups)
        coords['components'] = [f'population_{i}' for i in range(p)] + ['group_effects']
    else:
        coords['components'] = [f'predictor_{i}' for i in range(p)]

    with pm.Model(coords=coords) as model:
        # Intercept
        beta0 = pm.Normal("beta0", mu=0, sigma=beta0_prior_sd)

        # R² prior
        r_squared = pm.Beta("r_squared", alpha=a, beta=b)

        # Variance allocation (Dirichlet)
        phi = pm.Dirichlet("phi", a=np.full(n_components, xi0), dims="components")

        # Linear approximation scaling factor s²(β₀)
        if family == 'gaussian':
            # For Gaussian: s²(β₀) = σ²
            sigma_sq = pm.InverseGamma("sigma_sq", alpha=1, beta=1)
            s_sq = sigma_sq

        elif family == 'poisson':
            # For Poisson: s²(β₀) = e^{-β₀}
            s_sq = pm.Deterministic("s_sq", pt.exp(-beta0))

        elif family == 'binomial':
            # For binomial (logit): s²(β₀) = μ(β₀)(1-μ(β₀))/[μ'(β₀)]²
            exp_beta0 = pt.exp(beta0)
            mu_beta0 = exp_beta0 / (1 + exp_beta0)
            mu_prime_beta0 = exp_beta0 / pt.power(1 + exp_beta0, 2)
            s_sq = pm.Deterministic("s_sq", mu_beta0 * (1 - mu_beta0) / pt.power(mu_prime_beta0, 2))

        # W from Beta Prime relationship: if R² ~ Beta(a,b), then W = R²/(1-R²) ~ BetaPrime(a,b)
        # But we need W_scaled = W × s²(β₀) for the linear approximation
        W_base = pm.Deterministic("W_base", r_squared / (1 - r_squared))
        W = pm.Deterministic("W", W_base * s_sq)

        # Population-level effects
        if has_group_effects:
            population_var = phi[:-1] * W  # All but last component
            beta = pm.Normal("beta", mu=0, sigma=pt.sqrt(population_var), dims="predictors")
        else:
            population_var = phi * W  # Each component gets own variance
            beta = pm.Normal("beta", mu=0, sigma=pt.sqrt(population_var), dims="predictors")

        # Group-specific effects (if present)
        if has_group_effects:
            group_var = phi[-1] * W  # Last component
            group_intercepts = pm.Normal("group_intercepts", mu=0, sigma=pt.sqrt(group_var), dims="groups")

            # Linear predictor with group-specific intercepts
            eta = beta0 + pm.math.dot(X, beta) + group_intercepts[groups]
        else:
            eta = beta0 + pm.math.dot(X, beta)

        # Likelihood
        if family == 'gaussian':
            sigma = pm.Deterministic("sigma", pt.sqrt(sigma_sq))
            likelihood = pm.Normal("y", mu=eta, sigma=sigma, observed=y, dims="obs")

        elif family == 'poisson':
            mu = pm.Deterministic("mu", pt.exp(eta), dims="obs")
            likelihood = pm.Poisson("y", mu=mu, observed=y, dims="obs")

        elif family == 'binomial':
            # Assumes y is in {0,1} or needs to be converted
            p_success = pm.Deterministic("p", pm.math.sigmoid(eta), dims="obs")
            likelihood = pm.Bernoulli("y", p=p_success, observed=y, dims="obs")

        # Derived quantities
        # The approximation gives us: R² ≈ W/(W + s²(β₀))
        # But we constructed W = W_base × s²(β₀), so we need to be careful
        r_squared_approx = pm.Deterministic("r_squared_approx", W / (W + s_sq))

    return model

# Example usage
if __name__ == "__main__":
    # Generate synthetic data
    np.random.seed(42)
    n, p = 100, 5

    # Fixed effects design matrix
    X = np.random.randn(n, p)

    # Group assignment
    groups = np.random.choice(10, size=n)

    # True parameters
    beta_true = np.array([1, -0.5, 0, 0.8, 0])
    group_intercepts_true = np.random.normal(0, 0.5, 10)

    # Generate Poisson responses
    eta_true = 0.5 + X @ beta_true + group_intercepts_true[groups]
    y = np.random.poisson(np.exp(eta_true))

    print("Fitting R2D2 GLMM with Beta approximation...")
    model_beta = r2d2_glmm_beta_approximation(
        X, y, groups=groups, family='poisson', a=2, b=3, xi0=0.1
    )

    with model_beta:
        # Prior predictive check
        prior_pred = pm.sample_prior_predictive(samples=1000, random_seed=123)
        print(f"Prior R² mean: {prior_pred.prior['r_squared'].mean():.3f}")

        # Posterior sampling would go here
        # trace = pm.sample(2000, tune=1000, random_seed=123)

    print("Model created successfully!")
    print(f"Model has {len(model_approx.basic_RVs)} random variables")
```

## Key Implementation Notes

### Approximation Methods

1. **Linear Approximation**: Simple but may be inaccurate for strong nonlinearity
2. **GBP Approximation**: Recommended for practical use - fits flexible 4-parameter distribution
3. **QMC Integration**: More accurate but computationally intensive
4. **Exact Solutions**: Only available for special cases (Gaussian, some Poisson scenarios)

### Computational Considerations

- **GBP Distribution**: Not built into PyMC, requires custom implementation
- **Parameter Estimation**: For GBP approximation, parameters (a*, b*, c*, d*) must be pre-computed
- **Numerical Stability**: Log-space computations recommended for extreme parameter values

### Group-Specific Effects: Why Shared Variance?

**Why do all groups share the same variance component instead of getting individual allocations?**

The key insight is that we're modeling **two different levels of variation**:

**Population-level effects**: Each predictor gets its own variance allocation from φ
**Group-specific effects**: All groups share one variance component: `group_var = φ[-1] × W`

**Hierarchical Modeling Philosophy:**
1. **Exchangeability**: Groups are draws from the same population - School A and School B should have similar amounts of between-group variation
2. **Borrowing strength**: Groups with little data borrow from groups with more data
3. **Identifiability**: Avoids over-parameterization where each group gets its own variance
4. **Separation of concerns**: We allocate variance to the **concept** of "group-level variation" rather than individual groups

**What we model:**
- Groups differ in their **intercept values** (different μⱼ)
- Groups have the **same variance structure** (same σ²)

**Alternative (problematic) approach:**
```python
# This would be problematic:
n_components = p + n_groups  # population effects + one per group
phi = pm.Dirichlet("phi", a=np.full(n_components, xi0))
group_vars = phi[p:] * W  # Each group gets own variance - NO!
```

**Problems with per-group variance allocation:**
- Parameter explosion (10 groups → 10 extra variance components)
- Loss of hierarchical pooling
- Weak identification between group means and group variances

**Bottom line**: We preserve hierarchical borrowing of strength by allocating variance to the concept of "group effects" rather than to individual groups.

### Model Extensions

- **Mixed Effects**: Natural extension through variance decomposition
- **Spatial Random Effects**: Supported with non-diagonal covariance matrices
- **Multiple Link Functions**: Each requires specific R² derivation
- **Bounded R²**: Use four-parameter Beta when R² ∈ [R²_min, R²_max] ≠ [0,1]

## Advantages Over Standard Priors

1. **Intuitive Specification**: Direct control over model fit via R²
2. **Automatic Shrinkage**: No need to specify individual shrinkage parameters
3. **Group Effects Integration**: Natural framework for variance decomposition across population and group levels
4. **Computational Flexibility**: GBP approximation works in standard software
5. **High-Dimensional Performance**: Particularly effective with many parameters

This framework successfully extends R2D2's interpretability advantages to the rich class of generalized linear mixed models while maintaining computational feasibility through clever approximation strategies.
