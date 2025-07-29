# R2D2 Shrinkage Prior: Summary and PyMC Implementation

## Overview

The R2D2 (R² inducing Dirichlet Decomposition) shrinkage prior is a Bayesian regularization approach that places a prior directly on the coefficient of determination (R²) rather than on the regression coefficients themselves. This creates a natural framework for controlling model complexity while maintaining interpretability.

## Key Mathematical Framework

### R² Representation

In the marginal formulation (Equation 1 in Naughton & Bondell 2020), R² is expressed as:

```
R² = var(x^T β) / (var(x^T β) + σ²) = W / (W + 1)
```

where `W = Σⱼ λⱼ` is the sum of scaled prior variances (Equation 2).

### Understanding W: The Signal-to-Noise Ratio

**W represents the "signal-to-noise ratio" of your model** - it measures how much variance your predictors explain compared to the error variance.

**Intuitive Interpretation:**
- **W = 1**: Your model's signal equals the noise - you're explaining as much as you're missing
- **W = 4**: Your signal is 4 times stronger than noise - you're explaining much more than you're missing
- **W = 0.25**: Your noise is 4 times stronger than signal - you're mostly capturing noise

**Connection to R²:**
Since `R² = W/(W + 1)`, W provides a direct way to understand model fit:
- **W = 0** → **R² = 0** (no signal, pure noise)
- **W = 1** → **R² = 0.5** (signal equals noise)
- **W = 4** → **R² = 0.8** (strong signal)
- **W → ∞** → **R² → 1** (perfect signal)

**Why W is Useful:**
1. **Scale-free interpretation**: W = 2 always means "signal is twice as strong as noise" regardless of data scale
2. **Direct relationship to R²**: You can immediately see how much variance you're explaining
3. **Comparable across models**: W has the same meaning across different datasets and model specifications

### Prior Structure

The R2D2 shrinkage prior follows a global-local framework (Section 3.2):

1. **Global component**: `ω ~ BetaPrime(a, b)` controls overall shrinkage (Equation 3)
2. **Local components**: `φ ~ Dirichlet(a_π, ..., a_π)` allocates variance across coefficients
3. **Individual variances**: `λⱼ = φⱼω` (Equation 4)
4. **Coefficients**: `βⱼ | σ², λⱼ ~ DoubleExponential(σ√(λⱼ/2))` (Equation 5)

### Understanding φ × W: Allocating Signal Strength

**The multiplication `φⱼ × W = λⱼ` means "what fraction of the total signal strength does predictor j get?"**

**The Budget Allocation Analogy:**
Think of W as your **total explanatory budget** - the overall signal strength your model has to work with. The Dirichlet vector φ acts like a **budget allocation** that divides this total signal among your predictors.

**Example:**
- **W = 8** (total signal is 8 times stronger than noise)
- **φ = [0.5, 0.3, 0.2]** for 3 predictors
- **Individual allocations:**
  - Predictor 1: λ₁ = 0.5 × 8 = 4 (gets signal-to-noise ratio of 4)
  - Predictor 2: λ₂ = 0.3 × 8 = 2.4 (gets signal-to-noise ratio of 2.4)
  - Predictor 3: λ₃ = 0.2 × 8 = 1.6 (gets signal-to-noise ratio of 1.6)

**What This Creates:**
1. **Competitive allocation**: Predictors "compete" for explanatory power. If one predictor gets more signal strength (higher φⱼ), others must get less.
2. **Automatic relevance determination**: Important predictors get larger λⱼ values → less shrinkage; irrelevant predictors get smaller λⱼ values → more shrinkage toward zero.
3. **Interpretable trade-offs**: You can say "predictor 1 is responsible for 50% of the model's total explanatory power" when φ₁ = 0.5.

### Understanding Signal Variance = W × σ²

**Why does signal variance equal W times the noise variance σ²?**

**Because W is the signal-to-noise ratio!**

**The Mathematical Logic:**
- **W = signal-to-noise ratio = signal variance / noise variance**
- **W = signal variance / σ²**
- **Therefore: signal variance = W × σ²**

**Intuitive Example:**
- **σ² = 4** (noise variance)
- **W = 3** (signal is 3 times stronger than noise)
- **Signal variance = 3 × 4 = 12**

This makes perfect sense: if your noise has variance 4, and your signal is 3 times as strong, then your signal must have variance 12.

**Why This Is Elegant:**
1. **Direct interpretation**: You immediately see how much signal you have relative to your noise level
2. **Scale awareness**: If noise doubles (σ² increases), signal variance scales proportionally with the same W
3. **R² connection**: Since `R² = signal variance / (signal variance + σ²)`, we get `R² = (W×σ²) / (W×σ² + σ²) = W/(W+1)`

### Key Relationships

- `R² ~ Beta(a, b)` (target distribution)
- `W = R²/(1-R²) ~ BetaPrime(a, b)`
- Smaller `a_π` promotes sparsity by concentrating `φⱼ` values
- The prior variance structure is `Cov(β) = σ² Λ` where `Λ = diag(λ₁, ..., λₚ)`

### How R², W, and σ Interact: The Complete Picture

**The key insight: R² = W/(W+1) means R² depends ONLY on W, not directly on σ!**

**Understanding the Relationships:**

Since `R² = signal variance / total variance = (W×σ²) / (W×σ² + σ²) = W/(W+1)`, we see that:

**What happens when W changes (signal strength)?**
- **W increases** → **R² increases** (more signal relative to noise)
- **W decreases** → **R² decreases** (less signal relative to noise)
- **W = 1** → **R² = 0.5** (signal equals noise)
- **W = 9** → **R² = 0.9** (signal is 9 times stronger than noise)

**What happens when σ changes (noise level)?**
- **σ increases** → **R² stays the same!** (both signal and noise scale up proportionally)
- **σ decreases** → **R² stays the same!** (both signal and noise scale down proportionally)

**The Crucial Insight:**
R² is **scale-invariant** with respect to σ because it's a ratio. If you double the noise (σ²), the signal variance (W×σ²) also doubles, so the ratio stays constant.

**Example Scenarios:**
1. **Scenario A**: W=4, σ²=1 → Signal=4, Noise=1 → R²=4/5=0.8
2. **Scenario B**: W=4, σ²=100 → Signal=400, Noise=100 → R²=400/500=0.8
3. **Scenario C**: W=1, σ²=1 → Signal=1, Noise=1 → R²=1/2=0.5

**Key Takeaway:** W controls the signal-to-noise ratio and thus R², while σ controls the absolute scale of both signal and noise without affecting their ratio.

## PyMC Implementation

```python
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "pymc",
#   "numpy",
#   "pandas",
#   "scipy",
# ]
# ///

import pymc as pm
import numpy as np
import pandas as pd
from scipy.stats import betaprime

def r2d2_shrinkage_model(X, y, a=1, b=1, a_pi=0.5, sigma_alpha=1, sigma_beta=1):
    """
    Illustrative R2D2 shrinkage prior implementation in PyMC.

    This implements the marginal R2D2 prior from Naughton & Bondell (2020)
    with Laplace (Double Exponential) priors on the coefficients.

    Parameters:
    -----------
    X : array-like, shape (n, p)
        Design matrix
    y : array-like, shape (n,)
        Response vector
    a, b : float
        Beta parameters for R² ~ Beta(a, b)
    a_pi : float
        Dirichlet concentration parameter (smaller = more sparse)
    sigma_alpha, sigma_beta : float
        Inverse-Gamma parameters for σ²
    """
    n, p = X.shape

    # Define coordinates for model dimensions
    coords = {
        'predictors': [f'x_{i}' for i in range(p)],
        'obs': range(n),
    }

    with pm.Model(coords=coords) as model:
        # Prior on error variance
        sigma_squared = pm.InverseGamma("sigma_squared", alpha=sigma_alpha, beta=sigma_beta)
        sigma = pm.Deterministic("sigma", pm.math.sqrt(sigma_squared))

        # R² prior (induces BetaPrime on W)
        r_squared = pm.Beta("r_squared", alpha=a, beta=b)

        # Global shrinkage parameter W = R²/(1-R²)
        W = pm.Deterministic("W", r_squared / (1 - r_squared))

        # Local shrinkage: Dirichlet allocation of variance
        phi = pm.Dirichlet("phi", a=np.full(p, a_pi), dims="predictors")

        # Individual variance components
        lambda_j = pm.Deterministic("lambda_j", phi * W, dims="predictors")

        # R2D2 coefficients with Laplace priors
        # βⱼ | σ², λⱼ ~ DE(σ√(λⱼ/2))
        # Note: Could also use Normal priors instead of Laplace
        scale_j = sigma * pm.math.sqrt(lambda_j / 2)
        beta = pm.Laplace("beta", mu=0, b=scale_j, dims="predictors")

        # Linear predictor and likelihood
        mu = pm.math.dot(X, beta)
        likelihood = pm.Normal("y", mu=mu, sigma=sigma, observed=y, dims="obs")

        # Derived quantities for interpretation
        signal_variance = pm.Deterministic("signal_variance", W * sigma_squared)
        total_variance = pm.Deterministic("total_variance", signal_variance + sigma_squared)

    return model

def r2d2_alternative_representation(X, y, a=1, b=1, a_pi=0.5):
    """
    Alternative R2D2 representation using the two-gamma decomposition
    of BetaPrime distribution: BP(a,b) ≡ Ga(a,ξ)/ξ where ξ ~ Ga(b,1)
    """
    n, p = X.shape

    # Define coordinates for model dimensions
    coords = {
        'predictors': [f'x_{i}' for i in range(p)],
        'obs': range(n),
    }

    with pm.Model(coords=coords) as model:
        # Error variance
        sigma_squared = pm.InverseGamma("sigma_squared", alpha=1, beta=1)
        sigma = pm.Deterministic("sigma", pm.math.sqrt(sigma_squared))

        # Two-gamma representation of BetaPrime(a,b)
        xi = pm.Gamma("xi", alpha=b, beta=1)
        omega = pm.Gamma("omega", alpha=a, beta=xi)

        # Dirichlet local shrinkage
        phi = pm.Dirichlet("phi", a=np.full(p, a_pi), dims="predictors")

        # When a = p × a_π, we get λⱼ ~ BP(a_π, b) independently
        lambda_j = phi * omega

        # Laplace coefficients
        scale_j = sigma * pm.math.sqrt(lambda_j / 2)
        beta = pm.Laplace("beta", mu=0, b=scale_j, dims="predictors")

        # Likelihood
        mu = pm.math.dot(X, beta)
        likelihood = pm.Normal("y", mu=mu, sigma=sigma, observed=y, dims="obs")

        # Derived R²
        W = pm.Deterministic("W", omega)
        r_squared = pm.Deterministic("r_squared", W / (W + 1))

    return model

def normal_mixture_representation(X, y, a=1, b=1):
    """
    Normal mixture representation for conditional R² prior (uniform-on-ellipsoid).
    This corresponds to Proposition 2 in the paper.

    Valid only when a ≤ p/2.
    """
    n, p = X.shape

    if a > p/2:
        raise ValueError(f"Normal mixture valid only when a ≤ p/2. Got a={a}, p/2={p/2}")

    # Define coordinates for model dimensions
    coords = {
        'predictors': [f'x_{i}' for i in range(p)],
        'obs': range(n),
    }

    with pm.Model(coords=coords) as model:
        # Error variance
        sigma_squared = pm.InverseGamma("sigma_squared", alpha=1, beta=1)

        # Mixture components for uniform-on-ellipsoid prior
        z = pm.InverseGamma("z", alpha=b, beta=n/2)
        w = pm.Beta("w", alpha=a, beta=p/2 - a)

        # Shrinkage factor
        c = pm.Deterministic("c", z * w / (z * w + 1))

        # Coefficients (shrunken toward zero)
        # β | σ², z, w ~ N(0, c σ² (X'X)⁻¹)
        precision_matrix = pm.math.dot(X.T, X) / sigma_squared
        cov_matrix = c * pm.math.matrix_inverse(precision_matrix)

        beta = pm.MvNormal("beta", mu=np.zeros(p), cov=cov_matrix, dims="predictors")

        # Likelihood
        mu = pm.math.dot(X, beta)
        likelihood = pm.Normal("y", mu=mu, sigma=pm.math.sqrt(sigma_squared), observed=y, dims="obs")

        # Derived R²
        signal_var = pm.math.dot(pm.math.dot(beta, X.T @ X), beta)
        r_squared = pm.Deterministic("r_squared", signal_var / (signal_var + n * sigma_squared))

    return model

# Example usage
if __name__ == "__main__":
    # Generate synthetic data
    np.random.seed(42)
    n, p = 100, 10

    # True sparse coefficients
    beta_true = np.zeros(p)
    beta_true[:3] = [2, -1.5, 1]  # Only first 3 coefficients non-zero

    X = np.random.randn(n, p)
    y = X @ beta_true + 0.5 * np.random.randn(n)

    # Fit R2D2 model
    print("Fitting R2D2 shrinkage model...")
    model = r2d2_shrinkage_model(X, y, a=1, b=1, a_pi=0.1)  # Small a_pi for sparsity

    with model:
        # Prior predictive check
        prior_pred = pm.sample_prior_predictive(samples=1000, random_seed=123)
        print(f"Prior R² mean: {prior_pred.prior['r_squared'].mean():.3f}")

        # Posterior sampling would go here
        # trace = pm.sample(2000, tune=1000, random_seed=123)

    print("Model created successfully!")
    print(f"Model has {len(model.basic_RVs)} random variables")
```

## Key Properties

### Shrinkage Behavior
- **Global shrinkage**: Controlled by R² prior Beta(a,b)
- **Local adaptation**: Dirichlet allocation allows differential shrinkage
- **Sparsity**: Small `a_π` concentrates `φⱼ` on few components

### Theoretical Properties
1. **Posterior contraction**: Under regularity conditions, posterior concentrates around true parameters
2. **Oracle properties**: Can achieve optimal rates for sparse estimation
3. **Computational tractability**: Gibbs sampling available with conjugate structure

### Comparison to Other Priors
- **Horseshoe**: R2D2 provides more direct control via R²
- **LASSO**: R2D2 is fully Bayesian with automatic λ selection
- **Ridge**: R2D2 adapts shrinkage locally rather than globally

## Implementation Notes

The PyMC implementation above is illustrative and demonstrates the core mathematical structure. For practical use:

1. **Hyperparameter selection**: Choose `a_π` based on expected sparsity level
2. **Computational considerations**: Large `p` may require specialized samplers
3. **Model checking**: Verify R² posterior aligns with prior beliefs
4. **Alternative base distributions**: The choice between Laplace vs Normal base distributions

### Base Distribution Choices: Laplace vs Normal

**The R2D2 framework is flexible regarding base distributions for coefficients.** While the original paper uses Laplace (Double Exponential) priors, **Normal priors work equally well** within the same mathematical framework.

**Laplace Prior Advantages:**
- **Stronger sparsity**: Higher concentration of mass around zero promotes more aggressive shrinkage of small coefficients
- **Heavier tails**: Less shrinkage of large coefficients compared to Normal
- **Natural sparsity**: More "spike-and-slab" behavior that naturally separates relevant from irrelevant predictors

**Normal Prior Advantages:**
- **Computational efficiency**: Differentiable everywhere, works better with gradient-based samplers (HMC/NUTS)
- **Conjugacy**: Easier analytical derivations in some contexts
- **Smoother shrinkage**: More gradual shrinkage profile

**To use Normal base distributions instead:**
```python
# Replace Laplace with Normal
# For Laplace: scale = σ√(λⱼ/2)
# For Normal: scale = σ√(λⱼ)
scale_j = sigma * pm.math.sqrt(lambda_j)
beta = pm.Normal("beta", mu=0, sigma=scale_j, dims="predictors")
```

**Practical recommendation**: Use Normal priors for computational convenience with HMC/NUTS, or Laplace priors when you want stronger sparsity-inducing behavior.

This framework provides a principled approach to Bayesian variable selection and regularization through direct modeling of the coefficient of determination.
