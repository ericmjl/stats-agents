# From Simple to Complex: Understanding the R2D2 Framework Progression

*A journey through R2D2 Shrinkage, GLM, and M2 variants for Bayesian regularization*

When I first encountered the R2D2 (R² inducing Dirichlet Decomposition) framework, I was struck by its intuitive approach to Bayesian regularization. Instead of placing priors on individual regression coefficients and hoping for the best, R2D2 lets you directly specify your beliefs about how much variance the model should explain. But what really fascinated me was how the framework elegantly extends from simple linear regression to complex multilevel models through a series of principled modifications.

This post documents my journey understanding the progression from the basic R2D2 shrinkage prior to its sophisticated multilevel variant (R2D2M2), with stops along the way to explore generalized linear models. What emerged was a beautiful mathematical architecture where each extension builds naturally on the previous, creating a unified framework for modern Bayesian modeling.

## The Foundation: R2D2 Shrinkage Prior

The journey begins with the elegant insight that motivated the original R2D2 framework: why not place a prior directly on the coefficient of determination (R²) rather than fumbling with individual coefficient priors?

### The Core Mathematical Insight

In linear regression, we have the fundamental relationship:
```
R² = signal variance / (signal variance + noise variance) = W / (W + σ²)
```

Where W represents the **signal-to-noise ratio** - literally how many times stronger your model's signal is compared to the noise. This gives us:
- **W = 1**: Signal equals noise (R² = 0.5)
- **W = 4**: Signal is 4 times stronger than noise (R² = 0.8)
- **W = 0.25**: Noise is 4 times stronger than signal (R² = 0.2)

The R2D2 framework starts by placing a Beta prior on R²:
```python
r_squared = pm.Beta("r_squared", alpha=a, beta=b)
W = pm.Deterministic("W", r_squared / (1 - r_squared))
```

This transforms our familiar Beta distribution into a BetaPrime distribution on W, giving us intuitive control over model fit.

### Allocating Signal Strength: The Dirichlet Decomposition

But here's where R2D2 gets clever. Instead of giving each predictor the same variance, it uses a **Dirichlet decomposition** to allocate the total signal strength W across predictors:

```python
# Think of W as your total explanatory budget
phi = pm.Dirichlet("phi", a=np.full(p, a_pi), dims="predictors")
lambda_j = pm.Deterministic("lambda_j", phi * W, dims="predictors")
```

This means `φⱼ × W = λⱼ` answers the question: *"What fraction of the total signal strength does predictor j get?"*

**Example**: If W = 8 (strong signal) and φ = [0.5, 0.3, 0.2], then:
- Predictor 1: λ₁ = 0.5 × 8 = 4 (gets signal-to-noise ratio of 4)
- Predictor 2: λ₂ = 0.3 × 8 = 2.4 (gets signal-to-noise ratio of 2.4)
- Predictor 3: λ₃ = 0.2 × 8 = 1.6 (gets signal-to-noise ratio of 1.6)

This creates **automatic relevance determination** - important predictors get larger λⱼ values (less shrinkage), while irrelevant predictors get smaller λⱼ values (more shrinkage toward zero).

### The Complete R2D2 Shrinkage Model

```python
with pm.Model(coords=coords) as model:
    # R² prior (intuitive model fit control)
    r_squared = pm.Beta("r_squared", alpha=a, beta=b)

    # Global signal-to-noise ratio
    W = pm.Deterministic("W", r_squared / (1 - r_squared))

    # Local variance allocation (competitive)
    phi = pm.Dirichlet("phi", a=np.full(p, a_pi), dims="predictors")
    lambda_j = pm.Deterministic("lambda_j", phi * W, dims="predictors")

    # Coefficients with allocated variance
    scale_j = sigma * pm.math.sqrt(lambda_j / 2)  # For Laplace priors
    beta = pm.Laplace("beta", mu=0, b=scale_j, dims="predictors")

    # Standard linear likelihood
    mu = pm.math.dot(X, beta)
    likelihood = pm.Normal("y", mu=mu, sigma=sigma, observed=y, dims="obs")
```

**Key insight**: All predictors compete for the total signal budget W. If one predictor becomes more important (higher φⱼ), others must become less important. This creates natural sparsity and prevents overfitting.

## First Extension: R2D2 for Generalized Linear Models

The first major challenge came when extending R2D2 to non-Gaussian outcomes. The beautiful relationship `R² = W/(W+σ²)` that made everything work cleanly suddenly becomes complex when dealing with Poisson counts, binary outcomes, or other GLM families.

### The Challenge: No More Simple σ²

In GLMs, the "noise" isn't a simple σ² anymore. Instead, we have:
- **Poisson**: Variance equals the mean (`σ²(η) = e^η`)
- **Binomial**: Variance depends on probability (`σ²(η) = μ(η)[1-μ(η)]`)
- **Gaussian**: Still simple (`σ²(η) = σ²`)

This breaks our clean R² = W/(W+σ²) relationship because now both the signal and noise are functions of the linear predictor η.

### The Elegant Solution: Linear Approximation

The GLM extension uses a brilliant insight from the delta method. (Please don't quiz me on this one!) We approximate the complex GLM relationship around the intercept β₀:

```
R² ≈ W/(W + s²(β₀))
```

Where `s²(β₀) = σ²(β₀)/[μ'(β₀)]²` is the "effective noise" for each GLM family:

- **Gaussian**: `s²(β₀) = σ²` (no change needed!)
- **Poisson**: `s²(β₀) = e^{-β₀}` (depends on baseline rate)
- **Logistic**: `s²(β₀) = μ(β₀)(1-μ(β₀))/[μ'(β₀)]²` (depends on baseline probability)

### What This Achieves

**The genius**: We keep all the interpretability and mathematical structure of the linear R2D2 case, but just compute a smarter "noise" term that respects the GLM family's variance structure.

```python
with pm.Model(coords=coords) as model:
    # Same intuitive R² prior!
    r_squared = pm.Beta("r_squared", alpha=a, beta=b)

    # GLM-specific "effective noise"
    if family == 'poisson':
        s_sq = pm.Deterministic("s_sq", pt.exp(-beta0))
    elif family == 'binomial':
        exp_beta0 = pt.exp(beta0)
        mu_beta0 = exp_beta0 / (1 + exp_beta0)
        mu_prime_beta0 = exp_beta0 / pt.power(1 + exp_beta0, 2)
        s_sq = pm.Deterministic("s_sq", mu_beta0 * (1 - mu_beta0) / pt.power(mu_prime_beta0, 2))

    # Same competitive allocation structure!
    W_base = pm.Deterministic("W_base", r_squared / (1 - r_squared))
    W = pm.Deterministic("W", W_base * s_sq)
    phi = pm.Dirichlet("phi", a=np.full(n_components, xi0), dims="components")
```

**The beautiful progression**: We're essentially asking "what would σ² be if this GLM were actually a linear model?" and using that as our effective noise term. This preserves all the intuitive benefits of R2D2 while handling GLM complexity.

## The Great Leap: R2D2M2 for Multilevel Models

The most sophisticated extension addresses the challenge of multilevel models with multiple grouping factors - the kind of complex experimental designs common in laboratory research.

### The Multilevel Challenge

Consider a laboratory experiment with:
- **Predictors**: Gene expression, Age, Treatment dose
- **Grouping factors**: Mouse ID, MicroRNA ID, Stress condition

Traditional approaches assign independent priors to each effect:
```python
# Traditional (problematic) approach
beta_gene ~ Normal(0, λ_gene²)
beta_age ~ Normal(0, λ_age²)
mouse_effects ~ Normal(0, λ_mouse²)
microRNA_effects ~ Normal(0, λ_microRNA²)
stress_effects ~ Normal(0, λ_stress²)
```

**The problem**: As you add more predictors and grouping factors, the implied R² prior becomes increasingly concentrated near 1, leading to overfitting-prone models that expect near-perfect fit a priori.

### The R2D2M2 Solution: Type-Level Variance Allocation

R2D2M2 extends the Dirichlet decomposition to handle multiple **types** of effects while preserving hierarchical pooling:

```python
# Component calculation for laboratory data
n_components = n_predictors + n_grouping_factors
n_components = 3 + 3 = 6  # gene_expr + age + dose + mouse + microRNA + stress

component_names = [
    'population_gene_expr',    # Population-level effects
    'population_age',
    'population_dose',
    'mouse_intercepts',        # Group-specific intercept types
    'microRNA_intercepts',
    'stress_intercepts'
]
```

**Crucial insight**: We allocate variance to **types** of effects, not individual groups. All mice share one variance component, all microRNAs share another, etc.

### The Complete R2D2M2 Framework

```python
with pm.Model(coords=coords) as model:
    # Same intuitive R² control
    r_squared = pm.Beta("r_squared", alpha=alpha_r2, beta=beta_r2)
    tau_squared = pm.Deterministic("tau_squared", r_squared / (1 - r_squared))

    # Extended Dirichlet allocation across ALL effect types
    phi = pm.Dirichlet("phi", a=np.full(n_components, concentration), dims="components")

    # Population-level effects (first p components)
    population_scale = pm.Deterministic("population_scale",
                                       pt.sqrt(sigma_squared * phi[:p] * tau_squared),
                                       dims="predictors")
    beta = pm.Normal("beta", mu=0, sigma=population_scale, dims="predictors")

    # Group-specific intercepts (remaining components)
    mouse_scale = pm.Deterministic("mouse_scale",
                                  pt.sqrt(sigma_squared * phi[p] * tau_squared))
    mouse_intercepts = pm.Normal("mouse_intercepts", mu=0, sigma=mouse_scale, dims="mice")

    microRNA_scale = pm.Deterministic("microRNA_scale",
                                     pt.sqrt(sigma_squared * phi[p+1] * tau_squared))
    microRNA_intercepts = pm.Normal("microRNA_intercepts", mu=0, sigma=microRNA_scale, dims="microRNAs")

    # Linear predictor combining all effects
    eta = (pm.math.dot(X, beta) +
           mouse_intercepts[mouse_ids] +
           microRNA_intercepts[microRNA_ids] +
           stress_intercepts[stress_conditions])
```

### Why This Works So Well

**Hierarchical pooling preserved**: Individual mice still borrow strength from each other because they share the same variance component. Mouse A and Mouse B both use `mouse_scale`, but have different intercept values.

**Automatic factor importance**: The φ allocation tells you which experimental factors matter most. If φ = [0.15, 0.25, 0.05, 0.35, 0.15, 0.05], then mouse differences account for 35% of total explained variance - more than any single predictor!

**Scalable complexity**: Works with any number of crossed or nested grouping factors without parameter explosion.

## The Unified Architecture

What strikes me most about this progression is how each extension preserves the core insights while elegantly handling new complexity:

### 1. **Consistent R² Control**
All three variants let you directly specify beliefs about model fit through the same Beta prior on R².

### 2. **Competitive Variance Allocation**
The Dirichlet mechanism creates healthy competition between effects across all variants, preventing any single component from dominating.

### 3. **Interpretable Results**
Every variant produces φ components that directly tell you "what percentage of explained variance does each effect contribute?"

### 4. **Mathematical Elegance**
Each extension modifies just what needs to change:
- **GLM**: Changes the noise term (σ² → s²(β₀))
- **M2**: Extends the allocation to multiple effect types

### 5. **Practical Benefits**
All variants provide automatic shrinkage, sparsity induction, and protection against overfitting while maintaining computational tractability.

## When to Use What

Through this exploration, clear use cases emerged:

**R2D2 Shrinkage**: Simple linear regression with multiple predictors, no grouping
- *Example*: Gene expression ~ drug dose + age + weight

**R2D2 GLM**: Non-Gaussian outcomes with simple structure
- *Example*: Bacterial counts, binary outcomes, rate data

**R2D2M2**: Complex laboratory designs with multiple grouping factors (**the laboratory default**)
- *Example*: Laboratory experiments with mouse ID + microRNA ID + stress condition

## Looking Forward

The R2D2 framework represents a fundamental shift in how we think about Bayesian regularization. Instead of getting lost in the weeds of individual coefficient priors, we can work at the level of model fit and variance decomposition - concepts that align much better with how scientists actually think about their experiments.

For laboratory researchers especially, R2D2M2 offers something remarkable: a principled way to automatically determine which experimental factors matter most, while preserving all the benefits of hierarchical modeling. When your model tells you that "mouse differences account for 35% of explained variance while stress conditions only account for 5%," you're getting scientific insight, not just statistical output.

The progression from simple to complex reveals the deep mathematical unity underlying modern Bayesian modeling. Sometimes the most sophisticated methods are just elegant extensions of simple, powerful ideas.

---

*This exploration was motivated by implementing R2D2 variants for laboratory data modeling. All code examples use PyMC and follow the mathematical frameworks described in the original papers by Zhang et al. (R2D2), Yanchenko et al. (GLM), and Aguilar & Bürkner (M2).*
