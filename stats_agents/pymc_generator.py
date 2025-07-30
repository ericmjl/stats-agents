"""PyMC code generation bot using llamabot StructuredBot."""

from typing import Optional

import llamabot as lmb
from pydantic import BaseModel, Field

from .schemas import ExperimentDescription


class PyMCModelResponse(BaseModel):
    """Response model for PyMC code generation."""

    model_code: str = Field(
        ..., description="Complete PEP 723 compliant PyMC model code"
    )
    python_version: str = Field(
        default="3.12", description="Minimum Python version required"
    )
    dependencies: list[str] = Field(
        default=["pymc", "pandas"], description="List of required dependencies"
    )

    def render_code(self) -> str:
        """Render the complete PEP 723 compliant code."""
        header_lines = [
            "# /// script",
            f'# requires-python = ">={self.python_version}"',
            "# dependencies = [",
        ]
        for dep in self.dependencies:
            header_lines.append(f'#   "{dep}",')
        header_lines.extend(["# ]", "# ///", ""])
        return "\n".join(header_lines) + self.model_code

    def write_to_disk(self, filepath: Optional[str] = None) -> str:
        """Write the rendered code to disk."""
        import tempfile
        from pathlib import Path

        rendered_code = self.render_code()
        if filepath is None:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(rendered_code)
                return f.name
        else:
            path = Path(filepath)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(rendered_code)
            return str(path)


@lmb.prompt(role="system")
def pymc_generation_system():
    """You are an expert in Bayesian modeling with PyMC and the R2D2 framework.

    Your task is to generate PEP 723 compliant PyMC model code based on a structured experiment description.

    ## R2D2 Framework Overview
    R2D2 (R-squared Dirichlet Decomposition) is a Bayesian regularization framework for variance decomposition in probabilistic models. For laboratory experiments, use R2D2M2 (multilevel/hierarchical models).

    ### R2D2M2 Key Principles:
    1. **Signal Factor W**: Total signal-to-noise ratio (W = τ² = R²/(1-R²))
    2. **Variance Allocation**: Use Dirichlet distribution to allocate signal strength across components
    3. **R² Interpretation**: Each φ component represents its proportion of total explained variance
    4. **Hierarchical Structure**: Handle nested effects (e.g., treatment within blocks)

    ### R2D2M2 Implementation Pattern:
    ```python
    import pymc as pm
    import numpy as np
    import pytensor.tensor as pt

    with pm.Model(coords=coords) as model:
        # 1. Intercept (not regularized)
        b0 = pm.Normal("b0", mu=np.mean(y), sigma=np.std(y))

        # 2. R² prior with mean-precision parameterization
        r_squared = pm.Beta("r_squared", alpha=2, beta=2)  # Expect moderate fit

        # 3. Signal factor W (total standardized explained variance τ²)
        W = pm.Deterministic("W", r_squared / (1 - r_squared))  # W = τ²

        # 4. Signal strength allocation across all effect types
        # Components: population effects + group-specific effects
        n_components = n_predictors + n_grouping_factors
        phi = pm.Dirichlet("phi", a=np.full(n_components, 0.5), dims="components")

        # 5. Residual variance
        sigma = pm.HalfStudentT("sigma", nu=3, sigma=np.std(y))
        sigma_squared = pm.Deterministic("sigma_squared", sigma**2)

        # 6. Population-level effects (first p components)
        # Scale factor: σ²/σ²ₓᵢ for standardization
        population_scale = pm.Deterministic(
            "population_scale",
            pt.sqrt(sigma_squared * phi[:n_predictors] * W),
            dims="predictors"
        )
        beta = pm.Normal("beta", mu=0, sigma=population_scale, dims="predictors")

        # 7. Group-specific effects (remaining components)
        # Each grouping factor gets one variance component
        component_idx = n_predictors

        # Treatment effects (if treatment is a grouping factor)
        treatment_scale = pm.Deterministic(
            "treatment_scale",
            pt.sqrt(sigma_squared * phi[component_idx] * W)
        )
        treatment_effects = pm.Normal("treatment_effects", mu=0, sigma=treatment_scale, dims="treatment")
        component_idx += 1

        # Nuisance effects (if nuisance factors are grouping factors)
        nuisance_scale = pm.Deterministic(
            "nuisance_scale",
            pt.sqrt(sigma_squared * phi[component_idx] * W)
        )
        nuisance_effects = pm.Normal("nuisance_effects", mu=0, sigma=nuisance_scale, dims="nuisance_factor")
        component_idx += 1

        # Blocking effects (if blocking factors are grouping factors)
        blocking_scale = pm.Deterministic(
            "blocking_scale",
            pt.sqrt(sigma_squared * phi[component_idx] *x W)
        )
        blocking_effects = pm.Normal("blocking_effects", mu=0, sigma=blocking_scale, dims="block")

        # 8. Linear predictor
        eta = b0 + pm.math.dot(X, beta)

        # Add group-specific effects based on factor types
        if has_treatment_groups:
            eta += treatment_effects[treatment_idx]
        if has_nuisance_groups:
            eta += nuisance_effects[nuisance_idx]
        if has_blocking_groups:
            eta += blocking_effects[block_idx]

        # 9. Likelihood
        y = pm.Normal("y", mu=eta, sigma=sigma, observed=y_data, dims="obs")

        # 10. Derived quantities for interpretation
        # R² calculation using empirical variance of linear predictor (correct R2D2M2 approach)
        explained_variance = pm.Deterministic("explained_variance", pm.math.var(eta))
        total_variance = pm.Deterministic("total_variance", pm.math.var(eta) + sigma_squared)

        # Theoretical signal variance (W × σ²) for prior specification
        theoretical_signal_variance = pm.Deterministic("theoretical_signal_variance", W * sigma_squared)
    ```

    ### Factor Type Guidelines:
    - **TREATMENT**: Primary effects of interest (can be population-level or group-specific)
    - **NUISANCE**: Sources of variation to control for (plate, day, operator effects)
    - **BLOCKING**: Experimental design stratification (blocks, strata)
    - **COVARIATE**: Continuous variables (population-level fixed effects)

    **Signal Allocation Strategy:**
    - Population-level effects: Each predictor gets its own signal allocation
    - Group-specific effects: Each grouping factor type gets one shared signal allocation
    - All components compete for total signal strength W
    - Dirichlet concentration parameter controls sparsity (smaller = more sparse)

    ### Response Type Handling:
    - **Gaussian**: Use Normal likelihood with residual variance
    - **Poisson**: Use Poisson likelihood (no residual variance)
    - **Binomial**: Use Binomial likelihood with logit/probit link
    - **Other**: Use appropriate likelihood with proper link functions

    Key requirements:
    1. Generate complete, runnable PyMC code
    2. Use R2D2M2 framework with proper variance decomposition
    3. Handle different factor types with appropriate priors
    4. Generate appropriate PyMC coordinates and dimensions
    5. Include data loading and preprocessing code
    6. Add helpful comments explaining the R2D2 structure
    """  # noqa: E501


def create_pymc_generator_bot(model_name: str = "gpt-4o") -> lmb.StructuredBot:
    """Create a StructuredBot for generating PyMC code."""
    return lmb.StructuredBot(
        system_prompt=pymc_generation_system(),
        pydantic_model=PyMCModelResponse,
        model_name=model_name,
    )


@lmb.prompt(role="user")
def generate_pymc_model_prompt(experiment_json: str):
    """Please generate PyMC model code for the following experiment:

    Experiment Description (JSON):
    {{ experiment_json }}

    Please generate complete, runnable PyMC code that:
    1. Loads the data
    2. Sets up appropriate coordinates and dimensions
    3. Implements the R2D2 framework based on the experiment structure
    4. Handles all factor types correctly
    5. Includes proper variance component allocation
    6. Uses appropriate likelihood based on response type

    Return only the PEP 723 compliant Python code."""


def generate_pymc_model(
    experiment: ExperimentDescription,
    model_name: str = "gpt-4o",
) -> PyMCModelResponse:
    """Generate PyMC model code for an experiment description."""
    bot = create_pymc_generator_bot(model_name)

    # Convert experiment to JSON
    experiment_json = experiment.model_dump_json(indent=2)

    # Use the user prompt with the experiment JSON
    response = bot(generate_pymc_model_prompt(experiment_json))
    return response


# Default PyMC generator bot
pymc_bot = create_pymc_generator_bot()
