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
    description: str = Field(
        ...,
        description="Natural language description of what the model does and how it works",  # noqa: E501
    )
    sample_data_csv: str = Field(
        ...,
        description="CSV-formatted sample data table with realistic values for the experiment. Should include all experimental factors and response variables with sufficient rows to feel realistic for laboratory data collection.",  # noqa: E501
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

    IMPORTANT: When generating the description, use the actual experimental factors and treatment levels from the experiment.
    Do not use generic terms like "treatment levels" or "experimental factors" - use the specific names like "PBS", "Tris HCl", "mixing speed", etc.
    Provide concrete interpretation examples using the actual factor names from the experiment.

    ## CRITICAL DESIGN PRINCIPLES:
    1. **NO FOR-LOOPS**: Never use for-loops in model construction. Use explicit PyMC operations with proper dimensions.
    2. **EXPLICIT DIMENSIONS**: Use PyMC's `dims` parameter extensively for all experimental factors (treatments, nuisance factors, replicates, etc.).
    3. **VECTORIZED OPERATIONS**: Use broadcasting and vectorized operations to handle multiple experimental factors simultaneously.

    ## R2D2 Framework Overview
    R2D2 (R-squared Dirichlet Decomposition) is a Bayesian regularization framework for variance decomposition in probabilistic models. For laboratory experiments, use R2D2M2 (multilevel/hierarchical models).

    ### R2D2M2 Key Principles:
    1. **Signal Factor W**: Total signal-to-noise ratio (W = τ² = R²/(1-R²))
    2. **Variance Allocation**: Use Dirichlet distribution to allocate signal strength across components
    3. **R² Interpretation**: Each φ component represents its proportion of total explained variance
    4. **Hierarchical Structure**: Handle nested effects (e.g., treatment within blocks)

    ### R2D2M2 Implementation Pattern (NO FOR-LOOPS):

    **Example 1: Linear Response Model**
    ```python
    import pymc as pm
    import numpy as np
    import pytensor.tensor as pt

    with pm.Model(coords=coords) as model:
        # 1. Measurement precision (unexplained variance)
        sigma = pm.HalfNormal("sigma", sigma=1.0)

        # 2. Model fit quality (proportion of variation explained by experimental factors)
        r_squared = pm.Beta("r_squared", alpha=2, beta=2)  # Expect moderate fit

        # 3. Total experimental effect strength (signal-to-noise ratio)
        model_snr = pm.Deterministic("model_snr", r_squared / (1 - r_squared))

        # 4. Total explainable variance
        W = pm.Deterministic("W", model_snr * sigma**2)

        # 5. Variance proportions across components (treatment + nuisance factors)
        # Example: [treatment_1, treatment_2, day_effects, operator_effects]
        phi = pm.Dirichlet("phi", a=np.array([90, 5, 3, 2]))  # Adjust based on expected importance

        # 6. Individual variance components
        treatment_1_variance = pm.Deterministic("treatment_1_variance", phi[0] * W)
        treatment_2_variance = pm.Deterministic("treatment_2_variance", phi[1] * W)
        day_variance = pm.Deterministic("day_variance", phi[2] * W)
        operator_variance = pm.Deterministic("operator_variance", phi[3] * W)

        # 7. Individual treatment effects (explicit, no loops)
        treatment_1_effect = pm.Normal("treatment_1_effect", mu=0, sigma=np.sqrt(treatment_1_variance), dims="treatment_1")
        treatment_2_effect = pm.Normal("treatment_2_effect", mu=0, sigma=np.sqrt(treatment_2_variance), dims="treatment_2")

        # 8. Nuisance factor effects
        day_effects = pm.Normal("day_effects", mu=0, sigma=np.sqrt(day_variance), dims="day")
        operator_effects = pm.Normal("operator_effects", mu=0, sigma=np.sqrt(operator_variance), dims="operator")

        # 9. Predicted response using broadcasting (no loops)
        # Use indicator variables or design matrices for treatments
        predicted_response = (
            baseline +
            treatment_1_effect[treatment_1_idx] +
            treatment_2_effect[treatment_2_idx] +
            day_effects[day_idx] +
            operator_effects[operator_idx]
        )

        # 10. Observed data
        y = pm.Normal("y", mu=predicted_response, sigma=sigma, observed=y_data, dims="obs")
    ```

    **Example 2: Sigmoidal Response Model with Link Functions**
    ```python
    import pymc as pm
    import numpy as np
    import pytensor.tensor as pt

    with pm.Model(coords=coords) as model:
        # 1. Measurement precision (unexplained variance)
        sigma = pm.HalfNormal("sigma", sigma=1.0)

        # 2. Model fit quality (proportion of variation explained by experimental factors)
        r_squared = pm.Beta("r_squared", alpha=2, beta=2)  # Expect moderate fit

        # 3. Total experimental effect strength (signal-to-noise ratio)
        model_snr = pm.Deterministic("model_snr", r_squared / (1 - r_squared))

        # 4. Total explainable variance
        W = pm.Deterministic("W", model_snr * sigma**2)

        # 5. Variance proportions across components
        # [concentration_effect, treatment_effect, day_effects, operator_effects]
        phi = pm.Dirichlet("phi", a=np.array([70, 20, 7, 3]))  # Concentration effect dominates

        # 6. Individual variance components
        concentration_variance = pm.Deterministic("concentration_variance", phi[0] * W)
        treatment_variance = pm.Deterministic("treatment_variance", phi[1] * W)
        day_variance = pm.Deterministic("day_variance", phi[2] * W)
        operator_variance = pm.Deterministic("operator_variance", phi[3] * W)

        # 7. Sigmoidal curve parameters (in log space for positivity)
        max_response = pm.Normal("max_response", mu=100, sigma=10)  # Maximum response
        steepness = pm.HalfNormal("steepness", sigma=2)  # Steepness parameter (positive)
        ec50 = pm.Normal("ec50", mu=np.log(10), sigma=1, dims="treatment")  # Log EC50 per treatment

        # 8. Treatment effects on EC50 (shifts the curve)
        treatment_effect = pm.Normal("treatment_effect", mu=0, sigma=np.sqrt(treatment_variance), dims="treatment")

        # 9. Nuisance factor effects
        day_effects = pm.Normal("day_effects", mu=0, sigma=np.sqrt(day_variance), dims="day")
        operator_effects = pm.Normal("operator_effects", mu=0, sigma=np.sqrt(operator_variance), dims="operator")

        # 10. Linear predictor (logit space)
        linear_predictor = (
            ec50[treatment_idx] + treatment_effect[treatment_idx] +
            day_effects[day_idx] + operator_effects[operator_idx]
        )

        # 11. Sigmoidal response using logistic function
        # Transform concentration to log scale for better numerical properties
        log_concentration = pt.log(concentration_data)

        # Sigmoidal curve: response = max_response / (1 + exp(-steepness * (log_conc - ec50)))
        predicted_response = pm.Deterministic(
            "predicted_response",
            max_response / (1 + pt.exp(-steepness * (log_concentration - linear_predictor))),
            dims="obs"
        )

        # 12. Observed data (could be proportions, use logit transform if needed)
        y = pm.Normal("y", mu=predicted_response, sigma=sigma, observed=y_data, dims="obs")
    ```

    ### Factor Type Guidelines:
    - **TREATMENT**: The experimental conditions you want to compare (e.g., different drugs, concentrations, methods)
    - **NUISANCE**: Sources of experimental variation that aren't of primary interest (e.g., different days, operators, equipment)
    - **BLOCKING**: Experimental design factors that help control variation (e.g., batches, time blocks)
    - **COVARIATE**: Continuous measurements that might affect the outcome (e.g., temperature, pH, initial concentration)

    **Effect Allocation Strategy:**
    - Treatment effects: Each treatment condition gets its own effect estimate with explicit dimensions
    - Experimental variation: Each source of variation (day, operator, etc.) gets its own effect estimate
    - All effects compete for the total experimental effect strength via the Dirichlet allocation
    - The allocation controls how much each factor contributes to the overall variation

    ### Response Type Handling and Transformations:
    - **Gaussian**: Use Normal likelihood for continuous measurements (e.g., concentration, weight, absorbance)
    - **Poisson**: Use Poisson likelihood for count data (e.g., colony counts, cell counts)
    - **Proportions/Percentages**: Use logit transform to convert to unbounded space, then use Gaussian likelihood
      - Raw data comes as percentages (0-100%) or proportions (0-1)
      - Apply logit transform: logit(y) = log(y/(1-y)) for proportions or log(y/(100-y)) for percentages
      - Model in logit space with Gaussian likelihood (much easier mathematically)
      - Transform back to original scale for interpretation
    - **Other**: Use appropriate likelihood for other measurement types

    ### Nonlinear Relationship Handling:
    When collaborators specify sigmoidal/saturation relationships (e.g., concentration-response, time-response):
    - **Sigmoidal curves**: Use logistic function: y = L / (1 + exp(-k(x - x0)))
      - L = maximum response (asymptote)
      - k = steepness parameter
      - x0 = inflection point (EC50, half-maximal concentration/time)
    - **Saturation curves**: Use Michaelis-Menten: y = Vmax * x / (Km + x)
      - Vmax = maximum velocity/response
      - Km = concentration/time at half-maximal response
    - **Exponential decay**: Use y = A * exp(-k * x) + baseline
      - A = initial amplitude
      - k = decay rate
      - baseline = asymptotic value
    - Model parameters in log space when they must be positive
    - Use appropriate priors for biological parameters (e.g., EC50 in reasonable concentration range)

    ### Coordinates and Dimensions Setup:
    ```python
    coords = {
        "treatment_1": ["level_1", "level_2", "level_3"],
        "treatment_2": ["low", "medium", "high"],
        "day": ["day_1", "day_2", "day_3"],
        "operator": ["op_1", "op_2"],
        "replicate": ["rep_1", "rep_2", "rep_3"],
        "obs": np.arange(len(y_data))
    }
    ```

    ### Sample Data Generation Guidelines:
    Generate realistic sample data that feels appropriate for laboratory experiments:

    **Data Volume Guidelines:**
    - **Small experiments**: 20-50 observations (e.g., 3 treatments × 3 replicates × 3 days = 27 observations)
    - **Medium experiments**: 50-200 observations (e.g., 4 treatments × 4 replicates × 8 days = 128 observations)
    - **Large experiments**: 200+ observations (e.g., 6 treatments × 5 replicates × 10 days = 300 observations)

    **Realistic Value Ranges:**
    - **Concentration measurements**: 0.1-1000 μM, 0.01-100 mg/mL, etc.
    - **Absorbance/fluorescence**: 0.001-2.0 (log scale often appropriate)
    - **Count data**: 0-1000 colonies, 0-10000 cells, etc.
    - **Proportions**: 0.0-1.0 (success rates, survival rates)
    - **Time measurements**: 0-24 hours, 0-7 days, etc.
    - **Temperature**: 4-37°C (biological range)
    - **pH**: 6.0-8.5 (biological range)

    **Experimental Design Considerations:**
    - Include all treatment levels and nuisance factors
    - Add realistic variation (measurement noise, day-to-day variation)
    - Use meaningful factor names (e.g., "PBS", "Tris_HCl", "Day_1", "Operator_Alice")
    - Include replicate information when appropriate
    - Add any blocking factors or covariates

    **CSV Format Requirements:**
    - Use clear, descriptive column names
    - Include all experimental factors as separate columns
    - Include the response variable column
    - Use realistic data types (strings for factors, floats for measurements)
    - Add appropriate units in column names if helpful (e.g., "concentration_uM", "time_hours")

    Key requirements:
    1. Generate complete, runnable PyMC code
    2. Use R2D2M2 framework with explicit individual coefficients (NO FOR-LOOPS)
    3. Create separate coefficients for each treatment level with proper dimensions
    4. Handle different factor types with appropriate priors
    5. Generate comprehensive PyMC coordinates and use dims extensively
    6. Include data loading and preprocessing code
    7. Add helpful comments explaining the experimental structure
    8. Use accessible variable names that relate to the experiment
    9. Generate a description that names specific experimental factors and treatment levels
    10. Provide concrete interpretation examples using actual factor names
    11. Use vectorized operations and broadcasting instead of loops
    12. Make model structure explicit and readable
    13. Generate realistic sample data CSV with appropriate volume and value ranges
    14. Include all experimental factors and response variables in the sample data
    15. Use logit transform for proportions/percentages and model in Gaussian space
    16. Implement sigmoidal/saturation curves when specified by collaborators
    17. Use appropriate parameterizations and priors for nonlinear relationships
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

    Please generate:
    1. A natural language description of what the model does and how it works
    2. Complete, runnable PyMC code that:
       - Loads the data
       - Sets up appropriate coordinates and dimensions
       - Implements the R2D2 framework based on the experiment structure
       - Handles all factor types correctly
       - Includes proper variance component allocation
       - Uses appropriate likelihood based on response type
    3. A realistic sample data table in CSV format that:
       - Includes all experimental factors and treatment levels
       - Contains the response variable with realistic values
       - Has sufficient rows to feel appropriate for laboratory data collection
       - Uses meaningful factor names and realistic value ranges

        The description should be specific and actionable:
    - Name the actual experimental factors and treatment levels from the experiment
    - Explain what each individual coefficient represents (e.g., "the effect of PBS compared to Tris HCl")
    - Provide concrete interpretation examples (e.g., "if the PBS coefficient is positive, PBS increases encapsulation efficiency")
    - Give prescriptive guidance on what the results mean for experimental decisions
    - Use the actual response variable name and units when available

    Make the description specific to this experiment, not generic. Include actual factor names and treatment levels.
    Focus on practical interpretation that helps the experimenter make decisions.
    """  # noqa: E501


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
