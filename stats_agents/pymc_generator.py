"""PyMC code generation bot using llamabot StructuredBot."""

from typing import Optional

import llamabot as lmb
from pydantic import BaseModel, Field

from .schemas import ExperimentDescription


class PyMCModelCode(BaseModel):
    """Response model for PyMC code generation only."""

    model_code: str = Field(
        ..., description="Complete PEP 723 compliant PyMC model code"
    )
    python_version: str = Field(
        default="3.12", description="Minimum Python version required"
    )
    dependencies: list[str] = Field(
        default=["pymc", "pandas", "memo"], description="List of required dependencies"
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

    def generate_sample_data(self, n_samples: int = 10, experiment=None) -> str:
        """
        Generate sample data by running prior predictive sampling on the model.

        Args:
            n_samples: Number of prior predictive samples to generate
            experiment: Optional ExperimentDescription for fallback data generation

        Returns:
            CSV string containing the sample data
        """
        from types import ModuleType

        # Create a new module to execute the code in
        module_name = f"pymc_model_{id(self)}"
        module = ModuleType(module_name)

        # Add necessary imports to the module's globals
        module.__dict__.update(
            {
                "pymc": __import__("pymc"),
                "pandas": __import__("pandas"),
                "numpy": __import__("numpy"),
                "pytensor": __import__("pytensor"),
                "memo": __import__("memo"),
            }
        )

        # Execute the model code to create the model object
        try:
            exec(self.model_code, module.__dict__)

            # Get the model object from the module
            if "model" not in module.__dict__:
                raise RuntimeError("Model code did not create a 'model' object")

            model = module.__dict__["model"]
            data = module.__dict__.get("data")

            # Run prior predictive sampling
            with model:
                idata = model.sample_prior_predictive(samples=n_samples)

            # Extract mean predictions
            y_pred = idata.prior_predictive.mean(dims=("chain", "draw"))

            # Create sample data with predicted values
            if data is not None:
                sample_data = data.copy()
                # Replace the response variable with the predicted values
                if hasattr(y_pred, "values"):
                    sample_data.iloc[:, -1] = y_pred.values.flatten()
                else:
                    sample_data.iloc[:, -1] = y_pred.flatten()
            else:
                # Fallback: create minimal data structure
                sample_data = module.__dict__.get("data", None)
                if sample_data is None:
                    # Create a simple fallback DataFrame
                    import numpy as np
                    import pandas as pd

                    sample_data = pd.DataFrame(
                        {
                            "treatment": ["control", "treatment_A"] * 3,
                            "response": np.random.normal(100, 20, 6),
                        }
                    )

            # Convert to CSV
            csv_output = sample_data.to_csv(index=False)
            return csv_output

        except Exception:
            # If execution fails, create fallback data based on the experiment structure
            import numpy as np
            import pandas as pd

            if experiment is not None:
                # Use experiment description to create better fallback data
                treatments = experiment.get_treatments()
                nuisance_factors = experiment.get_nuisance_factors()

                # Create data structure using memo.grid if possible
                try:
                    from memo import grid

                    # Create all combinations of factors
                    factor_dict = {}
                    for factor_name, levels in treatments.items():
                        factor_dict[factor_name] = levels
                    for factor_name, levels in nuisance_factors.items():
                        factor_dict[factor_name] = levels

                    if factor_dict:
                        settings = grid(**factor_dict, shuffle=False)
                        fallback_data = pd.DataFrame(settings)
                        # Add response variable
                        fallback_data[experiment.response] = np.random.normal(
                            100, 20, len(fallback_data)
                        )
                    else:
                        # No factors found, create minimal data
                        fallback_data = pd.DataFrame(
                            {
                                "treatment": ["control", "treatment_A"] * 3,
                                experiment.response: np.random.normal(100, 20, 6),
                            }
                        )
                except ImportError:
                    # Fallback without memo.grid
                    data_dict = {}
                    for factor_name, levels in treatments.items():
                        data_dict[factor_name] = levels * 3
                    for factor_name, levels in nuisance_factors.items():
                        data_dict[factor_name] = levels * 3

                    n_rows = len(list(data_dict.values())[0]) if data_dict else 6
                    data_dict[experiment.response] = np.random.normal(100, 20, n_rows)
                    fallback_data = pd.DataFrame(data_dict)
            else:
                # Create minimal fallback data without experiment info
                fallback_data = pd.DataFrame(
                    {
                        "treatment": ["control", "treatment_A"] * 3,
                        "plate": ["plate_1", "plate_2"] * 3,
                        "response": np.random.normal(100, 20, 6),
                    }
                )

            return fallback_data.to_csv(index=False)


class PyMCModelDescription(BaseModel):
    """Response model for PyMC model description generation."""

    description: str = Field(
        ...,
        description="Natural language description of what the model does and how it works",  # noqa: E501
    )


class PyMCSampleData(BaseModel):
    """Response model for sample data generation."""

    sample_data_csv: str = Field(
        ...,
        description="CSV-formatted sample data table with realistic values for the experiment. Should include all experimental factors and response variables with sufficient rows to feel realistic for laboratory data collection.",  # noqa: E501
    )


# Legacy class for backward compatibility
class PyMCModelResponse(BaseModel):
    """Legacy response model for PyMC code generation (deprecated)."""

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
        default=["pymc", "pandas", "memo"], description="List of required dependencies"
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

    **Example 1: Linear Response Model with Replicates**
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
        # Note: No replicate variance needed - replicates don't have systematic effects
        phi = pm.Dirichlet("phi", a=np.array([70, 20, 5, 5]), dims="variance_components")  # Allocate among meaningful effects only

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

        # 9. Global intercept (always include)
        mu = pm.Normal("mu", mu=0, sigma=10)

        # 10. Predicted response using broadcasting (no loops)
        # Note: No replicate effect needed - replicates are just multiple observations
        # of the same experimental conditions
        predicted_response = (
            mu +  # Global intercept
            treatment_1_effect[treatment_1_idx] +
            treatment_2_effect[treatment_2_idx] +
            day_effects[day_idx] +
            operator_effects[operator_idx]
        )

        # 11. Observed data
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
    - **REPLICATE**: Technical or biological replicates (e.g., multiple measurements of the same sample, multiple samples per condition)

    **CRITICAL: Replicate Handling - Two Cases:**

    **Case 1: Technical Replicates (most common)**
    - Multiple measurements of the same experimental unit/condition
    - **Do NOT add replicate effects** - they should have identical expected values
    - **Do NOT include replicate variance** in Dirichlet allocation
    - Measurement error (sigma) captures replicate-to-replicate variation
    - Example: "3 technical replicates per plate" = 3 measurements from each well

    **Case 2: Replicates with Systematic Differences**
    - When replicates have their own systematic variation (e.g., different starting values, spatial effects)
    - **DO include replicate effects** with proper nested dimensions
    - **DO include replicate variance** in Dirichlet allocation
    - Example phrases: "each replicate has its own starting value", "spatial variation across replicates", "replicate-specific baseline"

    **How to distinguish:**
    - Look for phrases indicating systematic replicate differences
    - If description mentions replicate-specific properties → Case 2
    - If just "technical replicates" or "multiple measurements" → Case 1

    **Effect Allocation Strategy:**
    - Treatment effects: Each treatment condition gets its own effect estimate with explicit dimensions
    - Experimental variation: Each source of variation (day, operator, etc.) gets its own effect estimate
    - **Replicates: NO separate effects** - they are just multiple observations of the same conditions
    - The Dirichlet allocation partitions the explained variance among different sources of variation
    - Each component represents the proportion of total explainable variance attributed to that factor
    - **IMPORTANT**: Do NOT include replicates in the variance allocation - only meaningful sources of variation

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
        "variance_components": ["treatment_1", "treatment_2", "day", "operator"],  # Components in phi
        "obs": np.arange(len(y_data))
    }
    ```

    **CRITICAL: Always include variance_components coordinate**
    - The `phi` parameter MUST have `dims="variance_components"`
    - The coordinate should list the effects that variance is being decomposed among
    - This makes the model interpretable and ensures proper indexing

    ### Replicate Structure Handling:
    When replicates are present in the experiment:
    1. **Include replicate dimensions** in PyMC coordinates for data structure
    2. **Do NOT create replicate effects** - replicates should have identical expected values
    3. **Use proper indexing** for treatments and nuisance factors only
    4. **Measurement error (sigma) captures** replicate-to-replicate variation

    **Example with nested replicates (correct approach):**
    ```python
    # If replicates are nested within plates
    coords = {
        "treatment": ["control", "treatment_A", "treatment_B"],
        "plate": ["plate_1", "plate_2", "plate_3"],
        "replicate": ["rep_1", "rep_2", "rep_3"],  # For data structure only
        "obs": np.arange(len(y_data))
    }

    with pm.Model(coords=coords) as model:
        # Treatment effects
        treatment_effect = pm.Normal("treatment_effect", mu=0, sigma=treatment_sd, dims="treatment")

        # Plate-level effects (nuisance factor)
        plate_effect = pm.Normal("plate_effect", mu=0, sigma=plate_sd, dims="plate")

        # NO replicate effects - replicates are just multiple observations

        # Linear predictor - same expected value for all replicates within a condition
        mu_obs = (
            baseline +
            treatment_effect[treatment_idx] +
            plate_effect[plate_idx]
            # No replicate indexing needed - all replicates have same expected value
        )
    ```

    **Key points for nested replicates:**
    1. Include "replicate" in coordinates for data structure clarity
    2. Do NOT create replicate effects - they should have the same expected value
    3. Do NOT include replicate variance in the Dirichlet allocation
    4. Measurement error (sigma) captures replicate-to-replicate variation naturally

    ### MANDATORY Model Generation Workflow:
    Before generating ANY model, you MUST:

    **STEP 1: Parse the experiment description JSON for replicate_structure**
    ```python
    # Check if experiment_description.replicate_structure exists
    # If it exists, extract:
    # - replicate_type (technical/biological)
    # - replicates_per_unit (number)
    # - nested_under (what unit they're nested in)
    ```

    **STEP 2: Set up variance decomposition**
    - ALWAYS create "variance_components" coordinate listing the effects being decomposed
    - Set phi parameter with dims="variance_components"
    - Determine if replicates need their own variance component (Case 1 vs Case 2)
    - If Case 2: Add replicate variance component to Dirichlet allocation
    - If Case 1: Do NOT include replicate variance

    **STEP 3: Generate proper indexing**
    ```python
    # Always use .values for indexing
    treatment_idx = data['treatment'].astype('category').cat.codes
    plate_idx = data['plate'].astype('category').cat.codes  # unit for nesting
    replicate_idx = data['replicate'].astype('category').cat.codes
    ```

    **STEP 4: Include global intercept**
    ```python
    # Always include baseline/intercept
    mu = pm.Normal("mu", mu=0, sigma=10)
    ```

    **TEMPLATE Case 1: Technical Replicates (simple multiple measurements):**
    ```python
    # Example: "3 technical replicates per plate"
    coords = {
        "treatment": ["control", "treatment_A", "treatment_B"],
        "plate": ["plate_1", "plate_2", "plate_3"],
        "cell_line": ["line_1", "line_2", "line_3", "line_4"],
        "replicate": ["rep_1", "rep_2", "rep_3"],  # For data structure only
        "variance_components": ["treatment", "cell_line", "plate", "other"],  # Components in phi
        "obs": np.arange(len(data))
    }

    # Dirichlet allocation for meaningful effects only
    phi = pm.Dirichlet("phi", a=np.array([40, 30, 20, 10]), dims="variance_components")

    # NO replicate effects - replicates have same expected value
    predicted_response = (
        mu +
        treatment_effect[treatment_idx] +
        cell_line_effect[cell_line_idx] +
        plate_effect[plate_idx]
        # No replicate terms
    )
    ```

    **TEMPLATE Case 2: Replicates with Systematic Differences:**
    ```python
    # Example: "each replicate has its own starting value" or "spatial variation across replicates"
    coords = {
        "treatment": ["control", "treatment_A", "treatment_B"],
        "plate": ["plate_1", "plate_2", "plate_3"],
        "replicate": ["rep_1", "rep_2", "rep_3"],
        "variance_components": ["treatment", "plate", "replicate", "other"],  # Components in phi
        "obs": np.arange(len(data))
    }

    # Include replicate variance in Dirichlet allocation
    phi = pm.Dirichlet("phi", a=np.array([40, 30, 20, 10]), dims="variance_components")
    replicate_variance = pm.Deterministic("replicate_variance", phi[2] * W)

    # Include replicate effects with nested dimensions
    replicate_effect = pm.Normal("replicate_effect", mu=0, sigma=pt.sqrt(replicate_variance), dims=("plate", "replicate"))

    predicted_response = (
        mu +
        treatment_effect[treatment_idx] +
        plate_effect[plate_idx] +
        replicate_effect[plate_idx, replicate_idx]  # Include replicate effects
    )
    ```

    ### Model Generation Checklist:
    Before generating the model, check the experiment description for:
    1. **Replicate structure**: Look for `replicate_structure` field in the experiment description
    2. **Replicate type**: Determine if replicates are technical or biological
    3. **Nesting structure**: Identify what the replicates are nested under (e.g., plate, mouse, treatment)
    4. **Number of replicates**: Get the number of replicates per experimental unit
    5. **Factor types**: Identify all treatment, nuisance, and replicate factors

    **If replicates are present:**
    - Include replicate dimensions in PyMC coordinates
    - Add replicate variance to the Dirichlet allocation
    - Use proper nested indexing: `replicate_effect[unit_idx, replicate_idx]`
    - Create appropriate indexing variables for both unit and replicate dimensions

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
    18. **ALWAYS include "variance_components" coordinate that lists the effects in the phi parameter**
    19. **The phi parameter MUST have dims="variance_components" for interpretability**
    20. **Distinguish between technical replicates (Case 1) and replicates with systematic differences (Case 2)**
    21. **Case 1 (technical replicates): Do NOT include replicate effects or variance in Dirichlet allocation**
    22. **Case 2 (systematic differences): DO include replicate effects with nested dims and variance in Dirichlet allocation**
    23. **Look for phrases like "each replicate has its own starting value" to identify Case 2**
    """  # noqa: E501


@lmb.prompt(role="system")
def pymc_code_generation_system():
    """You are an expert in Bayesian modeling with PyMC and the R2D2 framework.

    Your task is to generate PEP 723 compliant PyMC model code based on a structured experiment description.

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

    **Example 1: Linear Response Model with Replicates**
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
        # Note: No replicate variance needed - replicates don't have systematic effects
        phi = pm.Dirichlet("phi", a=np.array([70, 20, 5, 5]), dims="variance_components")  # Allocate among meaningful effects only

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

        # 9. Global intercept (always include)
        mu = pm.Normal("mu", mu=0, sigma=10)

        # 10. Predicted response using broadcasting (no loops)
        # Note: No replicate effect needed - replicates are just multiple observations
        # of the same experimental conditions
        predicted_response = (
            mu +  # Global intercept
            treatment_1_effect[treatment_1_idx] +
            treatment_2_effect[treatment_2_idx] +
            day_effects[day_idx] +
            operator_effects[operator_idx]
        )

        # 11. Observed data
        y = pm.Normal("y", mu=predicted_response, sigma=sigma, observed=y_data, dims="obs")
    ```

    ### Factor Type Guidelines:
    - **TREATMENT**: The experimental conditions you want to compare (e.g., different drugs, concentrations, methods)
    - **NUISANCE**: Sources of experimental variation that aren't of primary interest (e.g., different days, operators, equipment)
    - **BLOCKING**: Experimental design factors that help control variation (e.g., batches, time blocks)
    - **COVARIATE**: Continuous measurements that might affect the outcome (e.g., temperature, pH, initial concentration)
    - **REPLICATE**: Technical or biological replicates (e.g., multiple measurements of the same sample, multiple samples per condition)

    **CRITICAL: Replicate Handling - Two Cases:**

    **Case 1: Technical Replicates (most common)**
    - Multiple measurements of the same experimental unit/condition
    - **Do NOT add replicate effects** - they should have identical expected values
    - **Do NOT include replicate variance** in Dirichlet allocation
    - Measurement error (sigma) captures replicate-to-replicate variation
    - Example: "3 technical replicates per plate" = 3 measurements from each well

    **Case 2: Replicates with Systematic Differences**
    - When replicates have their own systematic variation (e.g., different starting values, spatial effects)
    - **DO include replicate effects** with proper nested dimensions
    - **DO include replicate variance** in Dirichlet allocation
    - Example phrases: "each replicate has its own starting value", "spatial variation across replicates", "replicate-specific baseline"

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
        "variance_components": ["treatment_1", "treatment_2", "day", "operator"],  # Components in phi
        "obs": np.arange(len(y_data))
    }
    ```

    **CRITICAL: Always include variance_components coordinate**
    - The `phi` parameter MUST have `dims="variance_components"`
    - The coordinate should list the effects that variance is being decomposed among
    - This makes the model interpretable and ensures proper indexing

    ### MANDATORY Model Generation Workflow:
    Before generating ANY model, you MUST:

    **STEP 1: Parse the experiment description JSON for replicate_structure**
    ```python
    # Check if experiment_description.replicate_structure exists
    # If it exists, extract:
    # - replicate_type (technical/biological)
    # - replicates_per_unit (number)
    # - nested_under (what unit they're nested in)
    ```

    **STEP 2: Set up variance decomposition**
    - ALWAYS create "variance_components" coordinate listing the effects being decomposed
    - Set phi parameter with dims="variance_components"
    - Determine if replicates need their own variance component (Case 1 vs Case 2)
    - If Case 2: Add replicate variance component to Dirichlet allocation
    - If Case 1: Do NOT include replicate variance

    **STEP 3: Generate proper indexing**
    ```python
    # Always use .values for indexing
    treatment_idx = data['treatment'].astype('category').cat.codes
    plate_idx = data['plate'].astype('category').cat.codes  # unit for nesting
    replicate_idx = data['replicate'].astype('category').cat.codes
    ```

    **STEP 4: Include global intercept**
    ```python
    # Always include baseline/intercept
    mu = pm.Normal("mu", mu=0, sigma=10)
    ```

        **STEP 5: Safe CSV Data Loading**
    Always include safe CSV handling with fallback data generation:
    ```python
    import pandas as pd
    import numpy as np
    from pathlib import Path
    from memo import grid

    # Try to load the CSV file, with fallback data generation
    try:
        data = pd.read_csv("experiment_data.csv")
        print("✓ Loaded data from CSV file")
    except FileNotFoundError:
        print("⚠ CSV file not found, generating sample data...")
        # Generate sample data based on the experiment structure
        # This ensures the model can run even without the CSV file
        treatments = ["control", "treatment_A", "treatment_B"]
        plates = ["plate_1", "plate_2", "plate_3"]
        replicates = ["rep_1", "rep_2", "rep_3"]

        # Create all combinations using memo.grid
        settings = grid(
            treatment=treatments,
            plate=plates,
            replicate=replicates,
            shuffle=False
        )

        # Convert to DataFrame and add response variable
        data = pd.DataFrame(settings)
        data["response"] = np.random.normal(100, 20, len(data))  # Sample response values

        print("✓ Generated sample data using memo.grid")
    except Exception as e:
        print(f"⚠ Error loading CSV: {e}, generating sample data...")
        # Additional fallback with minimal data
        data = pd.DataFrame({
            "treatment": ["control", "treatment_A"] * 3,
            "plate": ["plate_1", "plate_2"] * 3,
            "response": np.random.normal(100, 20, 6)
        })
        print("✓ Generated minimal fallback data")
    ```

    Key requirements:
    1. Generate complete, runnable PyMC code
    2. Use R2D2M2 framework with explicit individual coefficients (NO FOR-LOOPS)
    3. Create separate coefficients for each treatment level with proper dimensions
    4. Handle different factor types with appropriate priors
    5. Generate comprehensive PyMC coordinates and use dims extensively
    6. Include data loading and preprocessing code with safe CSV handling
    7. Add helpful comments explaining the experimental structure
    8. Use accessible variable names that relate to the experiment
    9. Use vectorized operations and broadcasting instead of loops
    10. Make model structure explicit and readable
    11. Use logit transform for proportions/percentages and model in Gaussian space
    12. Implement sigmoidal/saturation curves when specified by collaborators
    13. Use appropriate parameterizations and priors for nonlinear relationships
    14. **ALWAYS include "variance_components" coordinate that lists the effects in the phi parameter**
    15. **The phi parameter MUST have dims="variance_components" for interpretability**
    16. **Distinguish between technical replicates (Case 1) and replicates with systematic differences (Case 2)**
    17. **Case 1 (technical replicates): Do NOT include replicate effects or variance in Dirichlet allocation**
    18. **Case 2 (systematic differences): DO include replicate effects with nested dims and variance in Dirichlet allocation**
    19. **Look for phrases like "each replicate has its own starting value" to identify Case 2**
    20. **SAFE CSV HANDLING: Always include try-except blocks when loading CSV files and provide fallback data generation**
    21. **USE MEMO.GRID: Always use `from memo import grid` and the `grid()` function to generate all combinations of experimental factors in fallback data generation**
    """  # noqa: E501


@lmb.prompt(role="system")
def pymc_description_generation_system():
    """You are an expert in Bayesian modeling with PyMC and the R2D2 framework.

    Your task is to generate natural language descriptions of PyMC models based on experiment descriptions and model code.

    ## CRITICAL CONTEXT:
    This description is to be read **BEFORE model fitting** by intensely curious and critically-thinking laboratory and data scientists. Write in **prospective tense** and frame parameter explanations as **prior beliefs** that will be updated by the data.

    ## Description Guidelines:
    1. **Use specific experimental factor names**: Use the actual experimental factors and treatment levels from the experiment (e.g., "PBS", "Tris HCl", "mixing speed", etc.)
    2. **Avoid generic terms**: Do not use generic terms like "treatment levels" or "experimental factors" - use the specific names
    3. **Provide concrete interpretation examples**: Use actual factor names from the experiment
    4. **Focus on practical interpretation**: Help experimenters make decisions based on results
    5. **Explain individual coefficients**: What each coefficient represents in the context of the experiment
    6. **Use actual response variable names**: Use the specific response variable name and units when available
    7. **Write prospectively**: Use future tense and conditional language since this is pre-fitting
    8. **Frame parameters as prior beliefs**: Explain model parameters as initial assumptions that will be updated by data

    ## Description Structure:
    1. **Model Overview**: What the model will do and how it will work
    2. **Factor Interpretation**: What each experimental factor represents
    3. **Coefficient Meaning**: What individual coefficients will mean in the experiment context
    4. **Practical Examples**: Concrete examples using actual factor names
    5. **Decision Guidance**: How to interpret results for experimental decisions (CRITICAL SECTION)
    6. **Prior Beliefs and Variance Decomposition**: Initial assumptions about variance allocation that will be updated by data

    ## Decision Guidance Section (CRITICAL):
    This section should provide clear, actionable guidance for experimental decision-making:
    - **Treatment Effects**: How to interpret positive/negative coefficients for each treatment
    - **Effect Sizes**: What magnitude of effects would be practically meaningful
    - **Statistical vs Practical Significance**: How to distinguish between statistical significance and practical importance
    - **Next Steps**: What experimental decisions to make based on different outcomes
    - **Risk Assessment**: What to watch out for or be cautious about
    - **Optimization Opportunities**: How to use results to improve future experiments

    ## Prior Beliefs Framing:
    When explaining model parameters, frame them as initial assumptions that will be updated:
    - **Instead of**: "The Dirichlet distribution is used to model the proportions of variance explained by each component, with the treatment expected to explain the largest portion (70%)"
    - **Use**: "Our prior belief is that treatment effects will explain the largest portion of variance (70%), followed by cell line effects (20%) and plate effects (10%). These beliefs will be updated based on the actual data."
    - **Instead of**: "The model assumes..."
    - **Use**: "We start with the prior belief that..." or "Our initial assumption is that..."

    ## Key Requirements:
    - Name the actual experimental factors and treatment levels from the experiment
    - Explain what each individual coefficient will represent (e.g., "the effect of PBS compared to Tris HCl")
    - Provide concrete interpretation examples (e.g., "if the PBS coefficient is positive, PBS increases encapsulation efficiency")
    - Give detailed, actionable decision guidance for experimental decisions
    - Use the actual response variable name and units when available
    - Make the description specific to this experiment, not generic
    - Focus on practical interpretation that helps the experimenter make decisions
    - Write in prospective tense since this is pre-fitting
    - Frame all parameter explanations as prior beliefs that will be updated by data
    - Emphasize the decision guidance section as the most important part
    """  # noqa: E501


@lmb.prompt(role="system")
def pymc_sample_data_generation_system():
    """You are an expert in laboratory experimental design and data generation.

    Your task is to generate realistic sample data tables in CSV format based on experiment descriptions and PyMC model code.

    ## Sample Data Generation Guidelines:
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

    **Key Requirements:**
    1. Generate realistic sample data CSV with appropriate volume and value ranges
    2. Include all experimental factors and response variables in the sample data
    3. Use meaningful factor names and realistic value ranges
    4. Add appropriate experimental variation and noise
    5. Ensure data structure matches the PyMC model expectations
    6. Include all treatment levels, nuisance factors, and replicates
    7. Use realistic biological/chemical value ranges
    8. Add appropriate measurement noise and day-to-day variation
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


@lmb.prompt(role="user")
def generate_pymc_code_prompt(experiment_json: str):
    """Please generate PyMC model code for the following experiment:

    Experiment Description (JSON):
    {{ experiment_json }}

    Please generate complete, runnable PyMC code that:
    - Loads the data
    - Sets up appropriate coordinates and dimensions
    - Implements the R2D2 framework based on the experiment structure
    - Handles all factor types correctly
    - Includes proper variance component allocation
    - Uses appropriate likelihood based on response type
    - Includes data loading and preprocessing code
    - Adds helpful comments explaining the experimental structure
    - Uses accessible variable names that relate to the experiment
    - Uses vectorized operations and broadcasting instead of loops
    - Makes model structure explicit and readable
    """  # noqa: E501


@lmb.prompt(role="user")
def generate_pymc_description_prompt(experiment_json: str, model_code: str):
    """Please generate a natural language description of the PyMC model for the following experiment:

    Experiment Description (JSON):
    {{ experiment_json }}

    Model Code:
    {{ model_code }}

    Please generate a description that:
    - Names the actual experimental factors and treatment levels from the experiment
    - Explains what each individual coefficient represents (e.g., "the effect of PBS compared to Tris HCl")
    - Provides concrete interpretation examples (e.g., "if the PBS coefficient is positive, PBS increases encapsulation efficiency")
    - Gives prescriptive guidance on what the results mean for experimental decisions
    - Uses the actual response variable name and units when available
    - Makes the description specific to this experiment, not generic
    - Focuses on practical interpretation that helps the experimenter make decisions
    """  # noqa: E501


@lmb.prompt(role="user")
def generate_pymc_sample_data_prompt(experiment_json: str, model_code: str):
    """Please generate realistic sample data for the following experiment:

    Experiment Description (JSON):
    {{ experiment_json }}

    Model Code:
    {{ model_code }}

    Please generate a realistic sample data table in CSV format that:
    - Includes all experimental factors and treatment levels
    - Contains the response variable with realistic values
    - Has sufficient rows to feel appropriate for laboratory data collection
    - Uses meaningful factor names and realistic value ranges
    - Includes all treatment levels, nuisance factors, and replicates
    - Uses realistic biological/chemical value ranges
    - Adds appropriate measurement noise and day-to-day variation
    - Ensures data structure matches the PyMC model expectations
    """  # noqa: E501


# PyMC code generation bot
pymc_code_bot = lmb.StructuredBot(
    system_prompt=pymc_code_generation_system(),
    pydantic_model=PyMCModelCode,
    model_name="gpt-4o",
)

# PyMC description generation bot
pymc_description_bot = lmb.StructuredBot(
    system_prompt=pymc_description_generation_system(),
    pydantic_model=PyMCModelDescription,
    model_name="gpt-4o",
)

# PyMC sample data generation bot
pymc_sample_data_bot = lmb.StructuredBot(
    system_prompt=pymc_sample_data_generation_system(),
    pydantic_model=PyMCSampleData,
    model_name="gpt-4o",
)


def generate_pymc_code(
    experiment: ExperimentDescription,
    model_name: str = "gpt-4o",
) -> PyMCModelCode:
    """Generate PyMC model code for an experiment description."""
    experiment_json = experiment.model_dump_json(indent=2)
    response = pymc_code_bot(generate_pymc_code_prompt(experiment_json))
    return response


def generate_pymc_description(
    experiment: ExperimentDescription,
    model_code: PyMCModelCode,
    model_name: str = "gpt-4o",
) -> PyMCModelDescription:
    """Generate PyMC model description for an experiment and model code."""
    experiment_json = experiment.model_dump_json(indent=2)
    response = pymc_description_bot(
        generate_pymc_description_prompt(experiment_json, model_code.model_code)
    )
    return response


def generate_pymc_sample_data(
    experiment: ExperimentDescription,
    model_code: PyMCModelCode,
    model_name: str = "gpt-4o",
) -> PyMCSampleData:
    """Generate sample data by running prior predictive sampling on the model."""
    # Use the new method that runs the actual PyMC code
    try:
        csv_data = model_code.generate_sample_data(n_samples=10, experiment=experiment)
        return PyMCSampleData(sample_data_csv=csv_data)
    except Exception:
        # If the prior predictive sampling fails, create a simple fallback
        import numpy as np
        import pandas as pd

        # Create simple example data based on the experiment structure
        # This is a fallback when the model code can't be executed
        treatments = experiment.get_treatments()
        nuisance_factors = experiment.get_nuisance_factors()

        # Create a simple data structure
        data_dict = {}
        for factor_name, levels in treatments.items():
            data_dict[factor_name] = levels * 10  # Repeat each level 10 times

        for factor_name, levels in nuisance_factors.items():
            data_dict[factor_name] = levels * 10  # Repeat each level 10 times

        # Add response variable
        np.random.seed(42)
        n_rows = len(list(data_dict.values())[0]) if data_dict else 30
        data_dict[experiment.response] = np.random.normal(100, 20, n_rows)

        # Create DataFrame and convert to CSV
        df = pd.DataFrame(data_dict)
        csv_data = df.to_csv(index=False)

        return PyMCSampleData(sample_data_csv=csv_data)


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
