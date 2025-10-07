# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "anthropic==0.65.0",
#     "llamabot[all]==0.13.6",
#     "marimo",
#     "numpy==2.3.2",
#     "pandas==2.3.2",
#     "pymc==5.25.1",
# ]
# ///

import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell
def setup_1():
    import json

    return (json,)


@app.cell
def _():
    import numpy as np
    import pandas as pd

    # Set random seed for reproducibility
    np.random.seed(42)

    # Define experimental conditions
    molecules = [
        "Molecule A",
        "Molecule B",
        "Molecule C",
        "Molecule D",
        "Molecule E",
    ]
    concentrations = [0, 0.1, 1, 10, 100]  # µM, including control (0)

    # Create experimental data with some biological variability
    def generate_protein_expression_data():
        data = []
        for molecule in molecules:
            for conc in concentrations:
                # Simulate baseline with some variation
                base_expression = 100  # baseline protein expression

                # Add some concentration-dependent effect with noise
                if conc == 0:
                    # Control condition
                    effect = np.random.normal(0, 5, 3)
                else:
                    # Different molecules might have different effects
                    if molecule == "Molecule A":
                        # Slight increase
                        effect = base_expression * (
                            1 + 0.1 * np.log(conc)
                        ) + np.random.normal(0, 10, 3)
                    elif molecule == "Molecule B":
                        # Decrease
                        effect = base_expression * (
                            1 - 0.05 * np.log(conc)
                        ) + np.random.normal(0, 10, 3)
                    elif molecule == "Molecule C":
                        # Biphasic response
                        effect = base_expression * (
                            1 + 0.2 * np.log(conc) - 0.001 * conc
                        ) + np.random.normal(0, 10, 3)
                    elif molecule == "Molecule D":
                        # Minimal effect
                        effect = base_expression + np.random.normal(0, 10, 3)
                    else:  # Molecule E
                        # Strong increase
                        effect = base_expression * (
                            1 + 0.3 * np.log(conc)
                        ) + np.random.normal(0, 10, 3)

                # Create replicate data points
                for rep_effect in effect:
                    data.append(
                        {
                            "Molecule": molecule,
                            "Concentration": conc,
                            "Actin_Expression": max(
                                0, rep_effect
                            ),  # Ensure non-negative
                        }
                    )

        return pd.DataFrame(data)

    # Generate the dataset
    experimental_data = generate_protein_expression_data()
    experimental_data
    return experimental_data, pd


@app.cell
def _():
    from llamabot.bot.toolbot import ToolBot
    from llamabot.components.tools import write_and_execute_code

    bot = ToolBot(
        system_prompt="You are an expert at creating PyMC models.",
        model_name="gpt-4.1",
        tools=[write_and_execute_code(globals())],
    )
    return ToolBot, bot, write_and_execute_code


@app.cell
def _(experimental_data):
    def describe_dataframe(df):
        """
        Returns a dictionary with the dataframe schema and first 5 rows.

        Args:
            df (pd.DataFrame): Input dataframe

        Returns:
            dict: A dictionary containing schema and sample data
        """
        schema = {col: str(df[col].dtype) for col in df.columns}
        sample_data = df.head(5)

        return {"schema": schema, "sample_data": sample_data}

    describe_dataframe(experimental_data)
    return (describe_dataframe,)


@app.function
def model_user_prompt(schema: str) -> str:
    return f"""Fit a simple PyMC model to `experimental_data`. It has the following schema: {schema}.

    You should always fit the model with pm.sample(random_seed=42) and no other arguments unless instructed.
    You should always use the R2D2 framework for variance decomposition so that we can see how much noise is intrinsic v.s. explained by experimental factors.
    Always include the variance decomposition in pm.summary() at the end, as we want to know the proportion of variance that is explained by each of the experimental factors.

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
        y = pm.Normal("y", mu=predicted_response, sigma=sigma, observed=, dims="obs")
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
    """


@app.cell
def _(describe_dataframe, experimental_data):
    str(describe_dataframe(experimental_data))
    return


@app.cell
def _(describe_dataframe, experimental_data):
    fakedata_prompt = model_user_prompt(str(describe_dataframe(experimental_data)))
    return (fakedata_prompt,)


@app.cell
def _(bot, fakedata_prompt):
    model = bot(fakedata_prompt)
    return (model,)


@app.cell
def _(model):
    model
    return


@app.cell
def _(json, model):
    function_args = json.loads(model[0].function.arguments)
    print(function_args["placeholder_function"])
    return


@app.cell
def _(bot, json, model):
    trace_and_summary = bot.name_to_tool_map[model[0].function.name](
        **json.loads(model[0].function.arguments)
    )
    return (trace_and_summary,)


@app.cell
def _(trace_and_summary):
    trace_and_summary
    return


@app.cell
def _(pd):
    some_real_data = pd.read_csv("data.csv")
    some_real_data
    return (some_real_data,)


@app.cell
def _(ToolBot, write_and_execute_code):
    realbot = ToolBot(
        system_prompt="You are an expert at creating PyMC models.",
        model_name="gpt-4.1",
        tools=[write_and_execute_code(globals())],
    )
    return (realbot,)


@app.cell
def _(describe_dataframe, some_real_data):
    realdata_prompt = model_user_prompt(str(describe_dataframe(some_real_data)))
    return (realdata_prompt,)


@app.cell
def _(realbot, realdata_prompt):
    realmodel = realbot(realdata_prompt)
    return (realmodel,)


@app.cell
def _(realmodel):
    realmodel
    return


@app.cell
def _(json, realmodel):
    print(json.loads(realmodel[0].function.arguments)["placeholder_function"])
    return


@app.cell
def _(json, realbot, realmodel):
    real_trace_and_summary = realbot.name_to_tool_map[realmodel[0].function.name](
        **json.loads(realmodel[0].function.arguments)
    )
    return (real_trace_and_summary,)


@app.cell
def _(real_trace_and_summary):
    real_trace_and_summary
    return


if __name__ == "__main__":
    app.run()
