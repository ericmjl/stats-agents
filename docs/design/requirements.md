# Design Doc: PyMC Model Builder AgentBot with R2D2 Priors

## Overview

The PyMC Model Builder AgentBot is a command-line LLM agent that automatically generates R2D2-based probabilistic models for laboratory data. It:
- Accepts a freeform natural language description of a biological experiment.
- Socratically asks clarifying questions to elicit variables and experimental structure.
- **Automatically selects the appropriate R2D2 variant** (Shrinkage, GLM, or M2) based on experiment design.
- Proposes a PyMC model (code) as a PEP 723 script using **R2D2 priors as the default archetype**.
- **Calculates variance components** for multiple grouping factors using R2D2M2 framework.
- Suggests alternative distributions where appropriate.
- Outputs a sample pandas DataFrame for the experiment.
- Outputs a model graph and posterior plots (if possible).
- Ensures generated code is syntactically valid and can be compiled by PyMC.
- Iteratively corrects code if errors are detected.

## R2D2 Model Selection Strategy

The agent automatically selects the appropriate R2D2 variant based on experiment characteristics:

### R2D2 Shrinkage (Basic)
**Use when:** Simple linear regression with multiple predictors, no grouping factors
**Example:** Gene expression ~ drug dose + age + weight (no repeated measures)

### R2D2 GLM
**Use when:** Non-Gaussian outcomes (Poisson, Binomial, etc.) with or without simple grouping
**Example:** Count data, binary outcomes, rate data

### R2D2M2 (Multilevel/Laboratory Default)
**Use when:** Multiple grouping factors, repeated measures, complex laboratory designs
**Example:** Laboratory experiments with mouse ID + microRNA ID + stress condition
**This is the DEFAULT for most laboratory data**

---

## Example Conversation and Workflow

Below is a sample end-to-end interaction between the user and the agent, illustrating the Socratic process and the final outputs written to disk.

---

**User:**
> I want to compare the effect of two drugs on blood pressure in mice, measured at baseline and after treatment.

**Agent:**
> Are there any other variables (e.g., age, sex, batch) that should be included as covariates?

**User:**
> Yes, age and sex.

**Agent:**
> Is the assignment of drugs randomized? Are there repeated measurements per mouse, or just baseline and one post-treatment measurement?

**User:**
> Assignment is randomized. There are two measurements per mouse: baseline and post-treatment. For each mouse at each timepoint, we have 4 replicate measurements to capture variability per mouse.

**Agent:**
> Great! To confirm, your variables are:
> - Response: blood pressure
> - Predictors: drug (categorical), age (numeric), sex (categorical)
> - Repeated measures: baseline and post-treatment per mouse
> - Replicates: 4 measurements per mouse per timepoint
> - Randomized assignment
>
> Does this capture your experiment correctly?

**User:**
> Yes, that's correct.

**Agent:**
> Here is a sample pandas DataFrame structure for your experiment:

```python
import pandas as pd

data = pd.DataFrame({
    'mouse_id': [1, 1, 1, 1, 1, 1, 1, 1, ...],
    'timepoint': ['baseline', 'baseline', 'baseline', 'baseline', 'post', 'post', 'post', 'post', ...],
    'replicate': [1, 2, 3, 4, 1, 2, 3, 4, ...],
    'drug': ['A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', ...],
    'age': [12, 12, 12, 12, 12, 12, 12, 12, ...],
    'sex': ['M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', ...],
    'blood_pressure': [120, 121, 119, 122, 115, 116, 114, 117, ...]
})
```

**Agent:**
> I've identified this as a multilevel design with multiple grouping factors. I'll use the **R2D2M2 framework** to automatically allocate variance across predictors and grouping factors.
>
> **Component calculation:**
> - Predictors: age (continuous)
> - Grouping factors: mouse ID, drug, sex
> - Total components = 1 (age) + 3 (grouping factors) = 4 components
>
> Here is the proposed PyMC model as a PEP 723 script:

```python
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "pymc",
#   "pandas",
#   "numpy",
#   "pytensor",
# ]
# ///

import pymc as pm
import pandas as pd
import numpy as np
import pytensor.tensor as pt

data = pd.read_csv('experiment_data.csv')

# Create indices for grouping factors
mouse_ids = pd.Categorical(data['mouse_id']).codes
drug_ids = pd.Categorical(data['drug']).codes
sex_ids = pd.Categorical(data['sex']).codes

coords = {
    'obs': range(len(data)),
    'mice': range(len(data['mouse_id'].unique())),
    'drugs': data['drug'].unique(),
    'sexes': data['sex'].unique(),
    'components': ['age', 'mouse_intercepts', 'drug_intercepts', 'sex_intercepts']
}

with pm.Model(coords=coords) as model:
    # R2D2M2 Framework
    # 1. R² prior
    r_squared = pm.Beta("r_squared", alpha=1, beta=1)

    # 2. Total explained variance τ²
    tau_squared = pm.Deterministic("tau_squared", r_squared / (1 - r_squared))

    # 3. Variance allocation across components
    n_components = 4  # age + mouse + drug + sex
    phi = pm.Dirichlet("phi", a=np.full(n_components, 0.5), dims="components")

    # 4. Residual variance
    sigma = pm.HalfStudentT("sigma", nu=3, sigma=np.std(data['blood_pressure']))
    sigma_squared = pm.Deterministic("sigma_squared", sigma**2)

    # 5. Population-level effect (age)
    age_scale = pm.Deterministic("age_scale",
                                pt.sqrt(sigma_squared * phi[0] * tau_squared))
    age_effect = pm.Normal("age_effect", mu=0, sigma=age_scale)

    # 6. Group-specific intercepts (each gets own variance component)
    # Mouse-specific intercepts
    mouse_scale = pm.Deterministic("mouse_scale",
                                  pt.sqrt(sigma_squared * phi[1] * tau_squared))
    mouse_intercepts = pm.Normal("mouse_intercepts", mu=0, sigma=mouse_scale, dims="mice")

    # Drug-specific intercepts
    drug_scale = pm.Deterministic("drug_scale",
                                 pt.sqrt(sigma_squared * phi[2] * tau_squared))
    drug_intercepts = pm.Normal("drug_intercepts", mu=0, sigma=drug_scale, dims="drugs")

    # Sex-specific intercepts
    sex_scale = pm.Deterministic("sex_scale",
                                pt.sqrt(sigma_squared * phi[3] * tau_squared))
    sex_intercepts = pm.Normal("sex_intercepts", mu=0, sigma=sex_scale, dims="sexes")

    # 7. Linear predictor
    eta = (age_effect * data['age'].values +
           mouse_intercepts[mouse_ids] +
           drug_intercepts[drug_ids] +
           sex_intercepts[sex_ids])

    # 8. Likelihood
    likelihood = pm.Normal("blood_pressure", mu=eta, sigma=sigma,
                          observed=data['blood_pressure'], dims="obs")

# ---

*Note: This R2D2M2 model automatically determines which factors contribute most to explained variance through the φ allocation. For example, if φ = [0.1, 0.6, 0.2, 0.1], then mouse differences account for 60% of total explained variance.*

**Agent:**
> I have written the following files to disk:
> - `biomodel.py` (PyMC model with PEP 723 metadata)
> - `experiment_data.csv` (sample data structure for your experiment)

---

## Laboratory Data Archetype: Multi-Factor R2D2M2

For complex laboratory experiments with multiple crossed grouping factors, the agent should automatically generate R2D2M2 models following this archetype:

### Example: Multi-omics Laboratory Experiment

**Scenario:** Gene expression study with multiple experimental factors
- **Response:** Gene expression level (continuous)
- **Predictors:** Treatment dose (continuous), Age (continuous)
- **Grouping factors:** Mouse ID, MicroRNA ID, Stress condition

**Component calculation:**
```python
n_components = n_predictors + n_grouping_factors
n_components = 2 + 3 = 5 components
```

**Generated model structure:**
```python
def r2d2m2_laboratory_archetype(X, y, mouse_ids, microRNA_ids, stress_conditions):
    n, p = X.shape  # p predictors
    n_grouping_factors = 3  # mouse, microRNA, stress
    n_components = p + n_grouping_factors

    coords = {
        'predictors': [f'x_{i}' for i in range(p)],
        'obs': range(n),
        'components': ([f'population_{i}' for i in range(p)] +
                      ['mouse_intercepts', 'microRNA_intercepts', 'stress_intercepts']),
        'mice': range(len(np.unique(mouse_ids))),
        'microRNAs': range(len(np.unique(microRNA_ids))),
        'stress_conditions': range(len(np.unique(stress_conditions)))
    }

    with pm.Model(coords=coords) as model:
        # R2D2M2 framework
        r_squared = pm.Beta("r_squared", alpha=1, beta=1)
        tau_squared = pm.Deterministic("tau_squared", r_squared / (1 - r_squared))
        phi = pm.Dirichlet("phi", a=np.full(n_components, 0.5), dims="components")

        sigma = pm.HalfStudentT("sigma", nu=3, sigma=np.std(y))
        sigma_squared = pm.Deterministic("sigma_squared", sigma**2)

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

        stress_scale = pm.Deterministic("stress_scale",
                                       pt.sqrt(sigma_squared * phi[p+2] * tau_squared))
        stress_intercepts = pm.Normal("stress_intercepts", mu=0, sigma=stress_scale, dims="stress_conditions")

        # Linear predictor
        eta = (pm.math.dot(X, beta) +
               mouse_intercepts[mouse_ids] +
               microRNA_intercepts[microRNA_ids] +
               stress_intercepts[stress_conditions])

        # Likelihood
        likelihood = pm.Normal("y", mu=eta, sigma=sigma, observed=y, dims="obs")

    return model
```

**Key benefits of this archetype:**
1. **Automatic factor importance**: Model determines which factors contribute most to variance
2. **Interpretable results**: φ components show percentage of variance explained by each factor
3. **Hierarchical pooling**: Groups with little data borrow strength from other groups
4. **Scalable**: Works with any number of grouping factors

**Example interpretation:** If φ = [0.15, 0.25, 0.35, 0.15, 0.10], then:
- Predictor 1 accounts for 15% of explained variance
- Predictor 2 accounts for 25% of explained variance
- Mouse differences account for 35% of explained variance
- MicroRNA differences account for 15% of explained variance
- Stress condition differences account for 10% of explained variance

---

## Pre-Modeling Checklist

Before generating a PyMC model, the agent must ensure it has gathered the following information from the user:

1. **Response Variable**
   - What is the main outcome being measured? (e.g., blood pressure)
2. **Predictor Variables**
   - What are the main predictors or factors? (e.g., drug, age, sex)
   - Are they categorical or continuous?
3. **Experimental Units**
   - What is the unit of analysis? (e.g., mouse, well, patient)
4. **Grouping/Random Effects**
   - Are there repeated measures or nested/clustered data? (e.g., replicates within mouse)
5. **Timepoints**
   - Are there multiple timepoints or longitudinal measurements?
6. **Replicates**
   - How many replicates per unit/timepoint?
7. **Randomization**
   - How are treatments assigned? (randomized, blocked, etc.)
8. **Covariates**
   - Are there additional covariates to control for? (e.g., batch, litter)
9. **Sample Size**
   - How many units per group?
10. **Missing Data**
    - Are there missing values or dropouts to consider?
11. **Other Constraints or Special Features**
    - Any other relevant design features? (e.g., crossover, paired design)

*Note: This checklist should be encoded in the system prompt for the agent, ensuring all necessary information is collected before modeling proceeds.*

*Note: The agent should adapt its language and Socratic style to the user’s apparent statistical expertise. For novice users, minimize jargon and provide gentle guidance. For expert users, use concise, technical language and skip onboarding explanations.*

## AgentBot Structure

### 1. dunder `__call__` Method

The `__call__` method is the main entry point. It orchestrates the workflow:

```python
def __call__(self, experiment_description: str) -> dict:
    """
    :param experiment_description: Freeform description of the biological experiment.
    :return: Dictionary with keys:
        - 'clarifying_questions': List of questions asked to the user.
        - 'final_model_code': PEP 723 PyMC script as a string.
        - 'sample_dataframe': pandas DataFrame (or code to generate it).
        - 'model_graph': (optional) Path or object for model graph visualization.
        - 'posterior_plots': (optional) Path or object for posterior plots.
        - 'error': Any error encountered during model compilation.
    """
    # 1. Parse experiment description, extract variables and structure.
    # 2. Socratically ask clarifying questions (if needed).
    # 3. Propose model structure (best-guess priors/likelihoods, suggest alternatives).
    # 4. Generate PyMC code (PEP 723 script).
    # 5. Generate sample pandas DataFrame.
    # 6. Try compiling the model; if errors, iterate and correct.
    # 7. (Optional) Generate model graph and posterior plots.
    # 8. Return all outputs in a dictionary.
```

### 2. Tools Provided to the Agent

The agent should have access to the following tools (functions or methods):

- **parse_experiment_description**: Extracts variables, response, and structure from the user’s text.
- **ask_clarifying_questions**: Generates and asks follow-up questions to fill in missing details.
- **propose_model_structure**: Suggests a model structure, including best-guess priors/likelihoods and alternatives.
- **generate_pymc_code**: Produces a PEP 723-compliant PyMC script.
- **generate_sample_dataframe**: Produces a sample pandas DataFrame (or code to generate it).
- **compile_and_validate_model**: Attempts to compile the generated PyMC code, catches errors, and suggests corrections.
- **generate_model_graph**: (Optional) Produces a model graph visualization.
- **generate_posterior_plots**: (Optional) Produces posterior plots (using simulated data if needed).

### 3. Socratic Interaction Loop

- After parsing the initial description, the agent should identify ambiguities or missing information and ask targeted, Socratic questions.
- The loop continues until the agent has enough information to propose a model.
- The agent should avoid asking about priors/likelihoods directly, but may suggest alternatives in the output.

### 4. Output Format

- The main output is a PEP 723 script (for easy copy-paste or file output).
- The agent also outputs a sample DataFrame and, optionally, visualizations.
- All outputs are returned as a dictionary for easy CLI or programmatic use.

### 5. Error Handling

- If the generated PyMC code fails to compile, the agent should:
  - Parse the error message.
  - Attempt to correct the code.
  - Repeat until the code compiles or a maximum number of attempts is reached.

### 6. Testing

- Unit tests should:
  - Feed example experiment descriptions and check that the generated code compiles.
  - Check that the sample DataFrame matches the variables in the model.
  - Optionally, check that the model graph and posterior plots are generated.

---

## Example Workflow (Short Version)

1. **User:** “I want to compare the effect of two drugs on blood pressure in mice, measured at baseline and after treatment.”
2. **Agent:** “Are there any other variables (e.g., age, sex, batch) that should be included as covariates?”
3. **User:** “Yes, age and sex.”
4. **Agent:** (Proposes model, generates code, outputs sample DataFrame, etc.)

---

## Implementation Plan

1. Create a new bot in `llamabot/bot/biomodelbot.py` (or similar).
2. Implement the `__call__` method as described.
3. Implement each tool as a function or method.
4. Write CLI entry point using Typer.
5. Write pytest-style tests for code generation and validation.

---

## Notes

- The agent should be extensible to support more complex models in the future (e.g., hierarchical, time series).
- The Socratic questioning logic can be improved iteratively as you see more user interactions.
- For PEP 723, see [PEP 723: Inline Script Metadata](https://peps.python.org/pep-0723/).

## Adaptive Conversation Examples

### Example 1: Novice User

**User:**
I want to see if a new fertilizer helps my plants grow taller compared to the regular one.

**Agent:**
That sounds like a great experiment! To help you build a statistical model, could you tell me a bit more about your setup?
- What is the main thing you’re measuring? (For example, is it the height of each plant?)
- How many groups do you have (e.g., new fertilizer vs. regular)?
- Are there any other factors that might affect plant growth, like sunlight or watering?
- How many plants are in each group?
- Will you measure each plant more than once, or just at the end?

**User:**
I’m measuring the height of each plant at the end. There are two groups: one with the new fertilizer and one with the regular. I have 10 plants in each group. I think they all get the same sunlight and water.

**Agent:**
Perfect, thank you!
- So, your main measurement is plant height.
- You have two groups (new vs. regular fertilizer), with 10 plants in each group.
- No other factors to control for, and you’ll measure each plant once at the end.

Does that sound right? If so, I’ll build a model that compares the average height between the two groups. Would you like to see a sample data table and the code for the model?

**User:**
Yes, please!

**Agent:**
Here’s a sample data table you could use:

| plant_id | fertilizer   | height |
|----------|--------------|--------|
| 1        | new          | 15.2   |
| 2        | new          | 14.8   |
| ...      | ...          | ...    |
| 11       | regular      | 13.5   |
| ...      | ...          | ...    |

And here’s the code for your model (with explanations in the comments):

```python
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "pymc",
#   "pandas",
#   "numpy",
# ]
# ///

import pymc as pm
import pandas as pd

data = pd.read_csv('plant_data.csv')

coords = {'fertilizer': data['fertilizer'].unique()}

with pm.Model(coords=coords) as model:
    fertilizer_idx = pd.Categorical(data['fertilizer']).codes
    mu = pm.Normal('mu', 0, 10)
    fertilizer_effect = pm.Normal('fertilizer_effect', 0, 5, dims='fertilizer')
    sigma = pm.HalfNormal('sigma', 10)
    mu_obs = mu + fertilizer_effect[fertilizer_idx]
    height = pm.Normal('height', mu=mu_obs, sigma=sigma, observed=data['height'].values)
```

---

### Example 2: Expert User

**User:**
I have a 2x2 factorial design: two genotypes (WT, KO), two treatments (vehicle, drug). Each mouse is measured at baseline and post-treatment, with 3 technical replicates per timepoint. I want to model the effect of genotype, treatment, and their interaction on a continuous outcome, accounting for repeated measures and technical replicates.

**Agent:**
Understood.
- Outcome: continuous (please specify variable name if not “outcome”)
- Predictors: genotype (WT/KO), treatment (vehicle/drug), interaction
- Repeated measures: baseline/post per mouse
- Technical replicates: 3 per timepoint per mouse
- Random effects: mouse, replicate nested within mouse
- Any additional covariates or batch effects?

**User:**
No additional covariates. Outcome variable is “response”.

**Agent:**
Thanks. Here’s a sample DataFrame structure:

| mouse_id | timepoint | replicate | genotype | treatment | response |
|----------|-----------|-----------|----------|-----------|----------|
| 1        | baseline  | 1         | WT       | vehicle   | ...      |
| 1        | baseline  | 2         | WT       | vehicle   | ...      |
| ...      | ...       | ...       | ...      | ...       | ...      |

PyMC model (PEP 723):

```python
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "pymc",
#   "pandas",
#   "numpy",
# ]
# ///

import pymc as pm
import pandas as pd
import numpy as np

data = pd.read_csv('experiment_data.csv')

coords = {
    'mouse': data['mouse_id'].unique(),
    'timepoint': data['timepoint'].unique(),
    'genotype': data['genotype'].unique(),
    'treatment': data['treatment'].unique(),
    'replicate': np.arange(1, 4),
}

# ---
# Corrected R2D2M2 prior for all regression coefficients and variance decomposition
# Pattern follows protein_estimation.py and https://ericmjl.github.io/blog/2025/4/3/bayesian-superiority-estimation-with-r2d2-priors-a-practical-guide-for-protein-screening/
# IMPORTANT: W is partitioned ONCE among all effects using a single Dirichlet. Each effect (vector or matrix) gets its own unique variance component from this partition. Do NOT further decompose or multiply by W for subcomponents.

with pm.Model(coords=coords) as model:
    # 1. Residual variance (unexplained)
    sigma_squared = pm.HalfNormal("sigma_squared", sigma=1)
    residual_sd = pm.Deterministic("residual_sd", pm.math.sqrt(sigma_squared))

    # 2. R2 prior and induced total signal variance W
    r_squared = pm.Beta("r_squared", alpha=1, beta=1)
    W = pm.Deterministic("W", sigma_squared * r_squared / (1 - r_squared))

    # 3. Dirichlet split of W among all effects (single partition)
    n_components = 6  # e.g., genotype, treatment, interaction, time, mouse, replicate
    props = pm.Dirichlet("props", a=np.ones(n_components), dims="component")
    comp_vars = props * W
    comp_sds = pm.Deterministic("comp_sds", pm.math.sqrt(comp_vars))

    # 4. Effects (assign each effect its own unique variance component)
    genotype_idx = pd.Categorical(data['genotype']).codes
    treatment_idx = pd.Categorical(data['treatment']).codes
    time_idx = pd.Categorical(data['timepoint']).codes
    mouse_idx = pd.Categorical(data['mouse_id']).codes
    replicate_idx = data['replicate'].values - 1

    genotype_effect = pm.Normal("genotype_effect", mu=0, sigma=comp_sds[0], dims="genotype")
    treatment_effect = pm.Normal("treatment_effect", mu=0, sigma=comp_sds[1], dims="treatment")
    interaction = pm.Normal("interaction", mu=0, sigma=comp_sds[2], dims=("genotype", "treatment"))

    # For continuous predictors (e.g., time)
    # No further Dirichlet or scaling for time, mouse, or replicate effects:
    time_effect = pm.Normal("time_effect", mu=0, sigma=comp_sds[3], dims="timepoint")

    # For random effects (e.g., mouse, replicate)
    n_mouse = len(data['mouse_id'].unique())
    phi_mouse = pm.Dirichlet("phi_mouse", a=np.ones(n_mouse))
    lambda2_mouse = pm.Deterministic("lambda2_mouse", phi_mouse * W)
    mouse_offset = pm.Normal("mouse_offset", mu=0, sigma=pm.math.sqrt(lambda2_mouse), dims="mouse")

    n_replicate = 3 # Assuming 3 technical replicates per timepoint per mouse
    phi_replicate = pm.Dirichlet("phi_replicate", a=np.ones((n_mouse, n_replicate)))
    lambda2_replicate = pm.Deterministic("lambda2_replicate", phi_replicate * W)
    replicate_offset = pm.Normal("replicate_offset", mu=0, sigma=pm.math.sqrt(lambda2_replicate), dims=("mouse", "replicate"))

    # ---

    # 5. Global mean
    mu = pm.Normal("mu", mu=0, sigma=10)

    # 6. Linear predictor
    mu_obs = (
        mu
        + genotype_effect[genotype_idx]
        + treatment_effect[treatment_idx]
        + interaction[genotype_idx, treatment_idx]
        + time_effect[time_idx]
        + mouse_offset[mouse_idx]
        + replicate_offset[mouse_idx, replicate_idx]
    )

    # 7. Likelihood uses only the residual_sd
    response = pm.Normal('response', mu=mu_obs, sigma=residual_sd, observed=data['response'].values)

# ---
```

---

## Xarray for Unified Laboratory Data Storage

*Inspired by [How to use xarray for unified laboratory data storage](https://ericmjl.github.io/blog/2025/7/15/how-to-use-xarray-for-unified-laboratory-data-storage/)*

- All experimental data should be internally represented and output as both CSV (for accessibility) and xarray Datasets (for reproducibility, coordinate-based alignment, and cloud-native workflows).
- The agent should always write both a CSV and an xarray file (NetCDF or Zarr) to disk, and return the file paths as strings.
- All code examples and modeling code should demonstrate how to load and use xarray for coordinate-based data access.
- Tools should pass file paths (strings) for xarray data between steps, not in-memory objects, to ensure compatibility with agent loops and file-based workflows.

### Example: Converting CSV to xarray and Saving to Disk

```python
import pandas as pd
import xarray as xr

df = pd.read_csv('experiment_data.csv')
ds = xr.Dataset.from_dataframe(df.set_index(['mouse_id', 'timepoint', 'replicate', 'drug', 'age', 'sex']))
ds.to_netcdf('experiment_data.nc')  # or ds.to_zarr('experiment_data.zarr')
```

---

## Tool Function Signatures

*Note: All tool functions return strings (JSON, code, or file paths) for compatibility with LLM agent loops. Downstream tools should parse these strings or load files as needed. For xarray data, tools should pass file paths, not in-memory objects.*

Below are the proposed function signatures for each tool the agent will use. Each is decorated with @tool and includes a docstring describing its purpose, inputs, and outputs.

```python
from llamabot import tool, structured_bot
from pydantic import BaseModel, Field
from typing import List, Optional

class ExperimentDescription(BaseModel):
    """
    Structured representation of a biological experiment for modeling.
    """
    response_variable: str = Field(..., description="The main outcome being measured (e.g., blood pressure)")
    predictor_variables: List[str] = Field(..., description="List of main predictors or factors (e.g., drug, age, sex)")
    predictor_types: Optional[List[str]] = Field(None, description="Type of each predictor (e.g., categorical, continuous)")
    experimental_unit: str = Field(..., description="The unit of analysis (e.g., mouse, well, patient)")
    grouping_structure: Optional[str] = Field(None, description="Description of repeated measures or nested/clustered data (e.g., replicates within mouse)")
    timepoints: Optional[List[str]] = Field(None, description="List of timepoints or longitudinal measurements")
    replicates: Optional[int] = Field(None, description="Number of replicates per unit/timepoint")
    randomization: Optional[str] = Field(None, description="How treatments are assigned (e.g., randomized, blocked)")
    covariates: Optional[List[str]] = Field(None, description="Additional covariates to control for (e.g., batch, litter)")
    sample_size: Optional[int] = Field(None, description="Number of units per group")
    missing_data: Optional[str] = Field(None, description="Description of missing values or dropouts, if any")
    other_features: Optional[str] = Field(None, description="Any other relevant design features (e.g., crossover, paired design)")
    description: Optional[str] = Field(None, description="A freeform description or notes about the experiment")

@tool
def parse_experiment_description(description: str) -> ExperimentDescription:
    """
    Parse a freeform experiment description and return a structured ExperimentDescription object.

    This function uses a StructuredBot to extract experiment details from the user's text and returns a validated Pydantic model instance.

    :param description: The user's freeform text describing the experiment.
    :return: An ExperimentDescription Pydantic model instance containing all parsed fields.
    """
    from llamabot.bot.structuredbot import StructuredBot

    system_prompt = """You are an expert at parsing biological experiment descriptions.
    Extract all relevant details from the user's description and populate the ExperimentDescription model.
    Be thorough and ask for clarification if needed."""

    bot = StructuredBot(
        system_prompt=system_prompt,
        pydantic_model=ExperimentDescription,
        model_name="gpt-4o-mini"
    )

    return bot(description)

@tool
def ask_clarifying_questions(parsed_info: dict) -> list:
    """
    Generate a list of clarifying questions based on missing or ambiguous information in the parsed experiment description.

    :param parsed_info: Dictionary of structured experiment information (may have missing or incomplete fields).
    :return: A list of strings, each a question to ask the user to clarify or complete the experiment description.
    """

@tool
def select_r2d2_variant(parsed_info: dict) -> str:
    """
    Automatically select the appropriate R2D2 variant based on experiment characteristics.

    Selection logic:
    - R2D2 Shrinkage: Simple linear regression, no grouping factors
    - R2D2 GLM: Non-Gaussian outcomes (Poisson, Binomial, etc.)
    - R2D2M2: Multiple grouping factors, repeated measures (DEFAULT for laboratory data)

    :param parsed_info: Dictionary of structured experiment information.
    :return: String indicating the selected variant ("shrinkage", "glm", or "m2").
    """

@tool
def calculate_variance_components(parsed_info: dict, r2d2_variant: str) -> dict:
    """
    Calculate the number and names of variance components for R2D2 model.

    For R2D2M2: n_components = n_predictors + n_grouping_factors
    For R2D2 GLM: n_components = n_predictors + simple grouping structure
    For R2D2 Shrinkage: n_components = n_predictors

    :param parsed_info: Dictionary of structured experiment information.
    :param r2d2_variant: Selected R2D2 variant ("shrinkage", "glm", or "m2").
    :return: Dictionary with 'n_components', 'component_names', and 'component_types'.
    """

@tool
def propose_model_structure(parsed_info: dict, r2d2_variant: str, variance_components: dict) -> dict:
    """
    Propose a model structure (variables, effects, priors, likelihoods, and alternatives) based on the parsed experiment information and selected R2D2 variant.

    :param parsed_info: Dictionary of structured experiment information.
    :param r2d2_variant: Selected R2D2 variant ("shrinkage", "glm", or "m2").
    :param variance_components: Dictionary describing variance component structure.
    :return: A dictionary describing the model structure, including variables, effects, and suggested distributions (with alternatives where appropriate).
    """

@tool
def generate_pymc_code(model_structure: dict, parsed_info: dict) -> str:
    """
    Generate a PEP 723-compliant PyMC script for the experiment, using the proposed model structure and experiment information.

    :param model_structure: Dictionary describing the model structure.
    :param parsed_info: Dictionary of structured experiment information.
    :return: A string containing the full PyMC model script, including PEP 723 metadata.
    """

@tool
def generate_sample_dataframe(parsed_info_json: str) -> str:
    """
    Generate both a sample CSV file and an xarray file (NetCDF or Zarr) that match the experiment design, and return their file paths as a JSON string.

    :param parsed_info_json: JSON string of structured experiment information.
    :return: A JSON string with keys 'csv_path' and 'xarray_path', each a file path to the generated data files.
    """

@tool
def compile_and_validate_model(pymc_code: str) -> dict:
    """
    Attempt to compile the generated PyMC code and report any errors or success.

    :param pymc_code: The generated PyMC model code as a string.
    :return: A dictionary with keys 'success' (bool), 'error' (str, if any), and possibly 'traceback' or other diagnostic info.
    """

@tool
def generate_model_graph(pymc_code: str) -> str:
    """
    Generate a visualization of the model graph from the PyMC code.

    :param pymc_code: The generated PyMC model code as a string.
    :return: A string representing the path to the saved model graph image or the image data itself.
    """

@tool
def generate_posterior_plots(pymc_code: str, data_path: str) -> str:
    """
    Generate posterior plots from the PyMC model and data.

    :param pymc_code: The generated PyMC model code as a string.
    :param data_path: Path to the CSV or DataFrame containing the experiment data.
    :return: A string representing the path to the saved posterior plots or the image data itself.
    """
