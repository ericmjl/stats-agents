# stats-agents

A Python library for automatically generating Bayesian statistical models for laboratory experiments using the R2D2 framework and PyMC.

## Overview

stats-agents helps laboratory scientists quickly create sophisticated Bayesian models for their experimental data by:

- **Parsing natural language descriptions** of experiments into structured schemas
- **Automatically generating PyMC models** using the R2D2 (R-squared Dirichlet Decomposition) framework
- **Creating realistic sample data** for model testing and validation
- **Providing comprehensive model documentation** and interpretation guides

## Key Features

### 🔬 **Experiment Parsing**

- Convert freeform experiment descriptions into structured schemas
- Support for treatment factors, nuisance factors, blocking factors, and covariates
- Automatic detection of experimental design structure

### 📊 **R2D2 Model Generation**

- **R2D2M2**: Multilevel models for complex laboratory designs (default)
- **R2D2 GLM**: Generalized linear models for non-Gaussian outcomes
- **R2D2 Shrinkage**: Basic regression with regularization
- Automatic variance component allocation across experimental factors

### 🎯 **PyMC Integration**

- Generate PEP 723 compliant PyMC scripts
- Proper coordinate and dimension setup
- Vectorized operations (no for-loops)
- Support for sigmoidal curves and nonlinear relationships

### 📈 **Sample Data Generation**

- Realistic laboratory data with appropriate value ranges
- CSV and XArray output formats
- Configurable sample sizes and experimental designs

## Quick Start

### Installation

```bash
# Install from source
pip install git+https://github.com/ericmjl/stats-agents.git

# Or clone and install in development mode
git clone https://github.com/ericmjl/stats-agents.git
cd stats-agents
pip install -e .
```

### Basic Usage

```python
from stats_agents import experiment_bot, generate_pymc_model
from stats_agents.schemas import ExperimentDescription

# Describe your experiment
description = """
I want to compare the effect of two drugs (PBS and Tris HCl) on
encapsulation efficiency in mice. I have 20 mice per treatment group,
and I measure encapsulation efficiency as a percentage. The experiment
was conducted over 3 days with 2 operators.
"""

# Generate PyMC model
response = experiment_bot(description)

# Access the generated components
print(response.description)  # Natural language explanation
print(response.model_code)   # PyMC model code
print(response.sample_data_csv)  # Sample data table
```

### Command Line Interface

```bash
# Parse experiment and generate model
stats-agents parse-experiment "Compare drug A vs drug B on blood pressure in mice"

# Generate sample data
stats-agents generate-data "Drug comparison experiment" --samples 200

# Analyze experiment structure
stats-agents analyze-experiment "My experiment description"
```

## Example Output

### Generated PyMC Model

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
import numpy as np
import pandas as pd

# Load and preprocess data
data = pd.read_csv("experiment_data.csv")

# Set up coordinates
coords = {
    "drug": ["PBS", "Tris_HCl"],
    "day": ["day_1", "day_2", "day_3"],
    "operator": ["operator_1", "operator_2"],
    "mouse": data["mouse_id"].unique(),
    "obs": np.arange(len(data))
}

with pm.Model(coords=coords) as model:
    # R2D2M2 framework implementation
    sigma = pm.HalfNormal("sigma", sigma=1.0)
    r_squared = pm.Beta("r_squared", alpha=2, beta=2)
    model_snr = pm.Deterministic("model_snr", r_squared / (1 - r_squared))
    W = pm.Deterministic("W", model_snr * sigma**2)

    # Variance allocation across components
    phi = pm.Dirichlet("phi", a=np.array([70, 20, 7, 3]))

    # Individual effects
    drug_effect = pm.Normal("drug_effect", mu=0, sigma=np.sqrt(phi[0] * W), dims="drug")
    day_effect = pm.Normal("day_effect", mu=0, sigma=np.sqrt(phi[1] * W), dims="day")
    operator_effect = pm.Normal("operator_effect", mu=0, sigma=np.sqrt(phi[2] * W), dims="operator")

    # Predicted response
    predicted = (
        baseline +
        drug_effect[drug_idx] +
        day_effect[day_idx] +
        operator_effect[operator_idx]
    )

    # Likelihood
    y = pm.Normal("y", mu=predicted, sigma=sigma, observed=data["encapsulation_efficiency"], dims="obs")
```

### Sample Data

```csv
mouse_id,drug,day,operator,encapsulation_efficiency
1,PBS,day_1,operator_1,85.2
1,PBS,day_1,operator_1,87.1
2,Tris_HCl,day_1,operator_1,92.3
...
```

## Documentation

- **[API Reference](api.md)**: Complete API documentation
- **[R2D2 Framework](r2d2m2_summary.md)**: Understanding the R2D2 approach
- **[Design Requirements](design/requirements.md)**: Technical design details


## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use stats-agents in your research, please cite:

```bibtex
@software{stats_agents,
  title={stats-agents: Automated Bayesian Model Generation for Laboratory Experiments},
  author={Eric J. Ma},
  year={2024},
  url={https://github.com/ericmjl/stats-agents}
}
```
