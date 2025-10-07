# stats-agents

A Python library for automatically generating Bayesian statistical models for laboratory experiments using the R2D2 framework and PyMC.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

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

## Installation

### From Source (Recommended)

```bash
# Clone the repository
git clone https://github.com/ericmjl/stats-agents.git
cd stats-agents

# Install in development mode
pip install -e .
```

### Using pixi (Development)

```bash
# Install pixi if you haven't already
curl -fsSL https://pixi.sh/install.sh | bash

# Install dependencies and activate environment
pixi install
pixi run python -c "import stats_agents; print('Installation successful!')"
```

## Quick Start

### Basic Usage

```python
from stats_agents import experiment_bot

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

- **[Full Documentation](https://ericmjl.github.io/stats-agents/)**: Complete documentation with examples
- **[API Reference](https://ericmjl.github.io/stats-agents/api/)**: Detailed API documentation
- **[R2D2 Framework](https://ericmjl.github.io/stats-agents/r2d2m2_summary/)**: Understanding the R2D2 approach
- **[Design Requirements](https://ericmjl.github.io/stats-agents/design/requirements/)**: Technical design details

## Supported Experiment Types

### Basic Comparisons
- Drug A vs Drug B comparisons
- Treatment vs Control experiments
- Dose-response studies

### Complex Designs
- Multilevel experiments with nested factors
- Longitudinal studies with repeated measures
- Factorial designs with multiple treatments

### Specialized Data Types
- **Count data**: Colony counts, cell counts (Poisson/Negative Binomial)
- **Binary outcomes**: Success/failure, alive/dead (Binomial)
- **Proportions**: Percentages, rates (Beta with logit transform)
- **Continuous measurements**: Concentrations, weights, absorbance (Gaussian)

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone and install in development mode
git clone https://github.com/ericmjl/stats-agents.git
cd stats-agents
pip install -e .

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest

# Build documentation
mkdocs serve
```

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

## Acknowledgments

- Built with [PyMC](https://www.pymc.io/) for probabilistic programming
- Powered by [llamabot](https://github.com/ericmjl/llamabot) for LLM integration
- Inspired by the R2D2 framework for Bayesian regularization

---

Made with ❤️ by Eric J. Ma (@ericmjl).
