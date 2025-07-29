# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

**Package manager**: This project uses Pixi for dependency management.

**Setup**:
```bash
pixi install
```

**Testing**:
```bash
pixi run test          # Run all tests
pytest tests/          # Run tests directly (if in activated environment)
```

**Linting and Code Quality**:
```bash
pixi run lint          # Run pre-commit hooks on all files
```

**Documentation**:
```bash
pixi run build-docs    # Build documentation with MkDocs
pixi run serve-docs    # Serve documentation locally
```

**CLI Usage**:
The project provides a CLI entry point via `stats-agents` command:
```bash
stats-agents hello     # Echo project name
stats-agents describe  # Describe the project
```

## Architecture and Project Structure

**Core Purpose**: This is a collection of statistical agents implemented using LlamaBot and PyMC for biological experiment modeling and analysis.

**Key Components**:
- `stats_agents/cli.py` - Typer-based CLI with basic commands
- `stats_agents/models.py` - Core statistical models (currently minimal)
- `stats_agents/preprocessing.py` - Data preprocessing utilities
- `stats_agents/schemas.py` - Data validation schemas
- `stats_agents/utils.py` - General utility functions

**Design Philosophy**:
- Based on design document in `docs/design/requirements.md`, the main goal is a PyMC Model Builder AgentBot that:
  - Accepts natural language descriptions of biological experiments
  - Uses Socratic questioning to clarify experimental design
  - Generates PyMC models with R2D2M2 priors as PEP 723 scripts
  - Outputs sample DataFrames and visualizations
  - Uses both CSV and xarray formats for data storage

**PyMC Model Patterns**:
- Uses R2D2M2 priors for variance decomposition
- Follows pattern from `protein_estimation.py` reference
- Single Dirichlet partition of variance W among all effects
- Each effect gets unique variance component (no further decomposition)
- Always includes PEP 723 metadata for script execution

**Technology Stack**:
- **Backend**: PyMC for Bayesian modeling
- **CLI**: Typer for command-line interface
- **Data**: pandas + xarray for data handling
- **LLM Integration**: LlamaBot (particularly StructuredBot)
- **Dependencies**: JAX, NumPy, SciPy, Matplotlib, Seaborn

**Testing**: Uses pytest with coverage reporting, hypothesis for property-based testing.

**Documentation**: MkDocs with Material theme, mknotebooks plugin for executable notebooks.

**Environments**:
- `default` - main development environment
- `docs` - documentation building
- `tests` - testing only
- `cuda` - CUDA-enabled environment for GPU acceleration
