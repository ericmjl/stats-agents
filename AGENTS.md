# AGENTS.md - Project Documentation and Guidelines

## Overview

This document provides comprehensive guidance for working with the stats-agents
project, including development commands, architecture details, and markdown
linting insights.

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

**Marimo Notebook Validation**:

For any changes to marimo notebooks (notebooks/*.py files), always run:

```bash
uvx marimo check /path/to/notebook.py
```

**Important**: If marimo check finds issues, always fix them immediately.
Common issues include:

- Empty cells that can be removed (containing only whitespace, comments, or pass)
- Unused imports or variables
- Code quality issues

**Documentation Guidelines**:

- Do not write "recent fixes" or "changes made" into AGENTS.md
- Keep documentation focused on guidelines and best practices
- Avoid including historical change logs or specific fixes applied

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

**Core Purpose**: This is a collection of statistical agents implemented using
LlamaBot and PyMC for biological experiment modeling and analysis.

**Key Components**:

- `stats_agents/cli.py` - Typer-based CLI with basic commands
- `stats_agents/models.py` - Core statistical models (currently minimal)
- `stats_agents/preprocessing.py` - Data preprocessing utilities
- `stats_agents/schemas.py` - Data validation schemas
- `stats_agents/utils.py` - General utility functions

**Design Philosophy**:

- Based on design document in `docs/design/requirements.md`, the main goal is
  a PyMC Model Builder AgentBot that:
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

**Testing**: Uses pytest with coverage reporting, hypothesis for property-based
testing.

**Documentation**: MkDocs with Material theme, mknotebooks plugin for
executable notebooks.

**Environments**:

- `default` - main development environment
- `docs` - documentation building
- `tests` - testing only
- `cuda` - CUDA-enabled environment for GPU acceleration

## Blog Post Guidelines

- For the blog post, always check the .tex file for references

## Markdown Linting Insights

### Markdownlint Setup

- `markdownlint` is available at `/Users/ericmjl/.pixi/bin/markdownlint`
- Successfully ran on all markdown files in the project
- Found numerous linting issues across multiple files

### Common Issues Identified

#### **Line Length (MD013)**

- Most frequent issue: Lines exceeding 80 characters
- Found in virtually all markdown files
- Examples:
  - `CLAUDE.md:3:81` - Line 102 characters
  - `docs/blog.md:5:81` - Line 527 characters
  - `README.md:3:81` - Line 135 characters

#### **Fenced Code Block Issues**

- **MD031**: Missing blank lines around fenced code blocks
- **MD040**: Missing language specification in code blocks
- Found extensively in design documents and technical documentation

#### **List Formatting (MD032)**

- Missing blank lines around lists
- Affects readability and consistency
- Common in technical documentation

#### **Heading Issues**

- **MD022**: Missing blank lines around headings
- **MD024**: Duplicate headings in same document
- **MD036**: Emphasis used instead of headings

#### **URL Formatting (MD034)**

- Bare URLs without proper markdown link formatting
- Found in references and documentation

### Files with Most Issues

1. `docs/design/conversational_experiment_designer.md` - 100+ issues
2. `docs/design/requirements.md` - 80+ issues
3. `docs/blog.md` - 60+ issues
4. `docs/r2d2_glm_summary.md` - 40+ issues

### Recommended Actions

#### **Immediate Fixes**

1. **Line Length**: Break long lines at 80 characters
2. **Code Blocks**: Add language specifications and blank lines
3. **Lists**: Add blank lines around all lists
4. **Headings**: Ensure proper spacing around headings

#### **Automated Solutions**

- Set up pre-commit hooks to run markdownlint
- Configure IDE to show markdownlint warnings
- Use markdownlint configuration file for project-specific rules

### Project-Specific Insights

#### **Technical Documentation**

- Design documents have complex formatting needs
- Code examples need proper language tags
- API documentation requires consistent formatting

#### **Academic Content**

- R2D2 summaries contain mathematical content
- References need proper URL formatting
- Citations require consistent formatting

### Next Steps

1. Create `.markdownlint.json` configuration file
2. Set up automated linting in CI/CD
3. Fix critical formatting issues
4. Establish markdown style guide for the project

## Implementation Notes

- Markdownlint successfully identified formatting inconsistencies
- Most issues are fixable with automated tools
- Consistent formatting will improve documentation readability
- Consider markdownlint as part of the development workflow
