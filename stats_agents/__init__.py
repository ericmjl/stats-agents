"""Top-level API for stats-agents.

This is the file from which you can do:

    from stats_agents import some_function

Use it to control the top-level API of your Python data science project.
"""

# Core schemas and models
# Organized bots
from .experiment_parser import create_experiment_parser_bot
from .models import PyMCModelGenerator, parse_experiment_description
from .pymc_generator import (
    PyMCModelResponse,
    generate_pymc_model,
    generate_pymc_model_prompt,
)
from .schemas import ExperimentalFactor, ExperimentDescription, FactorType, ResponseType

__all__ = [
    # Core schemas
    "ExperimentDescription",
    "ResponseType",
    "FactorType",
    "ExperimentalFactor",
    # Core functions
    "parse_experiment_description",
    "PyMCModelGenerator",
    # Organized bots
    "create_experiment_parser_bot",
    "generate_pymc_model",
    "generate_pymc_model_prompt",
    "PyMCModelResponse",
]
