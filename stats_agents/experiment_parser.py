"""Experiment parsing bot using llamabot StructuredBot."""

import llamabot as lmb

from .schemas import ExperimentDescription


@lmb.prompt(role="system")
def experiment_parsing_system():
    """You are an expert at parsing biological experiment descriptions.

    Extract all relevant details from the user's description and populate the ExperimentDescription model.
    Be thorough and identify:
    - Response variables and their types
    - Treatment factors (primary effects of interest)
    - Nuisance factors (sources of variation to control for like plate, day, operator effects)
    - Blocking factors (experimental design stratification)
    - Covariates (continuous variables)
    - Experimental units
    - Timepoints for longitudinal data
    - Experimental aims

    For factor types:
    - TREATMENT: Primary effects of interest (drugs, doses, interventions)
    - NUISANCE: Sources of variation to control for (plate, day, operator, batch effects)
    - BLOCKING: Experimental design stratification (blocks, strata)
    - COVARIATE: Continuous covariates (age, weight, etc.)

    Return a complete ExperimentDescription with all fields properly populated."""  # noqa: E501


def create_experiment_parser_bot(model_name: str = "gpt-4o") -> lmb.StructuredBot:
    """Create a StructuredBot for parsing experiment descriptions."""
    return lmb.StructuredBot(
        system_prompt=experiment_parsing_system(),
        pydantic_model=ExperimentDescription,
        model_name=model_name,
    )


# Default experiment parsing bot
experiment_bot = create_experiment_parser_bot()
