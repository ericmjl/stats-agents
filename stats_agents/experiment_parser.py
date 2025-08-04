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
    - Replicate structure (technical or biological replicates)
    - Experimental aims

    For factor types:
    - TREATMENT: Primary effects of interest (drugs, doses, interventions)
    - NUISANCE: Sources of variation to control for (plate, day, operator, batch effects)
    - BLOCKING: Experimental design stratification (blocks, strata)
    - COVARIATE: Continuous covariates (age, weight, etc.)
    - REPLICATE: Technical or biological replicates

    **CRITICAL: Always look for replicate information in the description:**
    - Look for phrases like "3 technical replicates", "4 biological replicates", "replicated 5 times"
    - Identify what the replicates are nested under (e.g., "3 replicates per plate", "4 replicates per mouse")
    - Determine if they are technical replicates (same sample measured multiple times) or biological replicates (different samples)
    - If replicates are mentioned, create a ReplicateStructure object with:
      - replicate_type: "technical" or "biological"
      - replicates_per_unit: the number of replicates per experimental unit
      - nested_under: what the replicates are nested under (e.g., "plate", "mouse", "treatment")
      - description: a clear description of the replicate structure

    **Example replicate detection:**
    - "3 technical replicates per plate" → ReplicateStructure(replicate_type="technical", replicates_per_unit=3, nested_under="plate")
    - "4 biological replicates per treatment" → ReplicateStructure(replicate_type="biological", replicates_per_unit=4, nested_under="treatment")
    - "Each condition was replicated 5 times" → ReplicateStructure(replicate_type="technical", replicates_per_unit=5, nested_under="condition")

    Return a complete ExperimentDescription with all fields properly populated, including replicate structure when present."""  # noqa: E501


def create_experiment_parser_bot(model_name: str = "gpt-4o") -> lmb.StructuredBot:
    """Create a StructuredBot for parsing experiment descriptions."""
    return lmb.StructuredBot(
        system_prompt=experiment_parsing_system(),
        pydantic_model=ExperimentDescription,
        model_name=model_name,
    )


# Default experiment parsing bot
experiment_bot = create_experiment_parser_bot()
