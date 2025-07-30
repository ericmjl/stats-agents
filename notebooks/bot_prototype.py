# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "llamabot==0.13.0",
#     "marimo",
#     "pandas==2.3.1",
#     "pydantic==2.11.7",
#     "stats-agents==0.0.1",
# ]
#
# [tool.uv.sources]
# stats-agents = { path = "../", editable = true }
# ///

import marimo

__generated_with = "0.14.13"
app = marimo.App(width="full")


@app.cell
def _():
    import llamabot as lmb

    from stats_agents.experiment_parser import ExperimentDescription

    return ExperimentDescription, lmb


@app.cell
def _(ExperimentDescription, lmb):
    system_prompt = """You are an expert at parsing biological experiment descriptions.

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

    experiment_bot = lmb.StructuredBot(
        system_prompt=system_prompt,
        pydantic_model=ExperimentDescription,
        model_name="gpt-4o",
    )
    return (experiment_bot,)


@app.cell
def _():
    example_descriptions = [
        """We're studying the effect of two different drug treatments on blood pressure in mice.
        We have 20 mice total, 10 in each treatment group. We measure blood pressure at baseline
        and after 2 weeks of treatment. The experiment was run across 4 different plates, and
        we want to control for day-to-day variation.""",  # noqa: E501
        """This is a gene expression study comparing three treatment conditions: control,
        low dose, and high dose of a compound. We have 6 mice per treatment group.
        The experiment was conducted over 3 days, with 2 mice per treatment per day.
        We're measuring expression levels of 50 genes. We need to account for plate effects
        since we used 3 different plates.""",  # noqa: E501
        """We're looking at bacterial colony counts under different growth conditions.
        We have 4 different media types and 3 different temperatures. Each combination
        is replicated 5 times. The experiment was run by 2 different operators over
        2 weeks. We want to control for operator effects and time effects.""",
        """We're investigating siRNA encapsulation efficiency in a 96-well plate format
        using a stamp plating approach. The experiment tests 3 inorganic salt identities
        (NaCl, KCl, MgCl2) at 4 concentrations (50mM, 100mM, 200mM, 400mM), 2 buffer
        identities (PBS, Tris-HCl) at 3 concentrations (10mM, 25mM, 50mM), and 3 mixing
        speeds (500rpm, 1000rpm, 1500rpm). We're testing 5 different siRNA species
        (siRNA-A, siRNA-B, siRNA-C, siRNA-D, siRNA-E) across all treatment combinations.
        The experiment uses stamp plating where each treatment combination is applied to
        specific well positions across multiple 96-well plates. We need to control for
        plate effects, day effects, and operator effects, but we're not explicitly
        controlling for well position effects which could introduce spatial bias due
        to the stamp plating method.""",
    ]
    return (example_descriptions,)


@app.cell
def _(example_descriptions, experiment_bot):
    response = experiment_bot(example_descriptions[3])
    response
    return (response,)


@app.cell
def _(response):
    response.dict()
    return


@app.cell
def _(lmb, response):
    from stats_agents.pymc_generator import (
        PyMCModelResponse,
        generate_pymc_model_prompt,
        pymc_generation_system,
    )

    model_bot = lmb.StructuredBot(
        system_prompt=pymc_generation_system(),
        pydantic_model=PyMCModelResponse,
        model_name="gpt-4o",
    )

    model_code = model_bot(generate_pymc_model_prompt(response.model_dump_json()))

    return (model_code,)


@app.cell
def _(model_code):
    print(model_code.dict()["model_code"])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
