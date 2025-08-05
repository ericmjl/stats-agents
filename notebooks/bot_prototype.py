# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "llamabot==0.13.0",
#     "marimo",
#     "memo==0.2.4",
#     "numpy==2.3.2",
#     "pandas==2.3.1",
#     "pydantic==2.11.7",
#     "pymc==5.25.1",
#     "pytensor==2.31.7",
#     "stats-agents==0.0.1",
# ]
#
# [tool.uv.sources]
# stats-agents = { path = "../", editable = true }
# ///

import marimo

__generated_with = "0.14.16"
app = marimo.App(width="full")


@app.cell
def _():
    return


@app.cell
def _():
    from stats_agents.experiment_parser import experiment_bot

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
        2 weeks. We want to control for operator effects and time effects.""",  # noqa: E501
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
        to the stamp plating method. We  think there might be a sigmoidal dose-dependent effect
        for salt concentrations, buffer concentrations, and mixing speeds.""",  # noqa: E501
        """We're studying protein expression levels in response to different growth factor
        treatments. We have 3 treatment conditions: control, growth factor A, and growth
        factor B. Each treatment is applied to 4 different cell lines. The experiment
        uses 6 plates total, with each plate containing 3 technical replicates of each
        treatment-cell line combination. So each plate has 36 wells (3 treatments × 4
        cell lines × 3 replicates). We need to control for plate effects and account
        for the nested structure where replicates are nested within plates.
        The readout is done by mass spec with the readout being area under curve of the mass spec peaks""",  # noqa: E501
    ]
    return (example_descriptions,)


@app.cell
def _():
    # Import the new modular functions
    from stats_agents.pymc_generator import (
        generate_pymc_code,
        generate_pymc_description,
    )

    return generate_pymc_code, generate_pymc_description


@app.cell
def _(example_descriptions, experiment_bot):
    response = experiment_bot(
        example_descriptions[4]
    )  # Test the nested replicate example
    response
    return (response,)


@app.cell
def _(response):
    response.dict()
    return


@app.cell
def _(generate_pymc_code, response):
    # Step 1: Generate PyMC model code
    print("Generating PyMC model code...")
    model_code = generate_pymc_code(response)
    print("✓ Model code generated")
    return (model_code,)


@app.cell
def _(generate_pymc_description, model_code, response):
    # Step 2: Generate model description
    print("Generating model description...")
    model_description = generate_pymc_description(response, model_code)
    print("✓ Model description generated")
    return (model_description,)


@app.cell
def _(model_description):
    import marimo as mo

    mo.md(model_description.description)
    return (mo,)


@app.cell(hide_code=True)
def _(mo, model_code):
    mo.md(
        f"""
    ```python
    {model_code.model_code}
    ```
    """
    )
    return


@app.cell
def _(model_code):
    print(model_code.generate_sample_data())
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
