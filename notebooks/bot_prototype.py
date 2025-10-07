# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "llamabot==0.13.6",
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

__generated_with = "0.15.2"
app = marimo.App(width="medium")


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

    return (generate_pymc_description,)


@app.cell
def _(example_descriptions, experiment_bot):
    experiment = experiment_bot(
        example_descriptions[4]
    )  # Test the nested replicate example
    experiment
    return (experiment,)


@app.cell
def _(experiment):
    experiment.dict()
    return


@app.cell
def _():
    from stats_agents.pymc_generator import (
        pymc_code_bot,
        generate_pymc_code_prompt,
        pymc_code_generation_system,
    )

    # Step 1: Generate PyMC model code
    # print("Generating PyMC model code...")
    # model_code = pymc_code_bot(generate_pymc_code_prompt(experiment))
    # print("✓ Model code generated")
    return generate_pymc_code_prompt, pymc_code_generation_system


@app.cell
def _():
    import llamabot as lmb
    import ast

    def write_and_execute_code(globals_dict: dict):
        """Write and execute code in a secure sandbox.

        :param globals_dictionary: The dictionary of global variables to use in the sandbox.
        :return: A function that can be used to execute code in the sandbox.
        """

        @lmb.tool
        def write_and_execute_code_wrapper(
            placeholder_function: str, keyword_args: dict = dict()
        ):
            """Write and execute `placeholder_function` with the passed in `keyword_args`.

            Use this tool for any task that requires custom Python code generation and execution.
            This tool has access to ALL globals in the current runtime environment (variables, dataframes, functions, etc.).
            Perfect for: data analysis, calculations, transformations, visualizations, custom algorithms.

            ## Code Generation Guidelines:

            1. **Write self-contained Python functions** with ALL imports inside the function body
            2. **Place all imports at the beginning of the function**: import statements must be the first lines inside the function
            3. **Include all required libraries**: pandas, numpy, matplotlib, etc. - import everything the function needs
            4. **Leverage existing global variables**: Can reference variables that exist in the runtime
            5. **Include proper error handling** and docstrings
            6. **Provide keyword arguments** when the function requires parameters
            7. **Make functions reusable** - they will be stored globally for future use
            8. **ALWAYS RETURN A VALUE**: Every function must explicitly return something - never just print, display, or show results without returning them. Even for plotting functions, return the figure/axes object.

            ## Function Arguments Handling:

            **CRITICAL**: You MUST always pass in keyword_args, which is a dictionary that can be empty, and match the function signature with the keyword_args:

            - **If your function takes NO parameters** (e.g., `def analyze_data():`), then pass keyword_args as an **empty dictionary**: `{}`
            - **If your function takes parameters** (e.g., `def filter_data(min_age, department):`), then pass keyword_args as a dictionary: `{"min_age": 30, "department": "Engineering"}`
            - **Never pass keyword_args that don't match the function signature** - this will cause execution errors

            ## Code Structure Example:

            ```python
            # Function with NO parameters - use empty dict {}
            def analyze_departments():
                '''Analyze department performance.'''
                import pandas as pd
                import numpy as np
                result = fake_df.groupby('department')['salary'].mean()
                return result
            # Function WITH parameters - pass matching keyword_args
            def filter_employees(min_age, department):
                '''Filter employees by criteria.'''
                import pandas as pd
                filtered = fake_df[(fake_df['age'] >= min_age) & (fake_df['department'] == department)]
                return filtered
            ```

            ## Return Value Requirements:

            - **Data analysis functions**: Return the computed results (numbers, DataFrames, lists, dictionaries)
            - **Plotting functions**: Return the figure or axes object (e.g., `return fig` or `return plt.gca()`)
            - **Filter/transformation functions**: Return the processed data
            - **Calculation functions**: Return the calculated values
            - **Utility functions**: Return relevant output (status, processed data, etc.)
            - **Never return None implicitly** - always have an explicit return statement

            ## Code Access Capabilities:

            The generated code will have access to:

            - All global variables and dataframes in the current session
            - Any previously defined functions
            - The ability to import any standard Python libraries within the function
            - The ability to create new reusable functions that will be stored globally

            :param placeholder_function: The function to execute (complete Python function as string).
            :param keyword_args: The keyword arguments to pass to the function (dictionary matching function parameters).
            :return: The result of the function execution.
            """

            # Parse the code to extract the function name
            tree = ast.parse(placeholder_function)
            function_name = None
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    function_name = node.name
                    break

            if function_name is None:
                raise ValueError("No function definition found in the code")

            ns = globals_dict
            compiled = compile(placeholder_function, "<llm>", "exec")
            exec(compiled, globals_dict, ns)

            return ns[function_name](**keyword_args)

        return write_and_execute_code_wrapper

    return (write_and_execute_code,)


@app.cell
def _(
    experiment,
    generate_pymc_code_prompt,
    pymc_code_generation_system,
    write_and_execute_code,
):
    from llamabot.bot.toolbot import ToolBot

    bot = ToolBot(
        system_prompt=pymc_code_generation_system(),
        model_name="gpt-4o",
        tools=[write_and_execute_code(globals())],
    )
    resp = bot(generate_pymc_code_prompt(experiment))
    return bot, resp


@app.cell
def _(resp):
    resp
    return


@app.cell
def _(resp):
    resp[0].function.name, resp[0].function.arguments
    return


@app.cell
def _(bot, resp):
    import json

    model = bot.name_to_tool_map[resp[0].function.name](
        **json.loads(resp[0].function.arguments)
    )
    return json, model


@app.cell
def _(model):
    model
    return


@app.cell
def _(json, resp):
    print(json.loads(resp[0].function.arguments)["placeholder_function"])
    return


@app.cell
def _(bot, resp):
    bot.name_to_tool_map[resp[0].function.name]
    return


@app.cell
def _(experiment, generate_pymc_description, model_code):
    # Step 2: Generate model description
    print("Generating model description...")
    model_description = generate_pymc_description(experiment, model_code)
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
    import io

    sample_data = model_code.generate_sample_data()
    return io, sample_data


@app.cell
def _(sample_data):
    sample_data
    return


@app.cell
def _(io, sample_data):
    import pandas as pd

    pd.read_csv(io.StringIO(sample_data))
    return


if __name__ == "__main__":
    app.run()
