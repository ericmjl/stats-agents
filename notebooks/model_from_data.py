# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "anthropic",
#     "arviz",
#     "backoff",
#     "llamabot[all]",
#     "marimo",
#     "matplotlib",
#     "numpy",
#     "orjson",
#     "pandas",
#     "pyarrow",
#     "pymc",
#     "pytensor",
#     "seaborn",
#     "stats-agents",
# ]
#
# [tool.uv.sources]
# stats-agents = { path = "../", editable = true }
# ///

import marimo

__generated_with = "0.15.5"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""This notebook is all about constructing a plausible statistical estimation model using existing data that is similar but not identical to the actual lab data coming off a machine. The data might be dummy runs, or it might be range-finding experiments, or prior lab experiments where we are trying to "optimize" the experiment for logistical ease or readout sensitivity. The other setting in which I see this being handy is when we have the data table and a rough description of the experiment data and we just need to bang out a model real quick."""
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""For this, I am redoing an analysis of data from a paper I co-authored with Mia Lieberman while I was in graduate school. The data I am concerned with is biofilm formation, which is measured by OD570 in a crystal violet assay, and I want it to be normalized by OD600, which is a measure (though imperfect) of bacterial density."""
    )
    return


@app.cell
def _(pd):
    df = pd.read_csv(
        "https://raw.githubusercontent.com/ericmjl/mia-stats/refs/heads/master/biofilm/biofilm_2018_all.csv"
    )

    # First, let's see how many contaminated samples we have
    print("Contamination summary:")
    print(df["is_contaminated?"].value_counts())
    print(f"\nTotal samples before filtering: {len(df)}")

    # Filter out contaminated samples
    df_clean = df[df["is_contaminated?"] != 1].copy()

    print(f"Total samples after removing contaminated: {len(df_clean)}")
    print(f"Removed {len(df) - len(df_clean)} contaminated samples")

    # Replace df with the cleaned version
    df = df_clean

    # Verify the filtering worked
    print("Verification - is_contaminated values remaining:")
    print(df["is_contaminated?"].value_counts())
    return (df,)


@app.cell
def _(df):
    df
    return


@app.cell
def _():
    import pymc as pm

    return (pm,)


@app.cell
def _(df):
    df["expt"].unique()
    return


@app.cell
def _():
    import pandas as pd
    import marimo as mo
    from stats_agents import generate_pymc_model
    from stats_agents.experiment_parser import experiment_bot

    return experiment_bot, generate_pymc_model, mo, pd


@app.cell
def _(df):
    df["Isolate"].unique()
    return


@app.cell
def _():
    model_description = """This is a model that is used to estimate the biofilm-formation activity of bacterial isolates ("Isolate",
    numbered 11  13  16  46  85  90  58  83  86  91 100 124 125 128 129,
    from bacterial strains ("ST", and are 55, 4, and 48 respectively) in macaques that are infected with those isolates ("Macaque ID").
    So the strains are nested under isolates.
    We do not a priori believe that the strains are going to be necessarily different from one another.
    Macaque IDs are '1-11' '2-11' '2-14' '1-14' '2-16' '1-15' '4-13' '6-16' '5-16' '3-16'
     '6-17' '7-17' '8-16' '9-16'.
    There are six replicates of OD600 and OD570 readouts, from which a "Normalized OD570" is calculated
    by dividing OD570 by OD600.
    Our goal is to study normalized OD570. We think the OD570 should be log transformed to be in Gaussian space
    because it is a ratio.
    """
    return (model_description,)


@app.cell
def _(experiment_bot, model_description):
    experiment_description = experiment_bot(model_description)
    experiment_description
    return (experiment_description,)


@app.cell
def _(experiment_description):
    experiment_description.dict()
    return


@app.cell
def _(experiment_description, generate_pymc_model):
    model_code = generate_pymc_model(experiment_description)
    return (model_code,)


@app.cell
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
def _(df):
    df
    return


@app.cell
def _():
    import seaborn as sns

    return (sns,)


@app.cell
def _(df, sns):
    import matplotlib.pyplot as plt

    # Create a comprehensive plot showing normalized OD570 grouped by experiment, ST, and isolate
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Plot 1: Box plot by experiment
    sns.boxplot(data=df, x="expt", y="Normalized OD570", ax=axes[0, 0])
    axes[0, 0].set_title("Normalized OD570 by Experiment")
    axes[0, 0].set_xlabel("Experiment")

    # Plot 2: Box plot by ST
    sns.boxplot(data=df, x="ST", y="Normalized OD570", ax=axes[0, 1])
    axes[0, 1].set_title("Normalized OD570 by ST")
    axes[0, 1].set_xlabel("ST")

    # Plot 3: Box plot by Isolate (rotated labels due to many isolates)
    sns.boxplot(data=df, x="Isolate", y="Normalized OD570", ax=axes[1, 0])
    axes[1, 0].set_title("Normalized OD570 by Isolate")
    axes[1, 0].set_xlabel("Isolate")
    axes[1, 0].tick_params(axis="x", rotation=45)

    # Plot 4: Violin plot showing interaction between ST and experiment
    sns.violinplot(data=df, x="expt", y="Normalized OD570", hue="ST", ax=axes[1, 1])
    axes[1, 1].set_title("Normalized OD570 by Experiment and ST")
    axes[1, 1].set_xlabel("Experiment")

    plt.tight_layout()
    plt.gca()
    return (plt,)


@app.cell
def _(df, plt, sns):
    # More detailed faceted plot showing isolates within ST and experiments
    plt.figure(figsize=(16, 10))

    # Create a FacetGrid to show isolates grouped by ST and experiment
    g = sns.FacetGrid(
        df, col="ST", row="expt", margin_titles=True, height=3, aspect=1.2
    )
    g.map(sns.boxplot, "Isolate", "Normalized OD570")
    g.set_titles(col_template="ST {col_name}", row_template="Experiment {row_name}")

    # Add horizontal reference line at y=1.0 and rotate x-axis labels
    for ax in g.axes.flat:
        ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.7, linewidth=1)
        ax.tick_params(axis="x", rotation=45)

    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    plt.gca()
    return


@app.cell
def _(df, plt, sns):
    def _():
        # Check distribution of isolate, macaque ID, and ST across experiments
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Count plot: Isolates by experiment
        isolate_counts = (
            df.groupby(["expt", "Isolate"]).size().reset_index(name="count")
        )
        sns.heatmap(
            isolate_counts.pivot(index="Isolate", columns="expt", values="count"),
            annot=True,
            cmap="Blues",
            ax=axes[0, 0],
        )
        axes[0, 0].set_title("Isolate Distribution Across Experiments")
        axes[0, 0].set_xlabel("Experiment")

        # Count plot: Macaque ID by experiment
        macaque_counts = (
            df.groupby(["expt", "Macaque ID"]).size().reset_index(name="count")
        )
        sns.heatmap(
            macaque_counts.pivot(index="Macaque ID", columns="expt", values="count"),
            annot=True,
            cmap="Greens",
            ax=axes[0, 1],
        )
        axes[0, 1].set_title("Macaque ID Distribution Across Experiments")
        axes[0, 1].set_xlabel("Experiment")

        # Count plot: ST by experiment
        st_counts = df.groupby(["expt", "ST"]).size().reset_index(name="count")
        sns.heatmap(
            st_counts.pivot(index="ST", columns="expt", values="count"),
            annot=True,
            cmap="Reds",
            ax=axes[1, 0],
        )
        axes[1, 0].set_title("ST Distribution Across Experiments")
        axes[1, 0].set_xlabel("Experiment")

        # Summary statistics table
        axes[1, 1].axis("off")
        summary_text = f"""
        Distribution Summary:
        - Total observations: {len(df)}
        - Experiments: {df["expt"].nunique()}
        - Isolates: {df["Isolate"].nunique()}
        - Macaque IDs: {df["Macaque ID"].nunique()}
        - STs: {df["ST"].nunique()}
        - Observations per experiment: {df["expt"].value_counts().to_dict()}
        """
        axes[1, 1].text(0.1, 0.5, summary_text, fontsize=12, verticalalignment="center")

        plt.tight_layout()
        return plt.gca()

    _()
    return


@app.cell
def _(df, pd):
    # Create crosstab tables to check balance more precisely
    print("Isolate vs Experiment crosstab:")
    isolate_crosstab = pd.crosstab(df["Isolate"], df["expt"], margins=True)
    print(isolate_crosstab)

    print("\nMacaque ID vs Experiment crosstab:")
    macaque_crosstab = pd.crosstab(df["Macaque ID"], df["expt"], margins=True)
    print(macaque_crosstab)

    print("\nST vs Experiment crosstab:")
    st_crosstab = pd.crosstab(df["ST"], df["expt"], margins=True)
    print(st_crosstab)
    return


@app.cell
def _():
    import arviz as az

    return (az,)


@app.cell
def _(df, pm):
    import numpy as np
    import pytensor.tensor as pt

    # Load the data
    # Assuming the data is in a CSV file named 'experiment_data.csv'
    data = df

    # Convert categorical data to codes for indexing
    isolate_idx = data["Isolate"].astype("category").cat.codes
    st_idx = data["ST"].astype("category").cat.codes
    macaque_idx = data["Macaque ID"].astype("category").cat.codes

    # Define coordinates for PyMC model
    coords = {
        "isolate": [
            "11",
            "13",
            "16",
            "46",
            "85",
            "90",
            "58",
            "83",
            "86",
            "91",
            "100",
            "124",
            "125",
            "128",
            "129",
        ],
        "st": ["55", "4", "48"],
        "macaque": [
            "1-11",
            "2-11",
            "2-14",
            "1-14",
            "2-16",
            "1-15",
            "4-13",
            "6-16",
            "5-16",
            "3-16",
            "6-17",
            "7-17",
            "8-16",
            "9-16",
        ],
        "variance_components": ["isolate", "st", "macaque"],
        "obs": np.arange(len(data)),
    }

    # Build the PyMC model
    with pm.Model(coords=coords) as model:
        # 1. Measurement precision (unexplained variance)
        sigma = pm.HalfNormal("sigma", sigma=1.0)

        # 2. Model fit quality (proportion of variation explained by experimental factors)
        r_squared = pm.Beta("r_squared", alpha=2, beta=2)  # Expect moderate fit

        # 3. Total experimental effect strength (signal-to-noise ratio)
        model_snr = pm.Deterministic("model_snr", r_squared / (1 - r_squared))

        # 4. Total explainable variance
        W = pm.Deterministic("W", model_snr * sigma**2)

        # 5. Variance proportions across components
        phi = pm.Dirichlet("phi", a=np.array([50, 30, 20]), dims="variance_components")

        # 6. Individual variance components
        isolate_variance = pm.Deterministic("isolate_variance", phi[0] * W)
        st_variance = pm.Deterministic("st_variance", phi[1] * W)
        macaque_variance = pm.Deterministic("macaque_variance", phi[2] * W)

        # 7. Individual treatment effects
        isolate_effect = pm.Normal(
            "isolate_effect", mu=0, sigma=pt.sqrt(isolate_variance), dims="isolate"
        )

        # 8. Nuisance factor effects
        st_effect = pm.Normal("st_effect", mu=0, sigma=pt.sqrt(st_variance), dims="st")

        # 9. Blocking factor effects
        macaque_effect = pm.Normal(
            "macaque_effect", mu=0, sigma=pt.sqrt(macaque_variance), dims="macaque"
        )

        # 10. Global intercept
        mu = pm.Normal("mu", mu=0, sigma=10)

        # 11. Predicted response using broadcasting (no loops)
        predicted_response = (
            mu  # Global intercept
            + isolate_effect[isolate_idx]
            + st_effect[st_idx]
            + macaque_effect[macaque_idx]
        )

        # 12. Observed data
        y = pm.Normal(
            "y",
            mu=predicted_response,
            sigma=sigma,
            observed=np.log(data["Normalized OD570"]),
            dims="obs",
        )
    return (model,)


@app.cell
def _(log_OD570):
    log_OD570
    return


@app.cell
def _(mo, model, pm):
    mo.mermaid(pm.model_to_mermaid(model))
    return


@app.cell
def _(model, pm):
    with model:
        idata = pm.sample()
    return (idata,)


@app.cell
def _(az, idata):
    az.plot_posterior(idata, var_names=["r_squared"])
    return


@app.cell
def _(az, idata):
    az.plot_posterior(idata, var_names=["st_effect"], ref_val=0)
    return


@app.cell
def _(az, idata):
    az.plot_posterior(idata, var_names=["isolate_effect"], ref_val=0)
    return


@app.cell
def _(az, idata):
    az.plot_posterior(idata, var_names=["macaque_id_effect"], ref_val=0)
    return


@app.cell
def _(az, idata):
    az.plot_posterior(idata, var_names=["model_snr"], ref_val=0)
    return


@app.cell
def _(idata):
    idata.posterior
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    # Model Results Commentary

    ## ✅ What's Working Well

    **Blocking factors are behaving as expected:** The fact that both macaque ID and strain effects are centered around zero is excellent news. This suggests that:

    - Your experimental blocking by macaque was effective - individual macaques aren't introducing systematic bias
    - The strain-level grouping isn't the primary driver of biofilm formation differences
    - The biological signal is indeed coming from isolate-level variation, which is exactly what you want to measure

    **Strong isolate effects detected:** The isolates showing clear deviations from zero (isolates 11, 58, 91, 125, 128, 129) indicate your model is successfully capturing meaningful biological differences in biofilm formation capacity between bacterial isolates.

    ## 🚨 The Red Flag: Unexplained Variation

    Your R² of ~0.53 (95% CI: 0.36-0.69) is concerning and suggests a **missing variable problem**. With nearly half the variation unexplained, you're likely dealing with:

    **Experimental batch effects:** Different experiments may have had:

    - Varying incubation conditions (temperature, humidity, timing)
    - Different media preparations or reagent batches
    - Laboratory handling differences across experimental sessions
    - Plate-to-plate variation within experiments

    ## 📋 Recommended Next Steps

    1. **Include experiment ID as a random effect** in your model - this should be treated as another blocking factor alongside macaque ID

    2. **Examine experiment-specific patterns** more closely:
        - Look at residuals plotted by experiment
        - Check if certain isolates consistently perform differently across experiments
        - Investigate if there are systematic time trends across experiments

    4. **Consider a nested/hierarchical structure** where technical replicates are nested within biological replicates within experiments

    ## 💡 Key Takeaway

    The good news is that your core biological signal (isolate effects) appears robust despite the experimental noise. Adding experiment as a random effect should substantially improve your R² and give you more confidence in your isolate-level conclusions.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    We are going to try again, this time by adding in the effect for experiment.
    The rationale for this is to see whether the isolate effect is still present or not.
    """
    )
    return


@app.cell
def _():
    model2_description = """This is a model that is used to estimate the biofilm-formation activity of bacterial isolates ("Isolate",
    numbered 11  13  16  46  85  90  58  83  86  91 100 124 125 128 129,
    from bacterial strains ("ST", and are 55, 4, and 48 respectively) in macaques that are infected with those isolates ("Macaque ID").
    So the strains are nested under isolates.
    We do not a priori believe that the strains are going to be necessarily different from one another.
    Macaque IDs are '1-11' '2-11' '2-14' '1-14' '2-16' '1-15' '4-13' '6-16' '5-16' '3-16'
     '6-17' '7-17' '8-16' '9-16'.
    We have a nuisance factor to control for, which is the experiment number (column "expt"),
    there are the following: 1 2 3 4 5 6. We can just use df["expt"].unique() to get this out.
    There are six replicates of OD600 and OD570 readouts, from which a "Normalized OD570" is calculated
    by dividing OD570 by OD600.
    Our goal is to study normalized OD570. We think the OD570 should be log transformed to be in Gaussian space
    because it is a ratio.
    """
    return (model2_description,)


@app.cell
def _(experiment_bot, model2_description):
    experiment2_description = experiment_bot(model2_description)
    experiment2_description
    return (experiment2_description,)


@app.cell
def _(experiment2_description, generate_pymc_model):
    model2_code = generate_pymc_model(experiment2_description)
    return (model2_code,)


@app.cell
def _(mo, model2_code):
    mo.md(
        f"""
    ```python
    {model2_code.model_code}
    ```
    """
    )
    return


@app.cell
def _(df):
    def _():
        import pymc as pm
        import numpy as np
        import pandas as pd
        import pytensor.tensor as pt

        # Load the data
        # Assuming the data is in a CSV file named 'experiment_data.csv'
        data = df

        # Convert categorical data to codes for indexing
        isolate_idx = data["Isolate"].astype("category").cat.codes
        st_idx = data["ST"].astype("category").cat.codes
        macaque_id_idx = data["Macaque ID"].astype("category").cat.codes
        expt_idx = data["expt"].astype("category").cat.codes

        # Log transform the response variable
        log_od570 = np.log(data["OD570"])

        # Define coordinates for PyMC model
        treatment_levels = [
            "11",
            "13",
            "16",
            "46",
            "85",
            "90",
            "58",
            "83",
            "86",
            "91",
            "100",
            "124",
            "125",
            "128",
            "129",
        ]
        st_levels = ["55", "4", "48"]
        macaque_id_levels = [
            "1-11",
            "2-11",
            "2-14",
            "1-14",
            "2-16",
            "1-15",
            "4-13",
            "6-16",
            "5-16",
            "3-16",
            "6-17",
            "7-17",
            "8-16",
            "9-16",
        ]
        expt_levels = ["1", "2", "3", "4", "5", "6"]

        coords = {
            "isolate": treatment_levels,
            "st": st_levels,
            "macaque_id": macaque_id_levels,
            "expt": expt_levels,
            "variance_components": ["isolate", "st", "macaque_id", "expt"],
            "obs": np.arange(len(data)),
        }

        with pm.Model(coords=coords) as model:
            # 1. Measurement precision (unexplained variance)
            sigma = pm.HalfNormal("sigma", sigma=1.0)

            # 2. Model fit quality (proportion of variation explained by experimental factors)
            r_squared = pm.Beta("r_squared", alpha=2, beta=2)  # Expect moderate fit

            # 3. Total experimental effect strength (signal-to-noise ratio)
            model_snr = pm.Deterministic("model_snr", r_squared / (1 - r_squared))

            # 4. Total explainable variance
            W = pm.Deterministic("W", model_snr * sigma**2)

            # 5. Variance proportions across components
            phi = pm.Dirichlet(
                "phi", a=np.array([40, 30, 20, 10]), dims="variance_components"
            )

            # 6. Individual variance components
            isolate_variance = pm.Deterministic("isolate_variance", phi[0] * W)
            st_variance = pm.Deterministic("st_variance", phi[1] * W)
            macaque_id_variance = pm.Deterministic("macaque_id_variance", phi[2] * W)
            expt_variance = pm.Deterministic("expt_variance", phi[3] * W)

            # 7. Individual treatment effects
            isolate_effect = pm.Normal(
                "isolate_effect",
                mu=0,
                sigma=pt.sqrt(isolate_variance),
                dims="isolate",
            )

            # 8. Nuisance factor effects
            st_effect = pm.Normal(
                "st_effect", mu=0, sigma=pt.sqrt(st_variance), dims="st"
            )
            macaque_id_effect = pm.Normal(
                "macaque_id_effect",
                mu=0,
                sigma=pt.sqrt(macaque_id_variance),
                dims="macaque_id",
            )
            expt_effect = pm.Normal(
                "expt_effect", mu=0, sigma=pt.sqrt(expt_variance), dims="expt"
            )

            # 9. Global intercept
            mu = pm.Normal("mu", mu=0, sigma=10)

            # 10. Predicted response using broadcasting (no loops)
            predicted_response = (
                mu
                + isolate_effect[isolate_idx]
                + st_effect[st_idx]
                + macaque_id_effect[macaque_id_idx]
                + expt_effect[expt_idx]
            )

            # 11. Observed data
            y = pm.Normal(
                "y",
                mu=predicted_response,
                sigma=sigma,
                observed=log_od570,
                dims="obs",
            )

        return model

    model2 = _()
    return (model2,)


@app.cell
def _(model2, pm):
    with model2:
        idata2 = pm.sample()
    return (idata2,)


@app.cell
def _(az, idata2):
    az.plot_posterior(idata2, var_names=["r_squared"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""Much higher r2 indicating that we are explaining more of the variation in the output than with model1."""
    )
    return


@app.cell
def _(az, idata2):
    az.plot_forest(idata2, var_names=["st_effect"], rope=(-0.1, 0.1))
    return


@app.cell
def _(az, idata2):
    az.plot_forest(idata2, var_names=["isolate_effect"], rope=(-0.1, 0.1))
    return


@app.cell
def _(az, idata2):
    az.plot_forest(idata2, var_names=["expt_effect"], rope=(-0.1, 0.1))
    return


@app.cell
def _(az, idata2):
    az.plot_forest(idata2, var_names=["model_snr"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""We see that the model SNR improves, which is great from a model's performance perspective, but that is also not exactly confidence-building for inferential purposes. I think this has to do with the fact that the data collection effort was opportunistic rather than being a rigorously-designed experiment."""
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""More important point, however, is that we can explicitly quantify model performance as part of the model! Automatic regularization happens here. If the model SNR grows, the model's coefficients are capturing most of the variability that is observed. I love this ability with the R2D2-family of priors."""
    )
    return


if __name__ == "__main__":
    app.run()
