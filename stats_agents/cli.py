"""Command-line interface for stats-agents."""

from pathlib import Path
from typing import Optional

import typer

from stats_agents.models import (
    ExperimentAnalyzer,
    ExperimentDataGenerator,
    PyMCModelGenerator,
    parse_experiment_description,
)

app = typer.Typer()


@app.command()
def parse_experiment(
    description: str = typer.Argument(..., help="Freeform experiment description"),
    output_file: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output file for model code"
    ),
    generate_data: bool = typer.Option(
        False, "--data", "-d", help="Generate sample data"
    ),
    n_samples: int = typer.Option(
        100, "--samples", "-n", help="Number of samples to generate"
    ),
    model_name: str = typer.Option(
        "gpt-4o", "--model", "-m", help="LLM model to use for code generation"
    ),
):
    """Parse experiment description and generate PyMC model code."""

    # Parse the experiment description
    experiment = parse_experiment_description(description)

    # Display experiment summary
    typer.echo("📊 Experiment Summary:")
    typer.echo(f"  Response: {experiment.response}")
    typer.echo(f"  Response Type: {experiment.response_type}")
    typer.echo(f"  Units: {experiment.units}")

    # Show factors
    if experiment.factors:
        typer.echo("\n🔬 Experimental Factors:")
        for factor in experiment.factors:
            factor_type_desc = factor.factor_type.value.replace("_", " ").title()
            typer.echo(f"  - {factor.name} ({factor_type_desc}): {factor.levels}")

    # Show timepoints if present
    if experiment.timepoints:
        typer.echo(f"\n⏰ Timepoints: {experiment.timepoints}")

    # Generate sample data if requested
    data = None
    if generate_data:
        typer.echo("\n📈 Generating sample data...")
        data_generator = ExperimentDataGenerator()
        data, csv_path, xarray_path = data_generator.generate_sample_data(
            experiment, n_samples
        )
        typer.echo(f"  Generated {len(data)} samples")
        typer.echo(f"  CSV saved to: {csv_path}")
        typer.echo(f"  XArray saved to: {xarray_path}")

    # Generate model code
    typer.echo("\n🔧 Generating PyMC model code...")
    model_generator = PyMCModelGenerator(model_name=model_name)
    model_response = model_generator.generate_model_code(experiment, data)

    # Display model code
    typer.echo("\n📝 Generated PyMC Model Code:")
    typer.echo("=" * 60)
    typer.echo(model_response.render_code())
    typer.echo("=" * 60)

    # Save to file if requested
    if output_file:
        filepath = model_response.write_to_disk(str(output_file))
        typer.echo(f"\n💾 Model code saved to: {filepath}")
    else:
        # Save to temporary file for reference
        temp_filepath = model_response.write_to_disk()
        typer.echo(f"\n💾 Model code saved to temporary file: {temp_filepath}")

    # Display metadata
    typer.echo("\n📋 Model Metadata:")
    typer.echo(f"  Python Version: {model_response.python_version}")
    typer.echo(f"  Dependencies: {', '.join(model_response.dependencies)}")

    # Analyze the experiment
    typer.echo("\n🔍 Experiment Analysis:")
    analyzer = ExperimentAnalyzer()
    analysis = analyzer.analyze_experiment(experiment)

    typer.echo(
        f"  Model Complexity: {analysis['r2d2_recommendations']['model_complexity']}"
    )
    typer.echo(f"  Potential Issues: {len(analysis['potential_issues'])}")

    if analysis["potential_issues"]:
        typer.echo("  Issues to consider:")
        for issue in analysis["potential_issues"]:
            typer.echo(f"    - {issue}")


@app.command()
def analyze_experiment(
    description: str = typer.Argument(..., help="Freeform experiment description"),
):
    """Analyze experiment structure and provide recommendations."""

    # Parse the experiment description
    experiment = parse_experiment_description(description)

    # Analyze the experiment
    analyzer = ExperimentAnalyzer()
    analysis = analyzer.analyze_experiment(experiment)

    # Display detailed analysis
    typer.echo("🔍 Experiment Analysis:")
    typer.echo("=" * 50)

    # Experiment summary
    summary = analysis["experiment_summary"]
    typer.echo(f"Response: {summary['response']} ({summary['response_type']})")
    typer.echo(f"Units: {summary['units']}")

    typer.echo(
        f"Model Complexity: {analysis['r2d2_recommendations']['model_complexity']}"
    )

    # Factor breakdown
    typer.echo("\nFactors:")
    typer.echo(f"  Treatments: {summary['n_treatments']}")
    typer.echo(f"  Nuisance: {summary['n_nuisance_factors']}")
    typer.echo(f"  Blocking: {summary['n_blocking_factors']}")
    typer.echo(f"  Covariates: {summary['n_covariates']}")

    # PyMC coordinates and dimensions
    typer.echo(f"\nPyMC Coordinates: {len(analysis['pymc_coords'])}")
    for name, values in analysis["pymc_coords"].items():
        if values is None:
            typer.echo(f"  {name}: (dynamic)")
        else:
            typer.echo(f"  {name}: {len(values)} values")

    typer.echo(f"\nPyMC Dimensions: {len(analysis['pymc_dims'])}")
    for var_name, dim_list in analysis["pymc_dims"].items():
        typer.echo(f"  {var_name}: {dim_list}")

    # Variance components
    variance_components = analysis["variance_components"]
    typer.echo(f"\nVariance Components: {variance_components['n_components']}")
    for component in variance_components["components"]:
        typer.echo(f"  - {component['name']} ({component['type']}): {component['dim']}")

    # Recommendations
    recommendations = analysis["r2d2_recommendations"]
    typer.echo("\nPrior Suggestions:")
    for key, value in recommendations["prior_suggestions"].items():
        typer.echo(f"  {key}: {value}")

    # Potential issues
    if analysis["potential_issues"]:
        typer.echo("\n⚠️  Potential Issues:")
        for issue in analysis["potential_issues"]:
            typer.echo(f"  - {issue}")


@app.command()
def generate_data(
    description: str = typer.Argument(..., help="Freeform experiment description"),
    n_samples: int = typer.Option(
        100, "--samples", "-n", help="Number of samples to generate"
    ),
    output_dir: Optional[Path] = typer.Option(
        None, "--output-dir", "-o", help="Output directory for data files"
    ),
):
    """Generate sample data for an experiment."""

    # Parse the experiment description
    experiment = parse_experiment_description(description)

    # Generate sample data
    typer.echo("📈 Generating sample data...")
    data_generator = ExperimentDataGenerator()
    data, csv_path, xarray_path = data_generator.generate_sample_data(
        experiment, n_samples
    )

    # Display data summary
    typer.echo(f"✅ Generated {len(data)} samples")
    typer.echo(f"Data shape: {data.shape}")
    typer.echo(f"Columns: {list(data.columns)}")

    # Show data preview
    typer.echo("\n📊 Data Preview:")
    typer.echo(data.head().to_string())

    # Show file locations
    typer.echo("\n💾 Data Files:")
    typer.echo(f"  CSV: {csv_path}")
    typer.echo(f"  XArray: {xarray_path}")

    # Move files to output directory if specified
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        import shutil

        csv_dest = output_dir / "experiment_data.csv"
        xarray_dest = output_dir / "experiment_data.nc"

        shutil.move(csv_path, csv_dest)
        shutil.move(xarray_path, xarray_dest)

        typer.echo(f"\n📁 Files moved to: {output_dir}")
        typer.echo(f"  CSV: {csv_dest}")
        typer.echo(f"  XArray: {xarray_dest}")


if __name__ == "__main__":
    app()
