"""Tests for structured experiment parsing functionality."""

from pathlib import Path

import pandas as pd

from stats_agents.models import (
    ExperimentAnalyzer,
    ExperimentDataGenerator,
    PyMCModelGenerator,
    parse_experiment_description,
)
from stats_agents.schemas import (
    ExperimentDescription,
    R2D2Variant,
    ResponseType,
    create_sample_dataframe,
    validate_experiment_data,
)


class TestExperimentParsing:
    """Test experiment parsing functionality."""

    def test_basic_experiment_parsing(self):
        """Test basic experiment parsing from description."""
        description = "I want to compare the effect of two drugs on blood pressure in mice, measured at baseline and after treatment."  # noqa: E501

        experiment = parse_experiment_description(description)

        assert isinstance(experiment, ExperimentDescription)
        assert experiment.response == "blood_pressure"
        assert experiment.response_type == ResponseType.GAUSSIAN
        assert experiment.units == "mice"
        assert experiment.treatments is not None
        assert "treatment" in experiment.treatments
        assert experiment.suggested_r2d2_variant == R2D2Variant.M2

    def test_gene_expression_experiment(self):
        """Test gene expression experiment parsing."""
        description = "A gene expression study comparing three treatment conditions across different mouse strains."  # noqa: E501

        experiment = parse_experiment_description(description)

        assert experiment.response == "gene_expression"
        assert experiment.response_type == ResponseType.GAUSSIAN
        assert experiment.suggested_r2d2_variant == R2D2Variant.M2

    def test_count_data_experiment(self):
        """Test count data experiment parsing."""
        description = "A study measuring bacterial colony counts under different growth conditions."  # noqa: E501

        experiment = parse_experiment_description(description)

        assert experiment.response == "count"
        assert experiment.response_type == ResponseType.POISSON
        assert experiment.suggested_r2d2_variant == R2D2Variant.GLM


class TestPyMCModelResponse:
    """Test PyMCModelResponse functionality."""

    def test_pymc_model_response_creation(self):
        """Test creating a PyMCModelResponse."""
        from stats_agents.pymc_generator import PyMCModelResponse

        model_code = "import pymc as pm\n\nwith pm.Model() as model:\n    pass"
        description = "A simple PyMC model for testing purposes"
        response = PyMCModelResponse(
            model_code=model_code,
            description=description,
            python_version="3.12",
            dependencies=["pymc", "pandas", "numpy"],
        )

        assert response.model_code == model_code
        assert response.description == description
        assert response.python_version == "3.12"
        assert response.dependencies == ["pymc", "pandas", "numpy"]

    def test_pymc_model_response_render_code(self):
        """Test rendering PEP 723 compliant code."""
        from stats_agents.pymc_generator import PyMCModelResponse

        model_code = "import pymc as pm\n\nwith pm.Model() as model:\n    pass"
        description = "A simple PyMC model for testing purposes"
        response = PyMCModelResponse(
            model_code=model_code,
            description=description,
            python_version="3.12",
            dependencies=["pymc", "pandas"],
        )

        rendered = response.render_code()

        assert "# /// script" in rendered
        assert '# requires-python = ">=3.12"' in rendered
        assert '#   "pymc",' in rendered
        assert '#   "pandas",' in rendered
        assert model_code in rendered

    def test_pymc_model_response_write_to_disk(self, tmp_path):
        """Test writing code to disk."""
        from stats_agents.pymc_generator import PyMCModelResponse

        model_code = "import pymc as pm\n\nwith pm.Model() as model:\n    pass"
        description = "A simple PyMC model for testing purposes"
        response = PyMCModelResponse(
            model_code=model_code,
            description=description,
            python_version="3.12",
            dependencies=["pymc", "pandas"],
        )

        # Test writing to specified path
        filepath = tmp_path / "test_model.py"
        written_path = response.write_to_disk(str(filepath))

        assert Path(written_path).exists()
        assert Path(written_path).read_text().startswith("# /// script")
        assert model_code in Path(written_path).read_text()

        # Test writing to temporary file
        temp_path = response.write_to_disk()
        assert Path(temp_path).exists()
        assert Path(temp_path).read_text().startswith("# /// script")


class TestR2D2ModelGeneration:
    """Test R2D2 model generation."""

    def test_shrinkage_model_generation(self):
        """Test R2D2 Shrinkage model generation."""
        # Create a simple experiment
        experiment = ExperimentDescription(
            response="y",
            treatments={"treatment": ["control", "treatment"]},
            units="subjects",
            design_name="completely_randomized",
            response_type=ResponseType.GAUSSIAN,
        )

        generator = PyMCModelGenerator()
        model_response = generator.generate_model_code(experiment)

        assert "R2D2 Shrinkage Framework" in model_response.model_code
        assert "r_squared = pm.Beta" in model_response.model_code
        assert "phi = pm.Dirichlet" in model_response.model_code
        assert "treatment_effect" in model_response.model_code

    def test_m2_model_generation(self):
        """Test R2D2M2 model generation."""
        # Create a multilevel experiment
        experiment = ExperimentDescription(
            response="y",
            treatments={"treatment": ["control", "treatment"]},
            units="mice",
            design_name="randomized_block",
            response_type=ResponseType.GAUSSIAN,
        )

        generator = PyMCModelGenerator()
        model_response = generator.generate_model_code(experiment)

        assert "R2D2M2 Framework" in model_response.model_code
        assert "mice_intercepts" in model_response.model_code


class TestDataGeneration:
    """Test data generation functionality."""

    def test_sample_dataframe_generation(self):
        """Test sample DataFrame generation."""
        experiment = ExperimentDescription(
            response="response",
            treatments={"treatment": ["control", "treatment"]},
            units="subjects",
            design_name="completely_randomized",
            response_type=ResponseType.GAUSSIAN,
        )

        df = create_sample_dataframe(experiment, n_samples=50)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 50
        assert "subjects" in df.columns
        assert "treatment" in df.columns
        assert "response" in df.columns
        assert set(df["treatment"].unique()) == {"control", "treatment"}

    def test_data_generator_with_files(self):
        """Test data generator with file output."""
        experiment = ExperimentDescription(
            response="response",
            treatments={"treatment": ["control", "treatment"]},
            units="subjects",
            design_name="completely_randomized",
            response_type=ResponseType.GAUSSIAN,
        )

        generator = ExperimentDataGenerator()
        df, csv_path, xarray_path = generator.generate_sample_data(
            experiment, n_samples=25
        )

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 25
        assert Path(csv_path).exists()
        assert Path(xarray_path).exists()

        # Clean up
        Path(csv_path).unlink()
        Path(xarray_path).unlink()


class TestExperimentAnalysis:
    """Test experiment analysis functionality."""

    def test_experiment_analysis(self):
        """Test experiment analysis."""
        experiment = ExperimentDescription(
            response="response",
            treatments={"treatment": ["control", "treatment"]},
            units="subjects",
            design_name="completely_randomized",
            response_type=ResponseType.GAUSSIAN,
        )

        analyzer = ExperimentAnalyzer()
        analysis = analyzer.analyze_experiment(experiment)

        assert "experiment_summary" in analysis
        assert "r2d2_recommendations" in analysis
        assert "data_structure" in analysis
        assert "potential_issues" in analysis

        assert analysis["experiment_summary"]["response"] == "response"
        assert (
            analysis["experiment_summary"]["suggested_r2d2_variant"] == R2D2Variant.M2
        )


class TestDataValidation:
    """Test data validation functionality."""

    def test_data_validation_success(self):
        """Test successful data validation."""
        experiment = ExperimentDescription(
            response="response",
            treatments={"treatment": ["control", "treatment"]},
            units="subjects",
            design_name="completely_randomized",
            response_type=ResponseType.GAUSSIAN,
        )

        # Create valid data
        df = pd.DataFrame(
            {
                "subjects": ["subj_1", "subj_2", "subj_3"],
                "treatment": ["control", "treatment", "control"],
                "response": [1.0, 2.0, 1.5],
            }
        )

        validation = validate_experiment_data(df, experiment)

        assert validation["valid"] is True
        assert len(validation["errors"]) == 0

    def test_data_validation_failure(self):
        """Test data validation failure."""
        experiment = ExperimentDescription(
            response="response",
            treatments={"treatment": ["control", "treatment"]},
            units="subjects",
            design_name="completely_randomized",
            response_type=ResponseType.GAUSSIAN,
        )

        # Create invalid data (missing required columns)
        df = pd.DataFrame(
            {
                "subjects": ["subj_1", "subj_2", "subj_3"],
                "response": [1.0, 2.0, 1.5],
                # Missing 'treatment' column
            }
        )

        validation = validate_experiment_data(df, experiment)

        assert validation["valid"] is False
        assert len(validation["errors"]) > 0
        assert "treatment" in validation["missing_columns"]


def test_end_to_end_workflow():
    """Test complete end-to-end workflow."""
    # 1. Parse experiment description
    description = "I want to compare the effect of two drugs on blood pressure in mice, measured at baseline and after treatment. Each mouse is measured at baseline and post-treatment with 4 replicate measurements per timepoint."  # noqa: E501

    experiment = parse_experiment_description(description)

    # 2. Generate model code
    model_generator = PyMCModelGenerator()
    model_response = model_generator.generate_model_code(experiment)

    # Assertions for model response
    assert hasattr(model_response, "model_code")
    assert hasattr(model_response, "description")
    assert hasattr(model_response, "python_version")
    assert hasattr(model_response, "dependencies")
    assert model_response.python_version == "3.12"
    assert "pymc" in model_response.dependencies
    assert "pandas" in model_response.dependencies
    assert len(model_response.description) > 0

    # 3. Generate sample data
    data_generator = ExperimentDataGenerator()
    df, csv_path, xarray_path = data_generator.generate_sample_data(
        experiment, n_samples=50
    )

    # 4. Analyze experiment
    analyzer = ExperimentAnalyzer()
    analysis = analyzer.analyze_experiment(experiment)

    # 5. Validate data
    validation = validate_experiment_data(df, experiment)

    # Assertions
    assert isinstance(experiment, ExperimentDescription)
    assert experiment.suggested_r2d2_variant == R2D2Variant.M2
    assert "R2D2M2 Framework" in model_response.model_code
    assert len(df) == 50
    assert validation["valid"] is True

    # Assertions for analysis
    assert "experiment_summary" in analysis
    assert "r2d2_recommendations" in analysis
    assert "pymc_coords" in analysis
    assert "pymc_dims" in analysis
    assert "variance_components" in analysis
    assert "potential_issues" in analysis
    assert analysis["experiment_summary"]["response"] == "blood_pressure"

    # Clean up
    Path(csv_path).unlink()
    Path(xarray_path).unlink()

    print("✅ End-to-end workflow test passed!")


if __name__ == "__main__":
    # Run the end-to-end test
    test_end_to_end_workflow()
