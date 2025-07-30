"""Custom model code for stats-agents.

This module contains the core functionality for parsing experiment descriptions
and generating R2D2-based PyMC models using structured LLM agents.
"""

import tempfile
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import xarray as xr

from .experiment_parser import experiment_bot
from .pymc_generator import PyMCModelResponse, generate_pymc_model
from .schemas import (
    ExperimentDescription,
    ResponseType,
    create_sample_dataframe,
)


def parse_experiment_description(
    description: str, model_name: str = "gpt-4o-mini"
) -> ExperimentDescription:
    """
    Parse a freeform experiment description and return a structured ExperimentDescription.

    This function uses a structured LLM approach inspired by Emi Tanaka's work
    to extract statistical elements from experimental design descriptions.

    Args:
        description: Freeform text describing the experiment
        model_name: LLM model to use for parsing

    Returns:
        ExperimentDescription: Structured representation of the experiment
    """  # noqa: E501
    # Use the experiment parsing bot
    return experiment_bot(description)


class PyMCModelGenerator:
    """Structured bot for generating PyMC models from experiment descriptions."""

    def __init__(self, model_name: str = "gpt-4o"):
        self.model_name = model_name

    def generate_model_code(
        self, experiment: ExperimentDescription, data: Optional[pd.DataFrame] = None
    ) -> PyMCModelResponse:
        """
        Generate PyMC model code using a structured bot approach.

        Args:
            experiment: Structured experiment description
            data: Optional DataFrame (currently not used in prompt generation)

        Returns:
            PyMCModelResponse: Structured response containing model code and metadata
        """
        # Use the PyMC generator
        return generate_pymc_model(experiment, self.model_name)


class ExperimentDataGenerator:
    """Generator for sample experiment data."""

    def __init__(self):
        pass

    def generate_sample_data(
        self, experiment: ExperimentDescription, n_samples: int = 100
    ) -> Tuple[pd.DataFrame, str, str]:
        """
        Generate sample data for the experiment.

        Args:
            experiment: Structured experiment description
            n_samples: Number of samples to generate

        Returns:
            Tuple[pd.DataFrame, str, str]: DataFrame, CSV path, xarray path
        """
        # Generate DataFrame
        df = create_sample_dataframe(experiment, n_samples)

        # Save to CSV
        csv_path = self._save_csv(df, experiment)

        # Save to xarray
        xarray_path = self._save_xarray(df, experiment)

        return df, csv_path, xarray_path

    def _save_csv(self, df: pd.DataFrame, experiment: ExperimentDescription) -> str:
        """Save DataFrame to CSV and return path."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            df.to_csv(f.name, index=False)
            return f.name

    def _save_xarray(self, df: pd.DataFrame, experiment: ExperimentDescription) -> str:
        """Save DataFrame to xarray format and return path."""
        # Convert to xarray
        ds = xr.Dataset.from_dataframe(df)

        # Save to NetCDF
        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as f:
            ds.to_netcdf(f.name)
            return f.name


class ExperimentAnalyzer:
    """Analyzer for experiment structure and recommendations."""

    def __init__(self):
        pass

    def analyze_experiment(self, experiment: ExperimentDescription) -> Dict[str, Any]:
        """
        Analyze experiment structure and provide recommendations.

        Args:
            experiment: Structured experiment description

        Returns:
            Dict[str, Any]: Analysis results and recommendations
        """
        # Get PyMC-specific information
        pymc_coords = experiment.get_pymc_coords()
        pymc_dims = experiment.get_pymc_dims()
        variance_components = experiment.get_variance_components()

        analysis = {
            "experiment_summary": self._summarize_experiment(experiment),
            "r2d2_recommendations": self._get_r2d2_recommendations(experiment),
            "pymc_coords": pymc_coords,
            "pymc_dims": pymc_dims,
            "variance_components": variance_components,
            "potential_issues": self._identify_potential_issues(experiment),
        }

        return analysis

    def _summarize_experiment(
        self, experiment: ExperimentDescription
    ) -> Dict[str, Any]:
        """Summarize the experiment structure."""
        treatments = experiment.get_treatments()
        nuisance_factors = experiment.get_nuisance_factors()
        blocking_factors = experiment.get_blocking_factors()
        covariates = experiment.get_covariates()

        return {
            "response": experiment.response,
            "response_type": experiment.response_type,
            "n_treatments": len(treatments),
            "n_nuisance_factors": len(nuisance_factors),
            "n_blocking_factors": len(blocking_factors),
            "n_covariates": len(covariates),
            "units": experiment.units,
            "timepoints": experiment.timepoints,
        }

    def _get_r2d2_recommendations(
        self, experiment: ExperimentDescription
    ) -> Dict[str, Any]:
        """Get R2D2-specific recommendations."""
        recommendations = {
            "prior_suggestions": self._get_prior_suggestions(experiment),
            "model_complexity": self._assess_model_complexity(experiment),
        }

        return recommendations

    def _get_prior_suggestions(
        self, experiment: ExperimentDescription
    ) -> Dict[str, Any]:
        """Get suggestions for prior specifications."""
        suggestions = {
            "r2_prior": {"alpha": 1, "beta": 1},
            "concentration_prior": 0.5,
            "residual_prior": {"nu": 3},
        }

        # Adjust based on experiment characteristics
        if experiment.response_type != ResponseType.GAUSSIAN:
            suggestions["glm_specific"] = (
                "Consider GLM-specific effective noise calculation"
            )

        # Check for nuisance factors
        nuisance_factors = experiment.get_nuisance_factors()
        if nuisance_factors:
            suggestions["nuisance_factors"] = (
                f"Found {len(nuisance_factors)} nuisance factors: {list(nuisance_factors.keys())}"  # noqa: E501
            )

        return suggestions

    def _assess_model_complexity(self, experiment: ExperimentDescription) -> str:
        """Assess the complexity of the model."""
        n_factors = len(experiment.factors)
        has_units = experiment.units is not None
        has_time = experiment.timepoints is not None

        total_components = n_factors + (1 if has_units else 0) + (1 if has_time else 0)

        if total_components <= 2:
            return "simple"
        elif total_components <= 4:
            return "moderate"
        else:
            return "complex"

    def _identify_potential_issues(
        self, experiment: ExperimentDescription
    ) -> List[str]:
        """Identify potential issues with the experiment design."""
        issues = []

        # Check for missing information
        treatments = experiment.get_treatments()
        if not treatments:
            issues.append("No treatment variables specified")

        if not experiment.units:
            issues.append("No experimental units specified")

        # Check for GLM-specific issues
        if experiment.response_type != ResponseType.GAUSSIAN:
            issues.append(
                f"Non-Gaussian response ({experiment.response_type}) - consider GLM-specific handling"  # noqa: E501
            )

        # Check for nuisance factors
        nuisance_factors = experiment.get_nuisance_factors()
        if nuisance_factors:
            issues.append(
                f"Found nuisance factors: {list(nuisance_factors.keys())} - ensure these are properly controlled for"  # noqa: E501
            )

        return issues
