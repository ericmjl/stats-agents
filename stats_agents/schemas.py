"""Data schemas for stats-agents.

This module contains Pydantic models for representing laboratory experiments
and their statistical structure, inspired by Emi Tanaka's work on extracting
statistical elements from experimental design descriptions.

The models are designed to work with the R2D2 framework for Bayesian modeling
and focus on providing the information needed to construct PyMC coordinates
and dimensions.
"""

from enum import Enum
from typing import Any, Dict, List, Optional

import pandas as pd
from pydantic import BaseModel, Field, validator


class ResponseType(str, Enum):
    """Types of response variables for GLM selection."""

    GAUSSIAN = "gaussian"
    POISSON = "poisson"
    BINOMIAL = "binomial"
    NEGATIVE_BINOMIAL = "negative_binomial"
    GAMMA = "gamma"
    BETA = "beta"


class R2D2Variant(str, Enum):
    """R2D2 framework variants."""

    SHRINKAGE = "shrinkage"
    GLM = "glm"
    M2 = "m2"


class FactorType(str, Enum):
    """Types of experimental factors."""

    TREATMENT = "treatment"  # Primary effects of interest (drugs, doses, etc.)
    NUISANCE = (
        "nuisance"  # Sources of variation to control for (plate, day, operator effects)
    )
    BLOCKING = (
        "blocking"  # Factors used for blocking/stratification in experimental design
    )
    COVARIATE = "covariate"  # Continuous covariates
    REPLICATE = "replicate"  # Technical or biological replicates


class ExperimentalFactor(BaseModel):
    """Represents a single experimental factor."""

    name: str = Field(..., description="Name of the factor")
    factor_type: FactorType = Field(..., description="Type of factor")
    levels: List[str] = Field(..., description="Levels of the factor")
    description: Optional[str] = Field(None, description="Description of the factor")

    # For continuous covariates
    is_continuous: bool = Field(
        False, description="Whether this is a continuous factor"
    )
    range_min: Optional[float] = Field(
        None, description="Minimum value for continuous factors"
    )
    range_max: Optional[float] = Field(
        None, description="Maximum value for continuous factors"
    )

    @validator("levels")
    def validate_levels(cls, v, values):
        """Validate the levels of the factor."""
        if not values.get("is_continuous") and len(v) < 2:
            raise ValueError("Categorical factors must have at least 2 levels")
        return v


class ReplicateStructure(BaseModel):
    """Describes how replicates are nested in the experimental design."""

    replicate_type: str = Field(
        ..., description="Type of replicate (e.g., 'technical', 'biological')"
    )
    replicates_per_unit: int = Field(
        ..., description="Number of replicates per experimental unit"
    )
    nested_under: Optional[str] = Field(
        None,
        description="What the replicates are nested under (e.g., 'plate', 'mouse')",
    )
    description: Optional[str] = Field(
        None, description="Description of the replicate structure"
    )


class ExperimentDescription(BaseModel):
    """
    Basic experiment description focused on PyMC model construction.

    This captures the core elements needed to construct PyMC coordinates
    and dimensions for R2D2 models.
    """

    # Core elements for PyMC model construction
    response: str = Field(..., description="Response variable name")
    response_type: ResponseType = Field(
        ..., description="Type of response for GLM selection"
    )

    # Experimental factors organized by type
    factors: List[ExperimentalFactor] = Field(
        default_factory=list, description="All experimental factors"
    )

    # Variables that become PyMC coordinates/dimensions
    units: Optional[str] = Field(
        None, description="Experimental units (e.g., 'mice', 'plots')"
    )
    timepoints: Optional[List[str]] = Field(
        None, description="Timepoints for longitudinal data"
    )

    # Replicate structure
    replicate_structure: Optional[ReplicateStructure] = Field(
        None, description="How replicates are structured in the experiment"
    )

    # Additional context
    aim: Optional[str] = Field(None, description="Primary aim of the experiment")
    description: Optional[str] = Field(
        None, description="Original experiment description"
    )

    def get_treatments(self) -> Dict[str, List[str]]:
        """Get treatment factors and their levels."""
        treatments = {}
        for factor in self.factors:
            if factor.factor_type == FactorType.TREATMENT:
                treatments[factor.name] = factor.levels
        return treatments

    def get_nuisance_factors(self) -> Dict[str, List[str]]:
        """Get nuisance factors and their levels."""
        nuisance_factors = {}
        for factor in self.factors:
            if factor.factor_type == FactorType.NUISANCE:
                nuisance_factors[factor.name] = factor.levels
        return nuisance_factors

    def get_blocking_factors(self) -> Dict[str, List[str]]:
        """Get blocking factors and their levels."""
        blocking_factors = {}
        for factor in self.factors:
            if factor.factor_type == FactorType.BLOCKING:
                blocking_factors[factor.name] = factor.levels
        return blocking_factors

    def get_covariates(self) -> List[str]:
        """Get continuous covariate names."""
        return [
            factor.name
            for factor in self.factors
            if factor.factor_type == FactorType.COVARIATE
        ]

    def get_replicate_factors(self) -> Dict[str, List[str]]:
        """Get replicate factors and their levels."""
        replicate_factors = {}
        for factor in self.factors:
            if factor.factor_type == FactorType.REPLICATE:
                replicate_factors[factor.name] = factor.levels
        return replicate_factors

    def get_pymc_coords(self) -> Dict[str, Any]:
        """
        Generate PyMC coordinates dictionary.

        Returns:
            Dict mapping coordinate names to coordinate values
        """
        coords = {
            "obs": None  # Will be set to range(len(data)) when data is available
        }

        # Add all categorical factors as coordinates
        for factor in self.factors:
            if not factor.is_continuous:
                coords[factor.name] = factor.levels

        # Add unit coordinates
        if self.units:
            coords[self.units] = None  # Will be set to unique values from data

        # Add time coordinates
        if self.timepoints:
            coords["timepoint"] = self.timepoints

        return coords

    def get_pymc_dims(self) -> Dict[str, List[str]]:
        """
        Generate PyMC dimensions mapping.

        Returns:
            Dict mapping variable names to their dimension lists
        """
        dims = {}

        # Response variable dimensions
        dims[self.response] = ["obs"]

        # Treatment effect dimensions
        for factor in self.factors:
            if factor.factor_type == FactorType.TREATMENT:
                dims[f"{factor.name}_effect"] = [factor.name]

        # Nuisance factor dimensions (plate effects, day effects, operator effects)
        for factor in self.factors:
            if factor.factor_type == FactorType.NUISANCE:
                dims[f"{factor.name}_effect"] = [factor.name]

        # Blocking factor dimensions (experimental design stratification)
        for factor in self.factors:
            if factor.factor_type == FactorType.BLOCKING:
                dims[f"{factor.name}_effect"] = [factor.name]

        # Unit intercept dimensions
        if self.units:
            dims[f"{self.units}_intercepts"] = [self.units]

        # Time effect dimensions
        if self.timepoints:
            dims["timepoint_effect"] = ["timepoint"]

        return dims

    def get_variance_components(self) -> Dict[str, Any]:
        """
        Get variance component structure for R2D2 models.

        Returns:
            Dict with variance component information
        """
        components = []
        component_names = []

        # Add treatment components
        for factor in self.factors:
            if factor.factor_type == FactorType.TREATMENT:
                components.append(
                    {
                        "name": factor.name,
                        "type": "treatment",
                        "dim": factor.name,
                        "levels": factor.levels,
                    }
                )
                component_names.append(factor.name)

        # Add nuisance components (plate, day, operator effects)
        for factor in self.factors:
            if factor.factor_type == FactorType.NUISANCE:
                components.append(
                    {
                        "name": factor.name,
                        "type": "nuisance",  # Sources of variation to control for
                        "dim": factor.name,
                        "levels": factor.levels,
                    }
                )
                component_names.append(factor.name)

        # Add blocking components (experimental design stratification)
        for factor in self.factors:
            if factor.factor_type == FactorType.BLOCKING:
                components.append(
                    {
                        "name": factor.name,
                        "type": "blocking",  # Experimental design blocks/strata
                        "dim": factor.name,
                        "levels": factor.levels,
                    }
                )
                component_names.append(factor.name)

        # Add unit components
        if self.units:
            components.append(
                {
                    "name": f"{self.units}_intercepts",
                    "type": "unit",
                    "dim": self.units,
                    "levels": None,  # Will be determined from data
                }
            )
            component_names.append(f"{self.units}_intercepts")

        # Add time components
        if self.timepoints:
            components.append(
                {
                    "name": "timepoint_effect",
                    "type": "time",
                    "dim": "timepoint",
                    "levels": self.timepoints,
                }
            )
            component_names.append("timepoint_effect")

        return {
            "n_components": len(components),
            "components": components,
            "component_names": component_names,
        }


class ExperimentSummary(BaseModel):
    """Summary of experiment analysis."""

    experiment: ExperimentDescription = Field(
        ..., description="The experiment description"
    )
    r2d2_variant: R2D2Variant = Field(..., description="Selected R2D2 variant")
    pymc_coords: Dict[str, Any] = Field(..., description="PyMC coordinates")
    pymc_dims: Dict[str, List[str]] = Field(..., description="PyMC dimensions")
    variance_components: Dict[str, Any] = Field(
        ..., description="Variance component structure"
    )
    recommendations: Optional[List[str]] = Field(
        None, description="Modeling recommendations"
    )


# Utility functions


def parse_experiment_from_text(description: str) -> ExperimentDescription:
    """
    Parse experiment description using simple heuristics.

    This is a simplified version that can be enhanced with LLM extraction
    following Emi Tanaka's approach.
    """
    # Basic keyword extraction
    description_lower = description.lower()

    # Extract response variable
    response = "response"  # default
    if "blood pressure" in description_lower:
        response = "blood_pressure"
    elif "gene expression" in description_lower:
        response = "gene_expression"
    elif "count" in description_lower or "colony" in description_lower:
        response = "count"

    # Extract factors
    factors = []

    # Treatment factors
    if "drug" in description_lower or "treatment" in description_lower:
        factors.append(
            ExperimentalFactor(
                name="treatment",
                factor_type=FactorType.TREATMENT,
                levels=["control", "treatment"],
            )
        )

    if "dose" in description_lower:
        factors.append(
            ExperimentalFactor(
                name="dose",
                factor_type=FactorType.TREATMENT,
                levels=["low", "medium", "high"],
            )
        )

    # Nuisance factors (plate, day, operator effects)
    if "plate" in description_lower:
        factors.append(
            ExperimentalFactor(
                name="plate",
                factor_type=FactorType.NUISANCE,  # Source of variation to control for
                levels=["plate_1", "plate_2", "plate_3", "plate_4"],
            )
        )

    if "day" in description_lower:
        factors.append(
            ExperimentalFactor(
                name="day",
                factor_type=FactorType.NUISANCE,  # Source of variation to control for
                levels=["day_1", "day_2", "day_3"],
            )
        )

    if "operator" in description_lower or "technician" in description_lower:
        factors.append(
            ExperimentalFactor(
                name="operator",
                factor_type=FactorType.NUISANCE,  # Source of variation to control for
                levels=["operator_1", "operator_2"],
            )
        )

    # Blocking factors (experimental design stratification)
    if "block" in description_lower:
        factors.append(
            ExperimentalFactor(
                name="block",
                factor_type=FactorType.BLOCKING,  # Experimental design blocks/strata
                levels=["block_1", "block_2", "block_3"],
            )
        )

    # Extract units
    units = None
    if "mouse" in description_lower or "mice" in description_lower:
        units = "mice"
    elif "patient" in description_lower:
        units = "patients"
    elif "plot" in description_lower:
        units = "plots"

    # Extract timepoints
    timepoints = None
    if "baseline" in description_lower and "post" in description_lower:
        timepoints = ["baseline", "post_treatment"]
    elif "time" in description_lower and "point" in description_lower:
        timepoints = ["t1", "t2", "t3"]  # Generic timepoints

    # Determine response type
    response_type = ResponseType.GAUSSIAN
    if "count" in description_lower or "colony" in description_lower:
        response_type = ResponseType.POISSON
    elif "binary" in description_lower or "success" in description_lower:
        response_type = ResponseType.BINOMIAL

    return ExperimentDescription(
        response=response,
        response_type=response_type,
        factors=factors,
        units=units,
        timepoints=timepoints,
        aim=None,
        description=description,
    )


def create_sample_dataframe(
    experiment: ExperimentDescription, n_samples: int = 100
) -> pd.DataFrame:
    """Create a sample DataFrame based on the experiment description with proper replicate structure."""  # noqa: E501
    import numpy as np

    # Calculate total observations considering replicates
    if experiment.replicate_structure:
        replicates_per_unit = experiment.replicate_structure.replicates_per_unit
        n_units = n_samples // replicates_per_unit
        if n_units == 0:
            n_units = 1
        total_obs = n_units * replicates_per_unit
    else:
        total_obs = n_samples

    data = {}

    # Generate experimental unit IDs with proper replication
    unit_name = experiment.units or "subject"
    if experiment.replicate_structure:
        # Create unit IDs that repeat for each replicate
        unit_ids = []
        for i in range(n_units):
            unit_ids.extend([f"{unit_name}_{i + 1}"] * replicates_per_unit)
        data[unit_name] = unit_ids
    else:
        data[unit_name] = [f"{unit_name}_{i + 1}" for i in range(total_obs)]

    # Generate replicate IDs if replicates are specified
    if experiment.replicate_structure:
        replicate_ids = []
        for i in range(n_units):
            replicate_ids.extend(range(1, replicates_per_unit + 1))
        data["replicate"] = replicate_ids

    # Generate factor variables
    for factor in experiment.factors:
        if factor.is_continuous:
            min_val = factor.range_min or 0
            max_val = factor.range_max or 100
            if experiment.replicate_structure:
                # For continuous factors, generate one value per unit, then repeat for replicates # noqa: E501
                unit_values = np.random.uniform(min_val, max_val, n_units)
                data[factor.name] = np.repeat(unit_values, replicates_per_unit)
            else:
                data[factor.name] = np.random.uniform(min_val, max_val, total_obs)
        else:
            if experiment.replicate_structure:
                # For categorical factors, generate one value per unit, then repeat for replicates # noqa: E501
                unit_values = np.random.choice(factor.levels, n_units)
                data[factor.name] = np.repeat(unit_values, replicates_per_unit)
            else:
                data[factor.name] = np.random.choice(factor.levels, total_obs)

    # Generate timepoints if present
    if experiment.timepoints:
        if experiment.replicate_structure:
            # Generate timepoints per unit, then repeat for replicates
            unit_timepoints = np.random.choice(experiment.timepoints, n_units)
            data["timepoint"] = np.repeat(unit_timepoints, replicates_per_unit)
        else:
            data["timepoint"] = np.random.choice(experiment.timepoints, total_obs)

    # Generate response variable with realistic variation
    if experiment.response_type == ResponseType.POISSON:
        # Add some structure to make it more realistic
        base_rate = 5
        if experiment.replicate_structure:
            # Add unit-level variation plus replicate-level noise
            unit_rates = np.random.gamma(2, base_rate / 2, n_units)
            replicate_noise = np.random.normal(0, 0.5, total_obs)
            rates = np.repeat(unit_rates, replicates_per_unit) + replicate_noise
            rates = np.maximum(rates, 0.1)  # Ensure positive rates
            data[experiment.response] = np.random.poisson(rates)
        else:
            data[experiment.response] = np.random.poisson(base_rate, total_obs)
    elif experiment.response_type == ResponseType.BINOMIAL:
        if experiment.replicate_structure:
            # Add unit-level variation
            unit_probs = np.random.beta(2, 2, n_units)
            data[experiment.response] = np.random.binomial(
                1, np.repeat(unit_probs, replicates_per_unit)
            )
        else:
            data[experiment.response] = np.random.choice([0, 1], total_obs)
    else:
        # Gaussian response with realistic structure
        if experiment.replicate_structure:
            # Add unit-level variation plus replicate-level noise
            unit_means = np.random.normal(0, 2, n_units)
            replicate_noise = np.random.normal(0, 1, total_obs)
            data[experiment.response] = (
                np.repeat(unit_means, replicates_per_unit) + replicate_noise
            )
        else:
            data[experiment.response] = np.random.normal(0, 1, total_obs)

    return pd.DataFrame(data)


def validate_experiment_data(
    df: pd.DataFrame, experiment: ExperimentDescription
) -> Dict[str, Any]:
    """Validate that a DataFrame matches the experiment description."""
    validation_results = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "missing_columns": [],
        "extra_columns": [],
    }

    # Check required columns
    required_columns = set()

    # Experimental units
    if experiment.units:
        required_columns.add(experiment.units)

    # Factor variables
    for factor in experiment.factors:
        required_columns.add(factor.name)

    # Timepoints
    if experiment.timepoints:
        required_columns.add("timepoint")

    # Response variable
    required_columns.add(experiment.response)

    # Check for missing columns
    df_columns = set(df.columns)
    missing = required_columns - df_columns
    extra = df_columns - required_columns

    if missing:
        validation_results["valid"] = False
        validation_results["missing_columns"] = list(missing)
        validation_results["errors"].append(f"Missing required columns: {missing}")

    if extra:
        validation_results["extra_columns"] = list(extra)
        validation_results["warnings"].append(f"Extra columns found: {extra}")

    return validation_results
