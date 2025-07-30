#!/usr/bin/env python3
"""
Test script to verify the refactored parsing function works correctly.
"""

from stats_agents.models import PyMCModelGenerator, parse_experiment_description
from stats_agents.schemas import (
    ExperimentalFactor,
    ExperimentDescription,
    FactorType,
    ResponseType,
)


def test_parsing_function():
    """Test the new parse_experiment_description function."""
    print("Testing parse_experiment_description function...")

    # Test with a simple description
    description = "Study drug effects on blood pressure in mice with plate controls"

    try:
        experiment = parse_experiment_description(description)
        print(f"✅ Successfully parsed experiment: {experiment.response}")
        print(f"   Response type: {experiment.response_type}")
        print(f"   Units: {experiment.units}")
        print(f"   Factors: {len(experiment.factors)}")

        for factor in experiment.factors:
            print(f"     - {factor.name} ({factor.factor_type.value}): {factor.levels}")

    except Exception as e:
        print(f"❌ Error parsing experiment: {e}")
        return False

    return True


def test_model_generation():
    """Test the PyMCModelGenerator with the new PyMCModelResponse."""
    print("\nTesting PyMCModelGenerator...")

    # Create a simple experiment manually
    experiment = ExperimentDescription(
        response="blood_pressure",
        response_type=ResponseType.GAUSSIAN,
        factors=[
            ExperimentalFactor(
                name="treatment",
                factor_type=FactorType.TREATMENT,
                levels=["control", "drug_a", "drug_b"],
            ),
            ExperimentalFactor(
                name="plate",
                factor_type=FactorType.NUISANCE,
                levels=["plate_1", "plate_2", "plate_3"],
            ),
        ],
        units="mice",
        aim="Test experiment",
    )

    try:
        generator = PyMCModelGenerator(model_name="gpt-4o")
        model_response = generator.generate_model_code(experiment)

        print("✅ Successfully generated model response")
        print(f"   Python version: {model_response.python_version}")
        print(f"   Dependencies: {model_response.dependencies}")
        print(f"   Code length: {len(model_response.model_code)} characters")

        # Test rendering
        rendered = model_response.render_code()
        print(f"   Rendered code length: {len(rendered)} characters")
        print(f"   PEP 723 header present: {'# /// script' in rendered}")

        # Test writing to disk
        filepath = model_response.write_to_disk("test_model.py")
        print(f"   File written to: {filepath}")

    except Exception as e:
        print(f"❌ Error generating model: {e}")
        return False

    return True


if __name__ == "__main__":
    print("=" * 60)
    print("TESTING REFACTORED PARSING AND MODEL GENERATION")
    print("=" * 60)

    success1 = test_parsing_function()
    success2 = test_model_generation()

    if success1 and success2:
        print("\n✅ All tests passed! Refactoring successful.")
    else:
        print("\n❌ Some tests failed. Check the output above.")
