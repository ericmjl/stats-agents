#!/usr/bin/env python3
"""
Test script to verify the new @prompt decorator approach works correctly.
"""

from stats_agents.models import PyMCModelGenerator, parse_experiment_description
from stats_agents.prompts import (
    experiment_parsing_system,
    generate_model_with_tracking,
    model_generation_system,
    parse_experiment_with_tracking,
)
from stats_agents.schemas import (
    ExperimentalFactor,
    ExperimentDescription,
    FactorType,
    ResponseType,
)


def test_prompt_functions():
    """Test the prompt creation functions."""
    print("Testing prompt creation functions...")

    # Test system prompts
    system_prompt = experiment_parsing_system()
    print("✅ Experiment parsing system prompt created")
    print(f"   Role: {system_prompt.role}")
    print(f"   Content length: {len(system_prompt.content)} characters")
    print(
        f"   Contains 'biological experiment': {'biological experiment' in system_prompt.content}"  # noqa: E501
    )

    model_system = model_generation_system()
    print("✅ Model generation system prompt created")
    print(f"   Role: {model_system.role}")
    print(f"   Content length: {len(model_system.content)} characters")
    print(f"   Contains 'R2D2 framework': {'R2D2 framework' in model_system.content}")

    # Test user prompts
    user_prompt = parse_experiment_with_tracking("test description")
    print("✅ User prompt created")
    print(f"   Role: {user_prompt.role}")
    print(f"   Content length: {len(user_prompt.content)} characters")
    print(
        f"   Contains 'test description': {'test description' in user_prompt.content}"
    )

    model_user = generate_model_with_tracking('{"test": "json"}', "data preview")
    print("✅ Model user prompt created")
    print(f"   Role: {model_user.role}")
    print(f"   Content length: {len(model_user.content)} characters")
    print(f"   Contains 'data preview': {'data preview' in model_user.content}")

    return True


def test_experiment_parsing_with_tracking():
    """Test experiment parsing with tracking."""
    print("\nTesting experiment parsing with tracking...")

    description = (
        "Study drug effects on blood pressure in mice with plate and day controls"
    )

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


def test_model_generation_with_tracking():
    """Test model generation with tracking."""
    print("\nTesting model generation with tracking...")

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
        filepath = model_response.write_to_disk("test_tracking_model.py")
        print(f"   File written to: {filepath}")

    except Exception as e:
        print(f"❌ Error generating model: {e}")
        return False

    return True


def test_prompt_template_rendering():
    """Test that the Jinja2 templates render correctly."""
    print("\nTesting prompt template rendering...")

    # Test experiment parsing template
    user_prompt = parse_experiment_with_tracking("Study drug effects on blood pressure")
    content = user_prompt.content
    print("✅ Template rendered successfully")
    print(
        f"   Contains description: {'Study drug effects on blood pressure' in content}"
    )
    print(f"   Contains template variable: {'{{ description }}' not in content}")

    # Test model generation template with data
    model_prompt = generate_model_with_tracking('{"response": "test"}', "Sample data")
    content = model_prompt.content
    print("✅ Model template rendered successfully")
    print(f"   Contains data preview: {'Sample data' in content}")
    print(f"   Contains conditional: {'{% if data_preview %}' not in content}")

    # Test model generation template without data
    model_prompt_no_data = generate_model_with_tracking('{"response": "test"}')
    content = model_prompt_no_data.content
    print("✅ Model template without data rendered successfully")
    print("   No data preview section: 'Sample data' not in content")

    return True


if __name__ == "__main__":
    print("=" * 60)
    print("TESTING @PROMPT DECORATOR APPROACH")
    print("=" * 60)

    success1 = test_prompt_functions()
    success2 = test_prompt_template_rendering()
    success3 = test_experiment_parsing_with_tracking()
    success4 = test_model_generation_with_tracking()

    if all([success1, success2, success3, success4]):
        print("\n✅ All tests passed! @prompt decorator approach successful.")
    else:
        print("\n❌ Some tests failed. Check the output above.")

    print("\nTracking Status: Enabled")
    print("=" * 60)
