# Interactive Insight Simulator

The Interactive Insight Simulator is a powerful component of the Wearable Data Insight Generator that allows you to simulate wearable data scenarios and generate personalized LLM insights without needing real-time device data.

## Overview

This simulator enables you to:

1. Generate synthetic time-series data for various wearable metrics (HRV, sleep, activity, stress)
2. Apply predefined scenario patterns (overtraining, recovery, sleep deprivation, etc.)
3. Personalize simulations based on user profiles and goals
4. Generate LLM-based coaching insights for the simulated data
5. Visualize the simulation results with annotated insights

## Key Components

The simulator consists of two main modules:

1. **`simulator_engine.py`**: The core simulation engine that generates scenarios and insights
2. **`demo_data_generator.py`**: Utility for generating realistic synthetic wearable data

## Predefined Scenarios

The simulator includes several predefined scenarios:

- **Training Peak Day**: High intensity training with elevated heart rate and strain
- **Sleep Deprivation**: Period with reduced sleep quality and quantity
- **Overtraining Week**: Extended period of high training load without adequate recovery
- **Recovery Phase**: Period of active recovery with reduced training load
- **Circadian Disruption**: Disruption to normal sleep-wake patterns (e.g., travel, shift work)

## Usage Examples

### Basic Usage

```python
from src.simulator_engine import SimulatorEngine
from src.feature_engineer import FeatureEngineer
from src.llm_engine import LLMEngine

# Initialize components
feature_engineer = FeatureEngineer()
llm_engine = LLMEngine()

# Create a simulator instance
simulator = SimulatorEngine(
    feature_engineer=feature_engineer,
    llm_engine=llm_engine
)

# Set simulation parameters
simulator.set_simulation_parameters("overtraining", days=7)

# Run the simulation
sim_data, insights = simulator.run_simulation()

# Visualize the results
simulator.visualize_simulation("outputs/overtraining_simulation.png")

# Export the results
simulator.export_results("outputs/overtraining_results.json")
```

### Custom User Profile

```python
# Define a custom user profile
user_profile = {
    "user_id": "custom_user_001",
    "name": "Custom User",
    "age": 35,
    "training_personality": "aggressive",
    "goals": {
        "primary": "improve_endurance",
        "secondary": ["reduce_stress", "injury_prevention"]
    },
    "baseline_metrics": {
        "resting_hr": 58,
        "hrv_rmssd": 72,
        "sleep_hours": 7.2,
        "recovery_score": 80
    }
}

# Create a simulator with the custom profile
simulator = SimulatorEngine(
    feature_engineer=feature_engineer,
    llm_engine=llm_engine,
    user_profile=user_profile
)
```

### Generating a Demo Dataset

```python
from src.demo_data_generator import DemoDataGenerator

# Create a data generator
generator = DemoDataGenerator()

# Generate a complete demo dataset
profile, data, path = generator.generate_demo_dataset(
    profile_type="athlete",  # Options: athlete, casual, recovery_focused, random
    days=14,
    scenario="overtraining"  # Optional scenario to apply
)
```

## Integration with User Profiles

The simulator integrates with the user profile system to personalize simulations:

- **Goals**: Influences the type and content of insights generated
- **Training Personality**: Affects the simulation patterns (aggressive trainers show more variability)
- **Baseline Metrics**: Used to establish realistic baseline values for the user

## Visualization

The simulator includes visualization capabilities to help understand the simulated data:

- Time-series plots of key metrics (HRV, sleep, recovery, strain)
- Markers for generated insights
- Scenario comparison visualizations

## Example Scripts

Check out the example scripts in the `examples/` directory:

- `simulator_example.py`: Demonstrates basic and advanced simulator usage
- `scenario_comparison.py`: Shows how to compare different scenarios

## Adding Custom Scenarios

You can create custom scenarios by:

1. Adding a new scenario type to the `SCENARIO_TYPES` dictionary in `simulator_engine.py`
2. Implementing the scenario logic in the `apply_scenario` method
3. Creating a scenario template JSON file in `data/scenario_templates/`

## Output Format

The simulation results are exported as a JSON file with the following structure:

```json
{
  "scenario": "overtraining",
  "days": 7,
  "user_profile": { ... },
  "simulation_data": [ ... ],
  "insights": [ ... ],
  "metadata": {
    "generated_at": "2025-03-28T19:45:00.000Z",
    "version": "1.0.0"
  }
}
```

## Why This Feature Matters

- Enables testing and demonstration without real device data
- Allows exploring how the system responds to different physiological patterns
- Provides a powerful tool for developing and refining the insight generation logic
- Demonstrates the system's ability to generate personalized insights based on user profiles
