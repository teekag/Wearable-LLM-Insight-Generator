# Time-Series Insight Timeline

The Time-Series Insight Timeline is a comprehensive visualization component for the Wearable LLM Insight Generator that enables users to visualize their wearable data trends alongside AI-generated insights.

## Overview

This component provides both static and interactive visualizations of wearable metrics (such as HRV, sleep, recovery, and strain) with insight annotations overlaid at relevant points in time. The implementation is modular, consisting of three main components:

1. **Core Timeline Visualizer** - Creates static visualizations using Matplotlib
2. **Interactive Timeline** - Provides web-based interactive visualizations using Plotly
3. **Timeline Integration** - Connects the visualization components with the insight engine and simulator

## Features

- **Static Timeline Visualization**: Generate static timeline plots of multiple metrics with insight annotations
- **Interactive Timeline Visualization**: Create interactive web-based visualizations with zoom, hover tooltips, and time range selection
- **Metric Correlation Analysis**: Visualize relationships between different metrics
- **Insight Distribution Analysis**: Analyze the distribution of different insight types over time
- **Scenario Comparison**: Compare different simulation scenarios (e.g., overtraining vs. recovery)
- **User Data Visualization**: Visualize real user data with personalized insights

## Components

### Timeline Visualizer (`timeline_visualizer.py`)

The core visualization engine that creates static plots using Matplotlib:

- `TimelineVisualizer`: Class for creating static timeline visualizations
  - `plot_timeline()`: Plot multiple metrics with insight annotations
  - `plot_metric_comparison()`: Compare relationships between pairs of metrics
  - `plot_insight_distribution()`: Visualize the distribution of insight types

### Interactive Timeline (`timeline_interactive.py`)

Extends the core visualizer with interactive capabilities using Plotly:

- `InteractiveTimeline`: Class for creating interactive web-based visualizations
  - `create_interactive_timeline()`: Create an interactive timeline with zoom and hover functionality
  - `create_metric_correlation_dashboard()`: Create an interactive correlation matrix
  - `create_insight_timeline()`: Create a focused timeline of insights

### Timeline Integration (`timeline_integration.py`)

Integrates the visualization components with other system components:

- `TimelineIntegration`: Class for connecting visualizations with the insight engine and simulator
  - `visualize_user_data()`: Create visualizations for a user's wearable data
  - `visualize_simulation()`: Create visualizations for simulation results
  - `compare_scenarios()`: Compare different simulation scenarios

## Usage Examples

See the `examples/timeline_example.py` script for comprehensive examples of using the Time-Series Insight Timeline components.

### Basic Usage

```python
from src.visualization.timeline_visualizer import TimelineVisualizer

# Create visualizer
visualizer = TimelineVisualizer()

# Plot timeline with insights
visualizer.plot_timeline(
    data,  # DataFrame with wearable data
    insights=insights,  # List of insight dictionaries
    title="Wearable Data Timeline",
    output_path="outputs/timeline.png"
)
```

### Interactive Visualization

```python
from src.visualization.timeline_interactive import InteractiveTimeline

# Create interactive timeline
interactive = InteractiveTimeline()

# Create interactive timeline visualization
interactive.create_interactive_timeline(
    data,  # DataFrame with wearable data
    insights=insights,  # List of insight dictionaries
    title="Interactive Wearable Data Timeline",
    output_path="outputs/interactive_timeline.html"
)
```

### Integration with Simulator

```python
from src.visualization.timeline_integration import TimelineIntegration
from src.simulator_engine import SimulatorEngine

# Create simulator and integration
simulator = SimulatorEngine()
integration = TimelineIntegration()

# Run simulation
simulation_data, simulation_insights = simulator.simulate_scenario(
    scenario_type='overtraining',
    days=30,
    user_profile=user_profile
)

# Visualize simulation results
integration.visualize_simulation(
    simulation_data,
    simulation_insights,
    scenario_type='overtraining',
    user_profile=user_profile
)
```

## Data Format

### Required Data Format

The timeline visualizers expect data in a pandas DataFrame with the following structure:

- A `date` column with datetime values
- One or more metric columns (e.g., `hrv_rmssd`, `sleep_hours`, `recovery_score`, `strain`)

### Insight Format

Insights should be provided as a list of dictionaries with the following structure:

```python
insights = [
    {
        'date': '2025-01-05',  # Date of the insight (string or datetime)
        'type': 'recovery',    # Type of insight (recovery, sleep, activity, etc.)
        'summary': 'Your recovery is trending upward...'  # Insight content
    },
    # More insights...
]
```

## Output Examples

The Time-Series Insight Timeline generates various types of visualizations:

1. **Timeline Plots**: Show multiple metrics over time with insight annotations
2. **Metric Correlation Plots**: Show relationships between different metrics
3. **Insight Distribution Plots**: Show the distribution of different insight types
4. **Interactive Visualizations**: Web-based visualizations with zoom and hover functionality

Output files are saved to the specified output directory, with default locations in the `outputs/` directory.

## Dependencies

- pandas
- numpy
- matplotlib
- plotly
- datetime

## Future Enhancements

- Add support for real-time data streaming
- Implement more advanced statistical analysis of metric correlations
- Add customizable themes and visualization styles
- Integrate with web dashboard for online viewing
- Add export functionality for reports and presentations
