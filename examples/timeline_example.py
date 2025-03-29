"""
Example script demonstrating the Time-Series Insight Timeline functionality.

This example shows how to use both the static and interactive timeline visualizers
to create insightful visualizations of wearable data with AI-generated insights.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import logging
import random

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import project modules
from src.visualization.timeline_visualizer import TimelineVisualizer
from src.visualization.timeline_interactive import InteractiveTimeline
from src.visualization.timeline_integration import TimelineIntegration
from src.demo_data_generator import DemoDataGenerator
from src.simulator_engine import SimulatorEngine

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_basic_timeline_example():
    """Run a basic example of the timeline visualizer with demo data."""
    logger.info("Running basic timeline example...")
    
    # Create demo data generator
    data_gen = DemoDataGenerator()
    
    # Generate 30 days of wearable data
    profile, data, _ = data_gen.generate_demo_dataset(
        profile_type="active",
        days=30,
        scenario=None
    )
    
    # Manually add the metrics we need if they don't exist
    for metric in ['hrv_rmssd', 'sleep_hours', 'recovery_score', 'strain']:
        if metric not in data.columns:
            if metric == 'hrv_rmssd':
                data[metric] = np.random.normal(65, 10, 30)
            elif metric == 'sleep_hours':
                data[metric] = np.random.normal(7.5, 1, 30)
            elif metric == 'recovery_score':
                data[metric] = np.random.normal(75, 15, 30)
            elif metric == 'strain':
                data[metric] = np.random.normal(12, 4, 30)
    
    # Create sample insights
    insights = [
        {
            'date': '2025-01-05',
            'type': 'recovery',
            'summary': 'Your recovery is trending upward, indicating good adaptation to recent training.'
        },
        {
            'date': '2025-01-10',
            'type': 'sleep',
            'summary': 'Your sleep duration has decreased over the past 3 days, which may impact recovery.'
        },
        {
            'date': '2025-01-15',
            'type': 'strain',
            'summary': 'High strain detected. Consider a recovery day tomorrow.'
        },
        {
            'date': '2025-01-20',
            'type': 'warning',
            'summary': 'HRV trending downward for 5 consecutive days, indicating possible overtraining.'
        },
        {
            'date': '2025-01-25',
            'type': 'recommendation',
            'summary': 'Based on your recovery trends, consider increasing training intensity next week.'
        }
    ]
    
    # Create output directory
    os.makedirs("outputs/examples", exist_ok=True)
    
    # Create static timeline visualization
    visualizer = TimelineVisualizer()
    
    # Plot timeline with insights
    visualizer.plot_timeline(
        data, 
        insights=insights,
        title="Example Wearable Data Timeline",
        output_path="outputs/examples/basic_timeline.png"
    )
    
    # Plot metric comparison
    visualizer.plot_metric_comparison(
        data,
        metric_pairs=[('hrv_rmssd', 'recovery_score'), ('sleep_hours', 'strain')],
        title="Metric Correlations",
        output_path="outputs/examples/metric_correlations.png"
    )
    
    # Plot insight distribution
    visualizer.plot_insight_distribution(
        insights,
        title="Insight Type Distribution",
        output_path="outputs/examples/insight_distribution.png"
    )
    
    logger.info("Basic timeline example completed. Visualizations saved to outputs/examples/")
    
    return data, insights

def run_interactive_timeline_example(data, insights):
    """Run an example of the interactive timeline visualizer."""
    logger.info("Running interactive timeline example...")
    
    # Create interactive timeline
    interactive = InteractiveTimeline()
    
    # Create interactive timeline visualization
    interactive.create_interactive_timeline(
        data,
        insights=insights,
        title="Interactive Wearable Data Timeline",
        output_path="outputs/examples/interactive_timeline.html"
    )
    
    # Create metric correlation dashboard
    interactive.create_metric_correlation_dashboard(
        data,
        title="Interactive Metric Correlation Dashboard",
        output_path="outputs/examples/correlation_dashboard.html"
    )
    
    # Create insight timeline
    interactive.create_insight_timeline(
        insights,
        title="Interactive Insight Timeline",
        output_path="outputs/examples/insight_timeline.html"
    )
    
    logger.info("Interactive timeline example completed. Visualizations saved to outputs/examples/")

def run_simulation_example():
    """Run an example using the simulator engine with timeline visualization."""
    logger.info("Running simulation example...")
    
    # Create simulator engine
    simulator = SimulatorEngine()
    
    # Define user profile
    user_profile = {
        'name': 'Alex',
        'age': 35,
        'gender': 'male',
        'activity_level': 'active',
        'training_goals': ['improve endurance', 'maintain strength'],
        'sleep_target': 8.0,
        'recovery_sensitivity': 'medium'
    }
    
    # Set user profile
    simulator.set_user_profile(user_profile)
    
    # Run overtraining simulation
    simulator.set_simulation_parameters(scenario_type='overtraining', days=30)
    overtraining_data, overtraining_insights = simulator.run_simulation()
    
    # Run recovery simulation
    simulator.set_simulation_parameters(scenario_type='recovery_phase', days=30)
    recovery_data, recovery_insights = simulator.run_simulation()
    
    # Create timeline integration
    integration = TimelineIntegration()
    
    # Visualize overtraining scenario
    integration.visualize_simulation(
        overtraining_data,
        overtraining_insights,
        scenario_type='overtraining',
        user_profile=user_profile,
        output_dir="outputs/examples/simulations/overtraining"
    )
    
    # Visualize recovery scenario
    integration.visualize_simulation(
        recovery_data,
        recovery_insights,
        scenario_type='recovery_phase',
        user_profile=user_profile,
        output_dir="outputs/examples/simulations/recovery"
    )
    
    # Compare scenarios
    scenario_results = {
        "overtraining": {
            "simulation_data": overtraining_data,
            "insights": overtraining_insights
        },
        "recovery": {
            "simulation_data": recovery_data,
            "insights": recovery_insights
        }
    }
    
    integration.compare_scenarios(
        scenario_results,
        output_dir="outputs/examples/simulations/comparison"
    )
    
    logger.info("Simulation example completed. Visualizations saved to outputs/examples/simulations/")

def run_comprehensive_example():
    """Run a comprehensive example using all timeline components."""
    logger.info("Running comprehensive timeline example...")
    
    # Create demo data generator
    data_gen = DemoDataGenerator()
    
    # Generate data for multiple users
    users = {
        'user_001': 'very_active',
        'user_002': 'moderately_active',
        'user_003': 'lightly_active'
    }
    
    user_data = {}
    for user_id, profile_type in users.items():
        # Generate 60 days of data
        profile, data, _ = data_gen.generate_demo_dataset(
            profile_type=profile_type,
            days=60
        )
        
        # Manually add the metrics we need if they don't exist
        for metric in ['hrv_rmssd', 'sleep_hours', 'recovery_score', 'strain', 'resting_hr']:
            if metric not in data.columns:
                if metric == 'hrv_rmssd':
                    data[metric] = np.random.normal(65, 10, 60)
                elif metric == 'sleep_hours':
                    data[metric] = np.random.normal(7.5, 1, 60)
                elif metric == 'recovery_score':
                    data[metric] = np.random.normal(75, 15, 60)
                elif metric == 'strain':
                    data[metric] = np.random.normal(12, 4, 60)
                elif metric == 'resting_hr':
                    data[metric] = np.random.normal(60, 5, 60)
        
        user_data[user_id] = data
    
    # Create simulator for generating insights
    simulator = SimulatorEngine()
    
    # Create timeline integration
    integration = TimelineIntegration()
    
    # Process each user
    for user_id, data in user_data.items():
        # Generate insights using simulator
        profile = {
            'name': f"User {user_id}",
            'age': 35,
            'gender': 'male',
            'activity_level': users[user_id],
            'training_goals': ['improve endurance', 'maintain strength'],
            'sleep_target': 8.0,
            'recovery_sensitivity': 'medium'
        }
        
        # Set user profile and generate insights
        simulator.set_user_profile(profile)
        
        # We'll use a random scenario for each user
        scenario_type = random.choice(['training_peak', 'recovery_phase', 'overtraining'])
        simulator.set_simulation_parameters(scenario_type=scenario_type, days=len(data))
        
        # Apply the scenario to the data
        simulated_data = simulator.apply_scenario(data)
        
        # Generate insights based on the scenario
        _, insights = simulator.run_simulation()
        
        # Create visualizations
        integration.visualize_user_data(
            user_id=user_id,
            data=simulated_data,
            include_insights=True,
            interactive=True,
            output_dir=f"outputs/examples/users/{user_id}"
        )
    
    logger.info("Comprehensive example completed. Visualizations saved to outputs/examples/users/")

if __name__ == "__main__":
    # Create output directories
    os.makedirs("outputs/examples", exist_ok=True)
    
    # Run basic example
    data, insights = run_basic_timeline_example()
    
    # Run interactive example
    run_interactive_timeline_example(data, insights)
    
    # Run simulation example
    run_simulation_example()
    
    # Run comprehensive example
    run_comprehensive_example()
    
    print("\nAll examples completed successfully!")
    print("Visualizations have been saved to the outputs/examples/ directory")
    print("\nStatic visualizations:")
    print("- Basic timeline: outputs/examples/basic_timeline.png")
    print("- Metric correlations: outputs/examples/metric_correlations.png")
    print("- Insight distribution: outputs/examples/insight_distribution.png")
    
    print("\nInteractive visualizations:")
    print("- Interactive timeline: outputs/examples/interactive_timeline.html")
    print("- Correlation dashboard: outputs/examples/correlation_dashboard.html")
    print("- Insight timeline: outputs/examples/insight_timeline.html")
    
    print("\nSimulation visualizations:")
    print("- Overtraining scenario: outputs/examples/simulations/overtraining/")
    print("- Recovery scenario: outputs/examples/simulations/recovery/")
    print("- Scenario comparison: outputs/examples/simulations/comparison/")
    
    print("\nUser visualizations:")
    for user_id in ['user_001', 'user_002', 'user_003']:
        print(f"- User {user_id}: outputs/examples/users/{user_id}/")
