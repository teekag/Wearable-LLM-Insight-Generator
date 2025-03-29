"""
Example script demonstrating the Interactive Insight Simulator functionality.

This script shows how to use the simulator_engine.py and demo_data_generator.py
to create synthetic wearable data scenarios and generate personalized insights.
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import required modules
from src.simulator_engine import SimulatorEngine
from src.demo_data_generator import DemoDataGenerator
from src.feature_engineer import FeatureEngineer
from src.llm_engine import LLMEngine

# Create output directories
os.makedirs("outputs/simulations", exist_ok=True)
os.makedirs("outputs/visualizations", exist_ok=True)

def run_simulation_example():
    """Run a complete simulation example."""
    print("=== Interactive Insight Simulator Example ===")
    
    # Step 1: Generate a user profile
    print("\nGenerating user profile...")
    data_generator = DemoDataGenerator(seed=42)
    user_profile = data_generator.generate_user_profile("athlete")
    print(f"Created profile for {user_profile['name']}")
    print(f"Primary goal: {user_profile['goals']['primary']}")
    print(f"Training personality: {user_profile['training_personality']}")
    
    # Step 2: Initialize simulator components
    print("\nInitializing simulator components...")
    feature_engineer = FeatureEngineer()
    llm_engine = LLMEngine()
    
    # Step 3: Create simulator instance
    print("\nCreating simulator instance...")
    simulator = SimulatorEngine(
        feature_engineer=feature_engineer,
        llm_engine=llm_engine,
        user_profile=user_profile
    )
    
    # Step 4: Run different scenario simulations
    scenarios = [
        ("overtraining", 7),
        ("recovery_phase", 7),
        ("sleep_deprivation", 5),
        ("training_peak", 5),
        ("circadian_disruption", 7)
    ]
    
    results = {}
    
    for scenario, days in scenarios:
        print(f"\nRunning simulation: {scenario} for {days} days...")
        simulator.set_simulation_parameters(scenario, days=days)
        
        # Run simulation
        sim_data, insights = simulator.run_simulation()
        
        # Visualize results
        output_path = f"outputs/visualizations/{scenario}_simulation.png"
        simulator.visualize_simulation(output_path)
        
        # Export results
        results_path = f"outputs/simulations/{scenario}_results.json"
        results[scenario] = simulator.export_results(results_path)
        
        print(f"Generated {len(insights)} insights")
        print(f"Visualization saved to {output_path}")
        print(f"Results saved to {results_path}")
    
    # Step 5: Compare scenarios
    print("\nComparing scenarios...")
    compare_scenarios(results)
    
    return results

def compare_scenarios(results):
    """Compare metrics across different scenarios."""
    # Extract key metrics for comparison
    comparison = {}
    
    for scenario, data in results.items():
        if "simulation_data" in data:
            # Calculate average metrics
            metrics = {
                "avg_recovery": sum(item["recovery_score"] for item in data["simulation_data"]) / len(data["simulation_data"]),
                "avg_hrv": sum(item["hrv_rmssd"] for item in data["simulation_data"]) / len(data["simulation_data"]),
                "avg_sleep": sum(item["sleep_hours"] for item in data["simulation_data"]) / len(data["simulation_data"]),
                "avg_strain": sum(item["strain"] for item in data["simulation_data"]) / len(data["simulation_data"]),
                "insight_count": len(data.get("insights", [])),
            }
            comparison[scenario] = metrics
    
    # Create comparison visualization
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    scenarios = list(comparison.keys())
    
    # Recovery scores
    axs[0, 0].bar(scenarios, [comparison[s]["avg_recovery"] for s in scenarios], color='blue')
    axs[0, 0].set_title('Average Recovery Score')
    axs[0, 0].set_ylim(0, 100)
    axs[0, 0].set_xticklabels(scenarios, rotation=45, ha='right')
    
    # HRV
    axs[0, 1].bar(scenarios, [comparison[s]["avg_hrv"] for s in scenarios], color='green')
    axs[0, 1].set_title('Average HRV (RMSSD)')
    axs[0, 1].set_xticklabels(scenarios, rotation=45, ha='right')
    
    # Sleep
    axs[1, 0].bar(scenarios, [comparison[s]["avg_sleep"] for s in scenarios], color='purple')
    axs[1, 0].set_title('Average Sleep Hours')
    axs[1, 0].set_xticklabels(scenarios, rotation=45, ha='right')
    
    # Strain
    axs[1, 1].bar(scenarios, [comparison[s]["avg_strain"] for s in scenarios], color='orange')
    axs[1, 1].set_title('Average Strain')
    axs[1, 1].set_xticklabels(scenarios, rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig("outputs/visualizations/scenario_comparison.png")
    plt.close()
    
    print(f"Scenario comparison visualization saved to outputs/visualizations/scenario_comparison.png")
    
    # Save comparison data
    with open("outputs/simulations/scenario_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)
    
    print(f"Scenario comparison data saved to outputs/simulations/scenario_comparison.json")

def run_custom_simulation():
    """Run a custom simulation with specific parameters."""
    print("\n=== Custom Simulation Example ===")
    
    # Create a custom user profile
    custom_profile = {
        "user_id": "custom_user_001",
        "name": "Custom User",
        "age": 35,
        "gender": "not_specified",
        "height_cm": 175,
        "weight_kg": 70,
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
        },
        "preferences": {
            "insight_tone": "coach",
            "insight_detail_level": "detailed"
        }
    }
    
    # Initialize simulator
    simulator = SimulatorEngine(user_profile=custom_profile)
    
    # Set custom simulation parameters
    simulator.set_simulation_parameters("overtraining", days=10)
    
    # Run simulation
    sim_data, insights = simulator.run_simulation()
    
    # Visualize and export
    simulator.visualize_simulation("outputs/visualizations/custom_simulation.png")
    simulator.export_results("outputs/simulations/custom_simulation.json")
    
    print(f"Custom simulation complete with {len(insights)} insights generated")
    print("Visualization saved to outputs/visualizations/custom_simulation.png")
    print("Results saved to outputs/simulations/custom_simulation.json")

if __name__ == "__main__":
    # Check if OpenAI API key is set (for LLM insights)
    if not os.environ.get("OPENAI_API_KEY"):
        print("Note: OPENAI_API_KEY environment variable not set. LLM-based insights will not be generated.")
        print("You can set it with: export OPENAI_API_KEY=your_api_key")
        print("Running in demo mode without LLM insights...\n")
    
    # Create directories
    os.makedirs("outputs/simulations", exist_ok=True)
    os.makedirs("outputs/visualizations", exist_ok=True)
    
    # Run the examples
    try:
        results = run_simulation_example()
        run_custom_simulation()
        print("\nAll examples completed successfully!")
    except Exception as e:
        print(f"Error running examples: {str(e)}")
