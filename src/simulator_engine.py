"""
Interactive Insight Simulator Module for Wearable LLM Insight Generator

This module provides simulation capabilities for generating synthetic wearable data
and producing personalized insights based on predefined scenarios.
"""

import json
import os
import random
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime, timedelta
import logging
import matplotlib.pyplot as plt
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define scenario types
SCENARIO_TYPES = {
    "training_peak": "High intensity training day with elevated heart rate and strain",
    "sleep_deprivation": "Period with reduced sleep quality and quantity",
    "overtraining": "Extended period of high training load without adequate recovery",
    "recovery_phase": "Period of active recovery with reduced training load",
    "circadian_disruption": "Disruption to normal sleep-wake patterns (e.g., travel, shift work)"
}

class SimulatorEngine:
    """Class to simulate wearable data scenarios and generate insights."""
    
    def __init__(self, 
                 insight_engine=None, 
                 feature_engineer=None, 
                 llm_engine=None,
                 user_profile: Optional[Dict[str, Any]] = None):
        """
        Initialize the Simulator Engine.
        
        Args:
            insight_engine: Optional insight engine instance
            feature_engineer: Optional feature engineer instance
            llm_engine: Optional LLM engine instance
            user_profile: Optional user profile dictionary
        """
        self.insight_engine = insight_engine
        self.feature_engineer = feature_engineer
        self.llm_engine = llm_engine
        self.user_profile = user_profile or self._create_default_profile()
        self.sim_days = 7  # Default simulation period
        self.scenario_type = "recovery_phase"  # Default scenario
        self.simulation_data = None
        self.insights = []
        
        # Create output directory
        os.makedirs("outputs/simulations", exist_ok=True)
    
    def _create_default_profile(self) -> Dict[str, Any]:
        """
        Create a default user profile if none is provided.
        
        Returns:
            Default user profile dictionary
        """
        return {
            "user_id": "sim_user_001",
            "name": "Simulation User",
            "age": 35,
            "gender": "not_specified",
            "height_cm": 175,
            "weight_kg": 70,
            "goals": {
                "primary": "improve_fitness",
                "secondary": ["increase_endurance", "reduce_stress"]
            },
            "training_personality": "balanced",  # Options: cautious, balanced, aggressive
            "baseline_metrics": {
                "resting_hr": 60,
                "hrv_rmssd": 65,
                "sleep_hours": 7.5,
                "recovery_score": 75
            },
            "preferences": {
                "insight_tone": "coach",
                "insight_detail_level": "moderate"
            }
        }
    
    def set_user_profile(self, profile: Dict[str, Any]) -> None:
        """
        Set the user profile for simulation.
        
        Args:
            profile: User profile dictionary
        """
        self.user_profile = profile
        logger.info(f"Set user profile for simulation: {profile.get('name', 'Unknown')}")
    
    def set_simulation_parameters(self, scenario_type: str, days: int = 7) -> None:
        """
        Set simulation parameters.
        
        Args:
            scenario_type: Type of scenario to simulate
            days: Number of days to simulate
        """
        if scenario_type not in SCENARIO_TYPES:
            valid_scenarios = ", ".join(SCENARIO_TYPES.keys())
            logger.warning(f"Invalid scenario type '{scenario_type}'. Using default. Valid options: {valid_scenarios}")
            scenario_type = "recovery_phase"
        
        self.scenario_type = scenario_type
        self.sim_days = max(1, min(days, 30))  # Limit between 1-30 days
        
        logger.info(f"Set simulation parameters: {scenario_type} for {self.sim_days} days")
    
    def generate_baseline(self) -> pd.DataFrame:
        """
        Generate baseline data based on user profile.
        
        Returns:
            DataFrame with baseline data
        """
        # Get baseline metrics from user profile
        baseline = self.user_profile.get("baseline_metrics", {})
        resting_hr = baseline.get("resting_hr", 60)
        hrv_rmssd = baseline.get("hrv_rmssd", 65)
        sleep_hours = baseline.get("sleep_hours", 7.5)
        recovery_score = baseline.get("recovery_score", 75)
        
        # Create date range
        end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        start_date = end_date - timedelta(days=self.sim_days - 1)
        date_range = [start_date + timedelta(days=i) for i in range(self.sim_days)]
        
        # Create baseline DataFrame with natural variation
        data = {
            "date": date_range,
            "resting_hr": [max(40, min(100, resting_hr + random.normalvariate(0, 2))) for _ in range(self.sim_days)],
            "hrv_rmssd": [max(20, min(120, hrv_rmssd + random.normalvariate(0, 5))) for _ in range(self.sim_days)],
            "sleep_hours": [max(4, min(10, sleep_hours + random.normalvariate(0, 0.5))) for _ in range(self.sim_days)],
            "sleep_quality": [random.uniform(0.7, 0.9) for _ in range(self.sim_days)],
            "recovery_score": [max(30, min(100, recovery_score + random.normalvariate(0, 3))) for _ in range(self.sim_days)],
            "strain": [random.uniform(8, 12) for _ in range(self.sim_days)],
            "steps": [random.randint(7000, 12000) for _ in range(self.sim_days)],
            "active_calories": [random.randint(300, 600) for _ in range(self.sim_days)]
        }
        
        return pd.DataFrame(data)
    
    def apply_scenario(self, baseline_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply scenario pattern to baseline data.
        
        Args:
            baseline_df: DataFrame with baseline data
            
        Returns:
            DataFrame with scenario applied
        """
        df = baseline_df.copy()
        
        # Apply scenario-specific patterns
        if self.scenario_type == "training_peak":
            # Simulate a high-intensity training day in the middle of the period
            peak_day = self.sim_days // 2
            
            # Increase strain and HR on peak day and day after
            df.loc[peak_day, "strain"] = random.uniform(18, 20)
            df.loc[peak_day, "resting_hr"] = df.loc[peak_day, "resting_hr"] * 1.15
            df.loc[peak_day, "active_calories"] = random.randint(800, 1000)
            
            # Decrease recovery and HRV after peak day
            if peak_day + 1 < self.sim_days:
                df.loc[peak_day + 1, "recovery_score"] = df.loc[peak_day + 1, "recovery_score"] * 0.8
                df.loc[peak_day + 1, "hrv_rmssd"] = df.loc[peak_day + 1, "hrv_rmssd"] * 0.85
        
        elif self.scenario_type == "sleep_deprivation":
            # Simulate 3 days of poor sleep in the middle of the period
            mid_point = self.sim_days // 2
            start_idx = max(0, mid_point - 1)
            end_idx = min(self.sim_days - 1, mid_point + 1)
            
            # Decrease sleep metrics
            for i in range(start_idx, end_idx + 1):
                df.loc[i, "sleep_hours"] = random.uniform(3.5, 5.5)
                df.loc[i, "sleep_quality"] = random.uniform(0.4, 0.6)
                
                # Affect other metrics due to poor sleep
                if i > start_idx:  # Effects show up after first poor sleep night
                    df.loc[i, "recovery_score"] = df.loc[i, "recovery_score"] * 0.85
                    df.loc[i, "hrv_rmssd"] = df.loc[i, "hrv_rmssd"] * 0.8
                    df.loc[i, "resting_hr"] = df.loc[i, "resting_hr"] * 1.1
        
        elif self.scenario_type == "overtraining":
            # Simulate progressively increasing strain and decreasing recovery
            strain_progression = np.linspace(12, 20, self.sim_days)
            recovery_decline = np.linspace(1.0, 0.7, self.sim_days)
            hrv_decline = np.linspace(1.0, 0.75, self.sim_days)
            
            for i in range(self.sim_days):
                df.loc[i, "strain"] = strain_progression[i]
                df.loc[i, "recovery_score"] = df.loc[i, "recovery_score"] * recovery_decline[i]
                df.loc[i, "hrv_rmssd"] = df.loc[i, "hrv_rmssd"] * hrv_decline[i]
                df.loc[i, "resting_hr"] = df.loc[i, "resting_hr"] * (2 - recovery_decline[i])
        
        elif self.scenario_type == "recovery_phase":
            # Simulate a recovery period with decreasing strain and improving metrics
            strain_decrease = np.linspace(15, 8, self.sim_days)
            recovery_improvement = np.linspace(0.8, 1.1, self.sim_days)
            
            for i in range(self.sim_days):
                df.loc[i, "strain"] = strain_decrease[i]
                recovery_factor = recovery_improvement[i]
                df.loc[i, "recovery_score"] = min(100, df.loc[i, "recovery_score"] * recovery_factor)
                df.loc[i, "hrv_rmssd"] = df.loc[i, "hrv_rmssd"] * recovery_factor
                df.loc[i, "resting_hr"] = df.loc[i, "resting_hr"] / recovery_factor
        
        elif self.scenario_type == "circadian_disruption":
            # Simulate disrupted sleep patterns (e.g., travel, shift work)
            for i in range(self.sim_days):
                # Alternate between normal and disrupted sleep
                if i % 2 == 0:
                    df.loc[i, "sleep_hours"] = random.uniform(4.0, 5.5)
                    df.loc[i, "sleep_quality"] = random.uniform(0.5, 0.6)
                else:
                    # Recovery attempt but still disrupted
                    df.loc[i, "sleep_hours"] = random.uniform(6.0, 7.0)
                    df.loc[i, "sleep_quality"] = random.uniform(0.6, 0.7)
                
                # Affect other metrics based on cumulative disruption
                disruption_factor = 1.0 - (0.05 * min(i, 5))  # Max 25% disruption after 5 days
                df.loc[i, "recovery_score"] = df.loc[i, "recovery_score"] * disruption_factor
                df.loc[i, "hrv_rmssd"] = df.loc[i, "hrv_rmssd"] * disruption_factor
        
        return df
    
    def run_simulation(self) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """
        Run the full simulation workflow.
        
        Returns:
            Tuple of (simulation data DataFrame, list of insights)
        """
        logger.info(f"Starting simulation: {self.scenario_type} for {self.sim_days} days")
        
        # Generate baseline data
        baseline_df = self.generate_baseline()
        
        # Apply scenario to baseline
        self.simulation_data = self.apply_scenario(baseline_df)
        
        # Generate insights if we have the necessary components
        insights = []
        if all([self.insight_engine, self.feature_engineer, self.llm_engine]):
            try:
                # Generate insights at key points in the simulation
                insight_days = self._determine_insight_days()
                
                for day_idx in insight_days:
                    # Get data up to this day
                    day_data = self.simulation_data.iloc[:day_idx+1]
                    
                    # Extract features
                    features = self.feature_engineer.extract_features(day_data)
                    
                    # Generate insight
                    insight = self.insight_engine.generate_insight(
                        data=day_data,
                        user_id=self.user_profile.get("user_id"),
                        user_config={
                            "goals": self.user_profile.get("goals", {}),
                            "preferences": self.user_profile.get("preferences", {})
                        }
                    )
                    
                    if insight:
                        # Add day information
                        for i in insight:
                            i.day = day_idx
                            i.date = self.simulation_data.iloc[day_idx]["date"]
                        
                        insights.extend(insight)
            except Exception as e:
                logger.error(f"Error generating insights: {str(e)}")
        
        self.insights = insights
        logger.info(f"Simulation complete with {len(insights)} insights generated")
        
        return self.simulation_data, self.insights
    
    def _determine_insight_days(self) -> List[int]:
        """
        Determine which days to generate insights for based on scenario.
        
        Returns:
            List of day indices for insight generation
        """
        if self.sim_days <= 3:
            # For short simulations, generate insights for all days
            return list(range(self.sim_days))
        
        if self.scenario_type == "training_peak":
            # Generate insights for day before peak, peak day, and day after
            peak_day = self.sim_days // 2
            return [max(0, peak_day-1), peak_day, min(self.sim_days-1, peak_day+1)]
        
        elif self.scenario_type == "overtraining":
            # Generate insights at beginning, middle, and end to show progression
            return [0, self.sim_days//2, self.sim_days-1]
        
        elif self.scenario_type in ["sleep_deprivation", "circadian_disruption"]:
            # Generate insights after first disruption, during worst day, and at end
            return [1, self.sim_days//2, self.sim_days-1]
        
        elif self.scenario_type == "recovery_phase":
            # Generate insights at start of recovery, middle, and end
            return [0, self.sim_days//2, self.sim_days-1]
        
        # Default: beginning, middle, end
        return [0, self.sim_days//2, self.sim_days-1]
    
    def visualize_simulation(self, save_path: Optional[str] = None) -> None:
        """
        Visualize the simulation results.
        
        Args:
            save_path: Optional path to save the visualization
        """
        if self.simulation_data is None:
            logger.warning("No simulation data available to visualize")
            return
        
        # Create figure with subplots
        fig, axs = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
        
        # Convert date to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(self.simulation_data['date']):
            self.simulation_data['date'] = pd.to_datetime(self.simulation_data['date'])
        
        # Plot 1: Recovery and HRV
        ax1 = axs[0]
        ax1.plot(self.simulation_data['date'], self.simulation_data['recovery_score'], 'b-', label='Recovery Score')
        ax1.set_ylabel('Recovery Score', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        
        ax1b = ax1.twinx()
        ax1b.plot(self.simulation_data['date'], self.simulation_data['hrv_rmssd'], 'g-', label='HRV (RMSSD)')
        ax1b.set_ylabel('HRV (RMSSD)', color='g')
        ax1b.tick_params(axis='y', labelcolor='g')
        
        ax1.set_title(f'Simulation: {self.scenario_type.replace("_", " ").title()}')
        
        # Plot 2: Sleep
        ax2 = axs[1]
        ax2.bar(self.simulation_data['date'], self.simulation_data['sleep_hours'], color='purple', alpha=0.7, label='Sleep Hours')
        ax2.set_ylabel('Sleep Hours')
        
        ax2b = ax2.twinx()
        ax2b.plot(self.simulation_data['date'], self.simulation_data['sleep_quality'], 'r-', label='Sleep Quality')
        ax2b.set_ylabel('Sleep Quality', color='r')
        ax2b.tick_params(axis='y', labelcolor='r')
        
        # Plot 3: Strain and HR
        ax3 = axs[2]
        ax3.plot(self.simulation_data['date'], self.simulation_data['strain'], 'orange', label='Strain')
        ax3.set_ylabel('Strain', color='orange')
        ax3.tick_params(axis='y', labelcolor='orange')
        
        ax3b = ax3.twinx()
        ax3b.plot(self.simulation_data['date'], self.simulation_data['resting_hr'], 'brown', label='Resting HR')
        ax3b.set_ylabel('Resting HR', color='brown')
        ax3b.tick_params(axis='y', labelcolor='brown')
        
        ax3.set_xlabel('Date')
        
        # Add insight markers if available
        if self.insights:
            for insight in self.insights:
                if hasattr(insight, 'day') and insight.day < len(self.simulation_data):
                    insight_date = self.simulation_data.iloc[insight.day]['date']
                    
                    # Add marker on each plot
                    axs[0].axvline(x=insight_date, color='black', linestyle='--', alpha=0.3)
                    axs[1].axvline(x=insight_date, color='black', linestyle='--', alpha=0.3)
                    axs[2].axvline(x=insight_date, color='black', linestyle='--', alpha=0.3)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Saved simulation visualization to {save_path}")
        
        plt.close()
    
    def export_results(self, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Export simulation results as structured data.
        
        Args:
            output_path: Optional path to save results JSON
            
        Returns:
            Dictionary with simulation results
        """
        if self.simulation_data is None:
            logger.warning("No simulation data available to export")
            return {}
        
        # Convert DataFrame to dict for JSON serialization
        sim_data_dict = self.simulation_data.to_dict(orient='records')
        
        # Format insights for export
        formatted_insights = []
        for insight in self.insights:
            if hasattr(insight, '__dict__'):
                # Convert insight object to dict
                insight_dict = insight.__dict__.copy()
                
                # Convert datetime objects to strings
                for key, value in insight_dict.items():
                    if isinstance(value, datetime):
                        insight_dict[key] = value.isoformat()
                
                formatted_insights.append(insight_dict)
            else:
                # If insight is already a dict
                formatted_insights.append(insight)
        
        # Create results dictionary
        results = {
            "scenario": self.scenario_type,
            "days": self.sim_days,
            "user_profile": self.user_profile,
            "simulation_data": sim_data_dict,
            "insights": formatted_insights,
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "version": "1.0.0"
            }
        }
        
        # Save to file if path provided
        if output_path:
            try:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, 'w') as f:
                    json.dump(results, f, indent=2)
                logger.info(f"Saved simulation results to {output_path}")
            except Exception as e:
                logger.error(f"Error saving simulation results: {str(e)}")
        
        return results
    
    def load_scenario_template(self, scenario_name: str) -> Dict[str, Any]:
        """
        Load a predefined scenario template.
        
        Args:
            scenario_name: Name of the scenario template
            
        Returns:
            Scenario template dictionary
        """
        # Define path to scenario templates
        template_dir = Path("data/scenario_templates")
        template_path = template_dir / f"{scenario_name}.json"
        
        # Create directory if it doesn't exist
        os.makedirs(template_dir, exist_ok=True)
        
        # Check if template exists
        if template_path.exists():
            try:
                with open(template_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading scenario template: {str(e)}")
        
        # Return default template if not found
        logger.warning(f"Scenario template '{scenario_name}' not found, using default")
        
        # Create default templates if they don't exist
        self._create_default_templates()
        
        # Return default for requested scenario or generic default
        default_templates = {
            "training_peak": {
                "scenario": "training_peak",
                "days": 5,
                "strain": [10, 12, 19, 15, 10],
                "hrv": [68, 65, 55, 60, 65]
            },
            "sleep_deprivation": {
                "scenario": "sleep_deprivation",
                "days": 5,
                "sleep_hours": [7.5, 4.2, 4.5, 5.0, 6.8],
                "sleep_quality": [0.85, 0.55, 0.60, 0.65, 0.75]
            },
            "overtraining": {
                "scenario": "overtraining",
                "days": 7,
                "strain": [14, 15, 16, 17, 18, 19, 20],
                "hrv": [68, 65, 60, 55, 50, 45, 40],
                "recovery_score": [80, 75, 70, 65, 60, 55, 50]
            },
            "recovery_phase": {
                "scenario": "recovery_phase",
                "days": 7,
                "strain": [18, 16, 14, 12, 10, 8, 8],
                "hrv": [50, 55, 60, 65, 70, 75, 78],
                "recovery_score": [55, 60, 65, 70, 75, 80, 85]
            },
            "circadian_disruption": {
                "scenario": "circadian_disruption",
                "days": 7,
                "sleep_hours": [7.5, 5.0, 6.0, 4.5, 6.5, 5.0, 7.0],
                "sleep_quality": [0.85, 0.60, 0.70, 0.55, 0.65, 0.60, 0.75]
            }
        }
        
        return default_templates.get(scenario_name, default_templates["recovery_phase"])
    
    def _create_default_templates(self) -> None:
        """Create default scenario templates if they don't exist."""
        template_dir = Path("data/scenario_templates")
        os.makedirs(template_dir, exist_ok=True)
        
        templates = {
            "training_peak": {
                "scenario": "training_peak",
                "days": 5,
                "description": "High intensity training day with elevated heart rate and strain",
                "strain": [10, 12, 19, 15, 10],
                "hrv": [68, 65, 55, 60, 65],
                "resting_hr": [58, 60, 65, 62, 59],
                "recovery_score": [80, 75, 65, 70, 78]
            },
            "sleep_deprivation": {
                "scenario": "sleep_deprivation",
                "days": 5,
                "description": "Period with reduced sleep quality and quantity",
                "sleep_hours": [7.5, 4.2, 4.5, 5.0, 6.8],
                "sleep_quality": [0.85, 0.55, 0.60, 0.65, 0.75],
                "hrv": [68, 60, 55, 58, 64],
                "recovery_score": [80, 65, 60, 65, 75]
            },
            "overtraining": {
                "scenario": "overtraining",
                "days": 7,
                "description": "Extended period of high training load without adequate recovery",
                "strain": [14, 15, 16, 17, 18, 19, 20],
                "hrv": [68, 65, 60, 55, 50, 45, 40],
                "recovery_score": [80, 75, 70, 65, 60, 55, 50],
                "resting_hr": [58, 60, 62, 64, 66, 68, 70]
            },
            "recovery_phase": {
                "scenario": "recovery_phase",
                "days": 7,
                "description": "Period of active recovery with reduced training load",
                "strain": [18, 16, 14, 12, 10, 8, 8],
                "hrv": [50, 55, 60, 65, 70, 75, 78],
                "recovery_score": [55, 60, 65, 70, 75, 80, 85],
                "resting_hr": [68, 66, 64, 62, 60, 58, 57]
            },
            "circadian_disruption": {
                "scenario": "circadian_disruption",
                "days": 7,
                "description": "Disruption to normal sleep-wake patterns (e.g., travel, shift work)",
                "sleep_hours": [7.5, 5.0, 6.0, 4.5, 6.5, 5.0, 7.0],
                "sleep_quality": [0.85, 0.60, 0.70, 0.55, 0.65, 0.60, 0.75],
                "hrv": [68, 60, 63, 55, 62, 58, 65],
                "recovery_score": [80, 65, 70, 60, 68, 63, 75]
            }
        }
        
        # Save templates
        for name, template in templates.items():
            template_path = template_dir / f"{name}.json"
            if not template_path.exists():
                try:
                    with open(template_path, 'w') as f:
                        json.dump(template, f, indent=2)
                except Exception as e:
                    logger.error(f"Error creating template {name}: {str(e)}")


# Example usage
if __name__ == "__main__":
    try:
        from feature_engineer import FeatureEngineer
        from llm_engine import LLMEngine
        
        # Sample user profile
        sample_profile = {
            "user_id": "test_user_001",
            "name": "Test User",
            "age": 32,
            "gender": "female",
            "height_cm": 168,
            "weight_kg": 65,
            "goals": {
                "primary": "improve_endurance",
                "secondary": ["reduce_stress", "improve_sleep"]
            },
            "training_personality": "aggressive",
            "baseline_metrics": {
                "resting_hr": 55,
                "hrv_rmssd": 72,
                "sleep_hours": 7.2,
                "recovery_score": 78
            }
        }
        
        # Initialize components
        feature_engineer = FeatureEngineer()
        llm_engine = LLMEngine()
        
        # Initialize simulator
        simulator = SimulatorEngine(
            feature_engineer=feature_engineer,
            llm_engine=llm_engine,
            user_profile=sample_profile
        )
        
        # Set simulation parameters
        simulator.set_simulation_parameters("overtraining", days=7)
        
        # Run simulation
        sim_data, insights = simulator.run_simulation()
        
        # Visualize results
        simulator.visualize_simulation("outputs/simulations/overtraining_simulation.png")
        
        # Export results
        simulator.export_results("outputs/simulations/overtraining_results.json")
        
        print(f"Simulation complete with {len(insights)} insights generated")
        
    except ImportError as e:
        print(f"Could not run example: {str(e)}")
        print("This is expected if running as a module or if dependencies are not installed.")
