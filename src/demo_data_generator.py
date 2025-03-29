"""
Demo Data Generator Module for Wearable LLM Insight Generator

This module provides synthetic data generation capabilities for wearable metrics,
allowing for testing and demonstration without requiring real device data.
"""

import os
import json
import random
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime, timedelta
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DemoDataGenerator:
    """Class to generate synthetic wearable data for demonstration and testing."""
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the Demo Data Generator.
        
        Args:
            seed: Optional random seed for reproducibility
        """
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Create output directory
        os.makedirs("outputs/demo_data", exist_ok=True)
        
        # Default parameters for data generation
        self.default_params = {
            "heart_rate": {
                "resting": 60,
                "max": 180,
                "daily_variation": 5
            },
            "hrv": {
                "baseline": 65,
                "daily_variation": 10
            },
            "sleep": {
                "target_hours": 8,
                "variation": 1.5,
                "quality_baseline": 0.8,
                "quality_variation": 0.15
            },
            "activity": {
                "steps_target": 10000,
                "steps_variation": 3000,
                "active_minutes_target": 60,
                "active_minutes_variation": 20
            },
            "recovery": {
                "baseline": 75,
                "variation": 15
            }
        }
    
    def generate_time_series(self, 
                           days: int = 30, 
                           metrics: Optional[List[str]] = None,
                           user_profile: Optional[Dict[str, Any]] = None,
                           start_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Generate synthetic time series data for wearable metrics.
        
        Args:
            days: Number of days to generate data for
            metrics: List of metrics to include (default: all available)
            user_profile: Optional user profile to tailor data generation
            start_date: Optional start date (default: days ago from today)
            
        Returns:
            DataFrame with synthetic time series data
        """
        # Set default metrics if not specified
        if metrics is None:
            metrics = ["heart_rate", "hrv", "sleep", "activity", "recovery", "strain"]
        
        # Set default start date if not specified
        if start_date is None:
            start_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=days-1)
        
        # Create date range
        date_range = [start_date + timedelta(days=i) for i in range(days)]
        
        # Initialize data dictionary with dates
        data = {"date": date_range}
        
        # Apply user profile if provided
        params = self.default_params.copy()
        if user_profile:
            self._apply_user_profile_to_params(params, user_profile)
        
        # Generate data for each requested metric
        if "heart_rate" in metrics:
            data.update(self._generate_heart_rate_data(days, params["heart_rate"]))
        
        if "hrv" in metrics:
            data.update(self._generate_hrv_data(days, params["hrv"]))
        
        if "sleep" in metrics:
            data.update(self._generate_sleep_data(days, params["sleep"]))
        
        if "activity" in metrics:
            data.update(self._generate_activity_data(days, params["activity"]))
        
        if "recovery" in metrics:
            data.update(self._generate_recovery_data(days, params["recovery"]))
        
        if "strain" in metrics:
            data.update(self._generate_strain_data(days, data))
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        logger.info(f"Generated synthetic time series data for {days} days with metrics: {', '.join(metrics)}")
        return df
    
    def _apply_user_profile_to_params(self, params: Dict[str, Dict[str, float]], user_profile: Dict[str, Any]) -> None:
        """
        Adjust generation parameters based on user profile.
        
        Args:
            params: Parameters dictionary to modify
            user_profile: User profile dictionary
        """
        # Extract relevant profile information
        baseline_metrics = user_profile.get("baseline_metrics", {})
        
        # Adjust heart rate parameters
        if "resting_hr" in baseline_metrics:
            params["heart_rate"]["resting"] = baseline_metrics["resting_hr"]
            # Adjust max HR based on age if available
            if "age" in user_profile:
                params["heart_rate"]["max"] = 220 - user_profile["age"]
        
        # Adjust HRV parameters
        if "hrv_rmssd" in baseline_metrics:
            params["hrv"]["baseline"] = baseline_metrics["hrv_rmssd"]
        
        # Adjust sleep parameters
        if "sleep_hours" in baseline_metrics:
            params["sleep"]["target_hours"] = baseline_metrics["sleep_hours"]
        
        # Adjust recovery parameters
        if "recovery_score" in baseline_metrics:
            params["recovery"]["baseline"] = baseline_metrics["recovery_score"]
        
        # Adjust parameters based on training personality
        personality = user_profile.get("training_personality", "balanced")
        if personality == "aggressive":
            # Aggressive trainers have more variability and push harder
            params["heart_rate"]["daily_variation"] *= 1.3
            params["hrv"]["daily_variation"] *= 1.5
            params["recovery"]["variation"] *= 1.4
        elif personality == "cautious":
            # Cautious trainers have less variability and more consistent patterns
            params["heart_rate"]["daily_variation"] *= 0.7
            params["hrv"]["daily_variation"] *= 0.8
            params["recovery"]["variation"] *= 0.7
    
    def _generate_heart_rate_data(self, days: int, params: Dict[str, float]) -> Dict[str, List[float]]:
        """
        Generate synthetic heart rate data.
        
        Args:
            days: Number of days
            params: Heart rate parameters
            
        Returns:
            Dictionary with heart rate metrics
        """
        resting_hr = params["resting"]
        max_hr = params["max"]
        variation = params["daily_variation"]
        
        # Generate resting heart rate with natural variation
        resting_hr_values = [max(40, min(100, resting_hr + random.normalvariate(0, variation))) for _ in range(days)]
        
        # Generate average heart rate (higher than resting)
        avg_hr_values = [rhr * random.uniform(1.2, 1.4) for rhr in resting_hr_values]
        
        # Generate max heart rate (with some days having higher intensity)
        max_hr_values = []
        for i in range(days):
            # Higher intensity every 3-4 days
            intensity_day = (i % 4 == 0)
            if intensity_day:
                max_hr_values.append(max_hr * random.uniform(0.9, 1.0))
            else:
                max_hr_values.append(max_hr * random.uniform(0.7, 0.85))
        
        return {
            "resting_hr": resting_hr_values,
            "avg_hr": avg_hr_values,
            "max_hr": max_hr_values
        }
    
    def _generate_hrv_data(self, days: int, params: Dict[str, float]) -> Dict[str, List[float]]:
        """
        Generate synthetic HRV data.
        
        Args:
            days: Number of days
            params: HRV parameters
            
        Returns:
            Dictionary with HRV metrics
        """
        baseline = params["baseline"]
        variation = params["daily_variation"]
        
        # Generate HRV with natural variation and weekly pattern
        hrv_values = []
        for i in range(days):
            # Weekly pattern: lower on weekdays, higher on weekends
            day_of_week = (i % 7)
            weekend_factor = 1.1 if day_of_week >= 5 else 1.0
            
            # Random variation
            daily_hrv = max(20, min(120, baseline * weekend_factor + random.normalvariate(0, variation)))
            hrv_values.append(daily_hrv)
        
        # Calculate additional HRV metrics
        hrv_sdnn = [hrv * random.uniform(0.8, 1.2) for hrv in hrv_values]
        hrv_lf_hf = [random.uniform(0.5, 3.0) for _ in range(days)]
        
        return {
            "hrv_rmssd": hrv_values,
            "hrv_sdnn": hrv_sdnn,
            "hrv_lf_hf": hrv_lf_hf
        }
    
    def _generate_sleep_data(self, days: int, params: Dict[str, float]) -> Dict[str, List[float]]:
        """
        Generate synthetic sleep data.
        
        Args:
            days: Number of days
            params: Sleep parameters
            
        Returns:
            Dictionary with sleep metrics
        """
        target_hours = params["target_hours"]
        variation = params["variation"]
        quality_baseline = params["quality_baseline"]
        quality_variation = params["quality_variation"]
        
        # Generate sleep duration with natural variation and weekly pattern
        sleep_hours = []
        for i in range(days):
            # Weekly pattern: less sleep mid-week, more on weekends
            day_of_week = (i % 7)
            weekday_factor = 0.9 if 2 <= day_of_week <= 4 else 1.0
            
            # Random variation
            daily_sleep = max(3, min(10, target_hours * weekday_factor + random.normalvariate(0, variation)))
            sleep_hours.append(daily_sleep)
        
        # Generate sleep quality (correlated with duration)
        sleep_quality = []
        for hours in sleep_hours:
            # Quality tends to be better with more sleep, but not always
            duration_factor = min(1.0, hours / target_hours)
            quality = max(0.3, min(1.0, quality_baseline * duration_factor + random.normalvariate(0, quality_variation)))
            sleep_quality.append(quality)
        
        # Generate sleep stages
        deep_sleep_pct = [max(10, min(30, 20 + random.normalvariate(0, 5))) for _ in range(days)]
        rem_sleep_pct = [max(15, min(35, 25 + random.normalvariate(0, 5))) for _ in range(days)]
        light_sleep_pct = [max(35, min(70, 100 - deep - rem)) for deep, rem in zip(deep_sleep_pct, rem_sleep_pct)]
        
        # Calculate actual hours in each stage
        deep_sleep_hours = [hours * pct / 100 for hours, pct in zip(sleep_hours, deep_sleep_pct)]
        rem_sleep_hours = [hours * pct / 100 for hours, pct in zip(sleep_hours, rem_sleep_pct)]
        light_sleep_hours = [hours * pct / 100 for hours, pct in zip(sleep_hours, light_sleep_pct)]
        
        return {
            "sleep_hours": sleep_hours,
            "sleep_quality": sleep_quality,
            "deep_sleep_pct": deep_sleep_pct,
            "rem_sleep_pct": rem_sleep_pct,
            "light_sleep_pct": light_sleep_pct,
            "deep_sleep_hours": deep_sleep_hours,
            "rem_sleep_hours": rem_sleep_hours,
            "light_sleep_hours": light_sleep_hours
        }
    
    def _generate_activity_data(self, days: int, params: Dict[str, float]) -> Dict[str, List[float]]:
        """
        Generate synthetic activity data.
        
        Args:
            days: Number of days
            params: Activity parameters
            
        Returns:
            Dictionary with activity metrics
        """
        steps_target = params["steps_target"]
        steps_variation = params["steps_variation"]
        active_minutes_target = params["active_minutes_target"]
        active_minutes_variation = params["active_minutes_variation"]
        
        # Generate steps with natural variation and weekly pattern
        steps = []
        for i in range(days):
            # Weekly pattern: rest day once a week
            day_of_week = (i % 7)
            rest_day = (day_of_week == 6)  # Sunday as rest day
            
            if rest_day:
                # Rest days have fewer steps
                daily_steps = steps_target * random.uniform(0.4, 0.7)
            else:
                # Normal days with variation
                daily_steps = max(1000, steps_target + random.normalvariate(0, steps_variation))
            
            steps.append(int(daily_steps))
        
        # Generate active minutes (correlated with steps)
        active_minutes = []
        for step_count in steps:
            # Active minutes correlate with steps but have their own variation
            step_factor = step_count / steps_target
            minutes = max(5, int(active_minutes_target * step_factor + random.normalvariate(0, active_minutes_variation)))
            active_minutes.append(minutes)
        
        # Generate calories (correlated with steps and active minutes)
        base_calories = 1800  # Base metabolic calories
        active_calories = [int(am * random.uniform(8, 12)) for am in active_minutes]
        total_calories = [base_calories + ac for ac in active_calories]
        
        # Generate distance in km (based on steps, assuming average stride)
        distance_km = [steps * 0.0007 for steps in steps]  # ~0.7m per step
        
        return {
            "steps": steps,
            "active_minutes": active_minutes,
            "active_calories": active_calories,
            "total_calories": total_calories,
            "distance_km": distance_km
        }
    
    def _generate_recovery_data(self, days: int, params: Dict[str, float]) -> Dict[str, List[float]]:
        """
        Generate synthetic recovery data.
        
        Args:
            days: Number of days
            params: Recovery parameters
            
        Returns:
            Dictionary with recovery metrics
        """
        baseline = params["baseline"]
        variation = params["variation"]
        
        # Generate recovery scores with natural variation
        recovery_scores = []
        
        # Create a pattern with gradual decline and recovery
        cycle_length = random.randint(5, 10)  # Days in each decline/recovery cycle
        for i in range(days):
            cycle_position = i % cycle_length
            cycle_factor = 1.0 - 0.3 * (cycle_position / (cycle_length - 1))  # Decline by up to 30%
            
            # Add random variation
            score = max(30, min(100, baseline * cycle_factor + random.normalvariate(0, variation)))
            recovery_scores.append(score)
        
        # Generate readiness score (slightly different from recovery)
        readiness_scores = [max(30, min(100, rs + random.normalvariate(0, 5))) for rs in recovery_scores]
        
        return {
            "recovery_score": recovery_scores,
            "readiness_score": readiness_scores
        }
    
    def _generate_strain_data(self, days: int, existing_data: Dict[str, List]) -> Dict[str, List[float]]:
        """
        Generate synthetic strain data based on other metrics.
        
        Args:
            days: Number of days
            existing_data: Dictionary with already generated metrics
            
        Returns:
            Dictionary with strain metrics
        """
        # Strain is influenced by activity, heart rate, and inversely by recovery
        strain_values = []
        
        for i in range(days):
            base_strain = 10.0  # Base strain value
            
            # Adjust based on active minutes if available
            if "active_minutes" in existing_data:
                active_factor = existing_data["active_minutes"][i] / 60.0  # Normalize to 1 hour
                base_strain += active_factor * 5
            
            # Adjust based on heart rate if available
            if "max_hr" in existing_data:
                hr_factor = existing_data["max_hr"][i] / 180.0  # Normalize to max HR
                base_strain += hr_factor * 3
            
            # Adjust based on recovery if available (inverse relationship)
            if "recovery_score" in existing_data:
                recovery_factor = 1.0 - (existing_data["recovery_score"][i] / 100.0)
                base_strain += recovery_factor * 4
            
            # Add random variation
            strain = max(1, min(21, base_strain + random.normalvariate(0, 1.5)))
            strain_values.append(strain)
        
        return {
            "strain": strain_values
        }
    
    def generate_user_profile(self, profile_type: str = "random") -> Dict[str, Any]:
        """
        Generate a synthetic user profile.
        
        Args:
            profile_type: Type of profile to generate (random, athlete, casual, etc.)
            
        Returns:
            User profile dictionary
        """
        # Base profile with common fields
        profile = {
            "user_id": f"demo_{profile_type}_{random.randint(1000, 9999)}",
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat()
        }
        
        # Add demographic information
        if profile_type == "athlete":
            profile.update({
                "name": random.choice(["Alex Athlete", "Sam Sportsperson", "Taylor Trainer"]),
                "age": random.randint(25, 40),
                "gender": random.choice(["male", "female", "non_binary"]),
                "height_cm": random.randint(165, 190),
                "weight_kg": random.randint(60, 85),
                "training_personality": "aggressive",
                "goals": {
                    "primary": random.choice(["improve_performance", "increase_endurance", "race_preparation"]),
                    "secondary": random.sample(["maintain_fitness", "injury_prevention", "weight_management"], 2)
                },
                "baseline_metrics": {
                    "resting_hr": random.randint(45, 55),
                    "hrv_rmssd": random.randint(70, 90),
                    "sleep_hours": random.uniform(7.0, 8.0),
                    "recovery_score": random.randint(75, 85)
                }
            })
        
        elif profile_type == "casual":
            profile.update({
                "name": random.choice(["Casey Casual", "Jamie Jogger", "Riley Relaxed"]),
                "age": random.randint(30, 55),
                "gender": random.choice(["male", "female", "non_binary"]),
                "height_cm": random.randint(160, 185),
                "weight_kg": random.randint(65, 90),
                "training_personality": "balanced",
                "goals": {
                    "primary": random.choice(["improve_fitness", "weight_management", "stress_reduction"]),
                    "secondary": random.sample(["sleep_improvement", "general_health", "mood_enhancement"], 2)
                },
                "baseline_metrics": {
                    "resting_hr": random.randint(60, 70),
                    "hrv_rmssd": random.randint(50, 65),
                    "sleep_hours": random.uniform(6.5, 7.5),
                    "recovery_score": random.randint(65, 75)
                }
            })
        
        elif profile_type == "recovery_focused":
            profile.update({
                "name": random.choice(["Robin Recover", "Reese Restoration", "Morgan Mindful"]),
                "age": random.randint(35, 60),
                "gender": random.choice(["male", "female", "non_binary"]),
                "height_cm": random.randint(155, 180),
                "weight_kg": random.randint(60, 85),
                "training_personality": "cautious",
                "goals": {
                    "primary": random.choice(["stress_management", "sleep_improvement", "recovery_optimization"]),
                    "secondary": random.sample(["injury_prevention", "longevity", "mental_wellbeing"], 2)
                },
                "baseline_metrics": {
                    "resting_hr": random.randint(55, 65),
                    "hrv_rmssd": random.randint(60, 75),
                    "sleep_hours": random.uniform(7.5, 8.5),
                    "recovery_score": random.randint(70, 80)
                }
            })
        
        else:  # random profile
            profile.update({
                "name": f"User {random.randint(1000, 9999)}",
                "age": random.randint(25, 65),
                "gender": random.choice(["male", "female", "non_binary"]),
                "height_cm": random.randint(155, 190),
                "weight_kg": random.randint(55, 95),
                "training_personality": random.choice(["cautious", "balanced", "aggressive"]),
                "goals": {
                    "primary": random.choice([
                        "improve_fitness", "weight_management", "increase_endurance", 
                        "stress_reduction", "sleep_improvement", "recovery_optimization"
                    ]),
                    "secondary": random.sample([
                        "general_health", "injury_prevention", "mental_wellbeing",
                        "mood_enhancement", "longevity", "race_preparation"
                    ], 2)
                },
                "baseline_metrics": {
                    "resting_hr": random.randint(50, 75),
                    "hrv_rmssd": random.randint(45, 85),
                    "sleep_hours": random.uniform(6.0, 8.5),
                    "recovery_score": random.randint(60, 85)
                }
            })
        
        # Add preferences
        profile["preferences"] = {
            "insight_tone": random.choice(["coach", "scientist", "friend", "motivational"]),
            "insight_detail_level": random.choice(["basic", "moderate", "detailed"]),
            "notification_frequency": random.choice(["daily", "weekly", "important_only"])
        }
        
        logger.info(f"Generated synthetic user profile of type: {profile_type}")
        return profile
    
    def save_demo_data(self, df: pd.DataFrame, filename: str) -> str:
        """
        Save generated demo data to file.
        
        Args:
            df: DataFrame with demo data
            filename: Base filename (without extension)
            
        Returns:
            Path to saved file
        """
        # Create output directory
        output_dir = Path("outputs/demo_data")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save as CSV
        csv_path = output_dir / f"{filename}.csv"
        df.to_csv(csv_path, index=False)
        
        # Save as JSON
        json_path = output_dir / f"{filename}.json"
        
        # Convert date column to string for JSON serialization
        df_json = df.copy()
        if "date" in df_json.columns and pd.api.types.is_datetime64_any_dtype(df_json["date"]):
            df_json["date"] = df_json["date"].dt.strftime("%Y-%m-%d")
            
        df_json.to_json(json_path, orient="records", date_format="iso")
        
        logger.info(f"Saved demo data to {csv_path} and {json_path}")
        return str(csv_path)
    
    def save_user_profile(self, profile: Dict[str, Any], filename: Optional[str] = None) -> str:
        """
        Save generated user profile to file.
        
        Args:
            profile: User profile dictionary
            filename: Optional filename (default: based on user_id)
            
        Returns:
            Path to saved file
        """
        # Create output directory
        output_dir = Path("outputs/demo_data/profiles")
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate filename if not provided
        if filename is None:
            filename = f"profile_{profile.get('user_id', 'unknown')}"
        
        # Save as JSON
        json_path = output_dir / f"{filename}.json"
        with open(json_path, 'w') as f:
            json.dump(profile, f, indent=2)
        
        logger.info(f"Saved user profile to {json_path}")
        return str(json_path)
    
    def generate_demo_dataset(self, 
                            profile_type: str = "random", 
                            days: int = 30,
                            scenario: Optional[str] = None) -> Tuple[Dict[str, Any], pd.DataFrame, str]:
        """
        Generate a complete demo dataset with user profile and time series data.
        
        Args:
            profile_type: Type of user profile to generate
            days: Number of days of data to generate
            scenario: Optional scenario to apply (overtraining, recovery, etc.)
            
        Returns:
            Tuple of (user profile, data DataFrame, save path)
        """
        # Generate user profile
        profile = self.generate_user_profile(profile_type)
        
        # Generate time series data
        df = self.generate_time_series(days=days, user_profile=profile)
        
        # Apply scenario if specified
        if scenario:
            try:
                from simulator_engine import SimulatorEngine
                simulator = SimulatorEngine(user_profile=profile)
                simulator.set_simulation_parameters(scenario, days=days)
                df = simulator.apply_scenario(df)
                logger.info(f"Applied scenario '{scenario}' to demo data")
            except ImportError:
                logger.warning(f"Could not apply scenario '{scenario}': SimulatorEngine not available")
        
        # Save data and profile
        data_path = self.save_demo_data(df, f"{profile_type}_{profile['user_id']}")
        self.save_user_profile(profile)
        
        return profile, df, data_path


# Example usage
if __name__ == "__main__":
    # Initialize generator
    generator = DemoDataGenerator(seed=42)
    
    # Generate and save demo datasets
    for profile_type in ["athlete", "casual", "recovery_focused"]:
        profile, data, path = generator.generate_demo_dataset(
            profile_type=profile_type,
            days=14
        )
        print(f"Generated {profile_type} dataset with {len(data)} days of data")
        print(f"User: {profile['name']}, Goal: {profile['goals']['primary']}")
        print(f"Saved to: {path}")
        print("-" * 50)
