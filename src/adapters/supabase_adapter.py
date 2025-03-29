"""
Supabase Adapter Module for Wearable Data Insight Generator

This module provides adapter classes to integrate Supabase with existing components
of the Wearable Data Insight Generator.
"""

from typing import Dict, List, Any, Optional, Tuple
import logging
import pandas as pd
from datetime import date, datetime, timedelta

from src.services.supabase_data_service import SupabaseDataService
from src.simulator_engine import SimulatorEngine
from src.insight_engine import InsightEngine

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SupabaseSimulatorAdapter:
    """Adapter to integrate SimulatorEngine with Supabase."""
    
    def __init__(self, user_id: str):
        """
        Initialize the adapter.
        
        Args:
            user_id: The ID of the user in Supabase
        """
        self.user_id = user_id
        self.data_service = SupabaseDataService()
        self.simulator = SimulatorEngine()
    
    def initialize_simulator_from_supabase(self):
        """Initialize the simulator with user data from Supabase."""
        # Get user profile
        user = self.data_service.run_async(
            self.data_service.user_repo.find_by_id(self.user_id)
        )
        
        if not user:
            logger.error(f"User {self.user_id} not found in Supabase")
            return False
        
        # Create user profile for simulator
        user_profile = {
            "name": f"{user.get('first_name', '')} {user.get('last_name', '')}".strip() or "User",
            "age": self._calculate_age(user.get('birth_date')) if user.get('birth_date') else 30,
            "gender": user.get('gender', 'unknown'),
            "height": user.get('height', 170),
            "weight": user.get('weight', 70),
            "activity_level": user.get('activity_level', 'moderate')
        }
        
        # Set user profile in simulator
        self.simulator.set_user_profile(user_profile)
        
        return True
    
    def _calculate_age(self, birth_date_str: str) -> int:
        """Calculate age from birth date string."""
        if not birth_date_str:
            return 30
        
        try:
            birth_date = date.fromisoformat(birth_date_str)
            today = date.today()
            age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
            return age
        except Exception as e:
            logger.error(f"Error calculating age from birth date {birth_date_str}: {str(e)}")
            return 30
    
    def run_simulation_and_store_results(self, scenario_type: str, days: int = 30) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """
        Run a simulation and store the results in Supabase.
        
        Args:
            scenario_type: Type of scenario to simulate
            days: Number of days to simulate
            
        Returns:
            Tuple of (metrics_dataframe, insights_list)
        """
        # Set simulation parameters
        self.simulator.set_simulation_parameters(scenario_type, days)
        
        # Run simulation
        metrics_df, insights = self.simulator.run_simulation()
        
        # Convert DataFrame to list of dictionaries for storage
        metrics_records = metrics_df.to_dict('records')
        
        # Store metrics in Supabase
        self.data_service.run_async(
            self.data_service.sync_metrics_data(self.user_id, metrics_records)
        )
        
        # Store insights in Supabase
        self.data_service.run_async(
            self.data_service.store_insights(self.user_id, insights)
        )
        
        return metrics_df, insights
    
    def get_simulation_results(self, days: int = 30) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """
        Get simulation results from Supabase.
        
        Args:
            days: Number of days of data to retrieve
            
        Returns:
            Tuple of (metrics_dataframe, insights_list)
        """
        return self.data_service.run_async(
            self.data_service.get_user_data_for_visualization(self.user_id, days)
        )


class SupabaseInsightAdapter:
    """Adapter to integrate InsightEngine with Supabase."""
    
    def __init__(self, user_id: str):
        """
        Initialize the adapter.
        
        Args:
            user_id: The ID of the user in Supabase
        """
        self.user_id = user_id
        self.data_service = SupabaseDataService()
        self.insight_engine = InsightEngine()
    
    def generate_insights_from_supabase_data(self, days: int = 30) -> List[Dict[str, Any]]:
        """
        Generate insights from Supabase data.
        
        Args:
            days: Number of days of data to analyze
            
        Returns:
            List of generated insights
        """
        # Get metrics data from Supabase
        metrics_df = self.data_service.run_async(
            self.data_service.get_user_metrics_dataframe(self.user_id, days)
        )
        
        if metrics_df.empty:
            logger.warning(f"No metrics data found for user {self.user_id}")
            return []
        
        # Generate insights
        insights = self.insight_engine.generate_insights(metrics_df)
        
        # Store insights in Supabase
        self.data_service.run_async(
            self.data_service.store_insights(self.user_id, insights)
        )
        
        return insights
    
    def get_recent_insights(self, days: int = 7) -> List[Dict[str, Any]]:
        """
        Get recent insights from Supabase.
        
        Args:
            days: Number of days of insights to retrieve
            
        Returns:
            List of recent insights
        """
        end_date = date.today()
        start_date = end_date - timedelta(days=days-1)
        
        return self.data_service.run_async(
            self.data_service.insight_repo.find_by_user_and_date_range(self.user_id, start_date, end_date)
        )
    
    def get_insights_by_type(self, insight_type: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get insights of a specific type from Supabase.
        
        Args:
            insight_type: Type of insights to retrieve
            limit: Maximum number of insights to retrieve
            
        Returns:
            List of insights of the specified type
        """
        return self.data_service.run_async(
            self.data_service.insight_repo.find_by_user_and_type(self.user_id, insight_type, limit)
        )
    
    def mark_insight_as_read(self, insight_id: str) -> bool:
        """
        Mark an insight as read in Supabase.
        
        Args:
            insight_id: ID of the insight to mark as read
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.data_service.run_async(
                self.data_service.insight_repo.mark_as_read(insight_id)
            )
            return True
        except Exception as e:
            logger.error(f"Error marking insight {insight_id} as read: {str(e)}")
            return False
