"""
Wearable Metrics Repository Module for Wearable Data Insight Generator

This module provides a repository for interacting with the wearable_metrics table in Supabase.
"""

from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import date, datetime, timedelta
import pandas as pd

from src.data.repositories.base_repository import BaseRepository

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WearableMetricsRepository(BaseRepository):
    """Repository for wearable metrics database operations."""
    
    def __init__(self):
        """Initialize the repository with the wearable_metrics table."""
        super().__init__("wearable_metrics")
    
    async def find_by_user_and_date(self, user_id: str, date_value: date) -> Optional[Dict[str, Any]]:
        """
        Find metrics for a specific user on a specific date.
        
        Args:
            user_id: The ID of the user
            date_value: The date to get metrics for
            
        Returns:
            The metrics record if found, None otherwise
        """
        try:
            response = await self.table.select("*").eq("user_id", user_id).eq("date", date_value.isoformat()).execute()
            data = response.data
            
            if data and len(data) > 0:
                return data[0]
            return None
        except Exception as e:
            logger.error(f"Error finding metrics for user {user_id} on date {date_value}: {str(e)}")
            raise
    
    async def find_by_user_and_date_range(self, user_id: str, start_date: date, end_date: date) -> List[Dict[str, Any]]:
        """
        Find metrics for a specific user within a date range.
        
        Args:
            user_id: The ID of the user
            start_date: The start date (inclusive)
            end_date: The end date (inclusive)
            
        Returns:
            List of metrics records
        """
        try:
            response = await self.table.select("*").eq("user_id", user_id).gte("date", start_date.isoformat()).lte("date", end_date.isoformat()).order("date").execute()
            return response.data
        except Exception as e:
            logger.error(f"Error finding metrics for user {user_id} between {start_date} and {end_date}: {str(e)}")
            raise
    
    async def create_or_update_metrics(self, 
                                     user_id: str, 
                                     date_value: date, 
                                     device_id: Optional[str] = None,
                                     hrv_rmssd: Optional[float] = None,
                                     resting_hr: Optional[float] = None,
                                     sleep_hours: Optional[float] = None,
                                     sleep_quality: Optional[float] = None,
                                     recovery_score: Optional[float] = None,
                                     strain: Optional[float] = None,
                                     steps: Optional[int] = None,
                                     active_minutes: Optional[int] = None,
                                     calories: Optional[float] = None,
                                     raw_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create or update metrics for a specific user on a specific date.
        
        Args:
            user_id: The ID of the user
            date_value: The date for the metrics
            device_id: The ID of the device that collected the metrics
            hrv_rmssd: Heart rate variability (RMSSD)
            resting_hr: Resting heart rate
            sleep_hours: Hours of sleep
            sleep_quality: Sleep quality score (0-1)
            recovery_score: Recovery score (0-100)
            strain: Strain or activity level
            steps: Number of steps
            active_minutes: Minutes of active time
            calories: Calories burned
            raw_data: Raw data from the device
            
        Returns:
            The created or updated metrics record
        """
        # Check if metrics already exist for this user and date
        existing_metrics = await self.find_by_user_and_date(user_id, date_value)
        
        metrics_data = {
            "user_id": user_id,
            "date": date_value.isoformat()
        }
        
        if device_id:
            metrics_data["device_id"] = device_id
        
        if hrv_rmssd is not None:
            metrics_data["hrv_rmssd"] = hrv_rmssd
        
        if resting_hr is not None:
            metrics_data["resting_hr"] = resting_hr
        
        if sleep_hours is not None:
            metrics_data["sleep_hours"] = sleep_hours
        
        if sleep_quality is not None:
            metrics_data["sleep_quality"] = sleep_quality
        
        if recovery_score is not None:
            metrics_data["recovery_score"] = recovery_score
        
        if strain is not None:
            metrics_data["strain"] = strain
        
        if steps is not None:
            metrics_data["steps"] = steps
        
        if active_minutes is not None:
            metrics_data["active_minutes"] = active_minutes
        
        if calories is not None:
            metrics_data["calories"] = calories
        
        if raw_data is not None:
            metrics_data["raw_data"] = raw_data
        
        if existing_metrics:
            return await self.update(existing_metrics["id"], metrics_data)
        else:
            return await self.create(metrics_data)
    
    async def get_metrics_as_dataframe(self, user_id: str, days: int = 30) -> pd.DataFrame:
        """
        Get metrics for a user as a pandas DataFrame.
        
        Args:
            user_id: The ID of the user
            days: Number of days of data to retrieve
            
        Returns:
            DataFrame with metrics data
        """
        end_date = date.today()
        start_date = end_date - timedelta(days=days-1)
        
        metrics = await self.find_by_user_and_date_range(user_id, start_date, end_date)
        
        if not metrics:
            return pd.DataFrame()
        
        df = pd.DataFrame(metrics)
        
        # Convert date strings to datetime objects
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
        
        return df
    
    async def get_metric_averages(self, user_id: str, metric_name: str, period_days: int = 7) -> Tuple[float, float]:
        """
        Get average values for a specific metric over different time periods.
        
        Args:
            user_id: The ID of the user
            metric_name: The name of the metric to average
            period_days: Number of days for the recent period
            
        Returns:
            Tuple of (recent_average, overall_average)
        """
        today = date.today()
        recent_start = today - timedelta(days=period_days)
        
        # Get all metrics for this user
        all_metrics = await self.find_by_user_and_date_range(user_id, today - timedelta(days=365), today)
        
        if not all_metrics:
            return (0.0, 0.0)
        
        # Convert to DataFrame
        df = pd.DataFrame(all_metrics)
        
        # Convert date strings to datetime objects
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        
        # Calculate averages
        if metric_name in df.columns:
            # Recent average
            recent_df = df[df['date'] >= pd.Timestamp(recent_start)]
            recent_avg = recent_df[metric_name].mean() if not recent_df.empty else 0.0
            
            # Overall average
            overall_avg = df[metric_name].mean() if not df.empty else 0.0
            
            return (recent_avg, overall_avg)
        
        return (0.0, 0.0)
