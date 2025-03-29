"""
Supabase Data Service Module for Wearable Data Insight Generator

This module provides a service layer for interacting with Supabase repositories
and integrating them with the application logic.
"""

from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import date, datetime, timedelta
import pandas as pd
import asyncio

from src.data.repositories.user_repository import UserRepository
from src.data.repositories.wearable_metrics_repository import WearableMetricsRepository
from src.data.repositories.insight_repository import InsightRepository

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SupabaseDataService:
    """Service for interacting with Supabase data repositories."""
    
    def __init__(self):
        """Initialize the service with repositories."""
        self.user_repo = UserRepository()
        self.metrics_repo = WearableMetricsRepository()
        self.insight_repo = InsightRepository()
    
    async def get_or_create_user(self, email: str, profile_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get a user by email or create a new one if not found.
        
        Args:
            email: User's email address
            profile_data: Dictionary with user profile data
            
        Returns:
            User record
        """
        user = await self.user_repo.find_by_email(email)
        
        if user:
            # Update user profile if needed
            updated_fields = {}
            for key, value in profile_data.items():
                if key in user and user[key] != value:
                    updated_fields[key] = value
            
            if updated_fields:
                user = await self.user_repo.update_user_profile(user["id"], updated_fields)
            
            return user
        
        # Create new user
        return await self.user_repo.create_user(
            email=email,
            first_name=profile_data.get("first_name"),
            last_name=profile_data.get("last_name"),
            birth_date=profile_data.get("birth_date"),
            gender=profile_data.get("gender"),
            height=profile_data.get("height"),
            weight=profile_data.get("weight"),
            activity_level=profile_data.get("activity_level", "moderate"),
            preferences=profile_data.get("preferences", {})
        )
    
    async def sync_metrics_data(self, user_id: str, metrics_data: List[Dict[str, Any]]) -> int:
        """
        Sync multiple days of metrics data for a user.
        
        Args:
            user_id: The ID of the user
            metrics_data: List of daily metrics dictionaries
            
        Returns:
            Number of records created or updated
        """
        sync_count = 0
        
        for daily_metrics in metrics_data:
            # Extract date from the metrics
            metrics_date = None
            if "date" in daily_metrics:
                if isinstance(daily_metrics["date"], str):
                    metrics_date = datetime.fromisoformat(daily_metrics["date"].replace("Z", "+00:00")).date()
                elif isinstance(daily_metrics["date"], (datetime, date)):
                    metrics_date = daily_metrics["date"].date() if isinstance(daily_metrics["date"], datetime) else daily_metrics["date"]
            
            if not metrics_date:
                logger.warning(f"Skipping metrics without valid date: {daily_metrics}")
                continue
            
            # Create or update metrics
            try:
                await self.metrics_repo.create_or_update_metrics(
                    user_id=user_id,
                    date_value=metrics_date,
                    device_id=daily_metrics.get("device_id"),
                    hrv_rmssd=daily_metrics.get("hrv_rmssd"),
                    resting_hr=daily_metrics.get("resting_hr"),
                    sleep_hours=daily_metrics.get("sleep_hours"),
                    sleep_quality=daily_metrics.get("sleep_quality"),
                    recovery_score=daily_metrics.get("recovery_score"),
                    strain=daily_metrics.get("strain"),
                    steps=daily_metrics.get("steps"),
                    active_minutes=daily_metrics.get("active_minutes"),
                    calories=daily_metrics.get("calories"),
                    raw_data=daily_metrics.get("raw_data")
                )
                sync_count += 1
            except Exception as e:
                logger.error(f"Error syncing metrics for date {metrics_date}: {str(e)}")
        
        return sync_count
    
    async def store_insights(self, user_id: str, insights: List[Dict[str, Any]]) -> int:
        """
        Store multiple insights for a user.
        
        Args:
            user_id: The ID of the user
            insights: List of insight dictionaries
            
        Returns:
            Number of insights created
        """
        insight_count = 0
        
        for insight in insights:
            # Extract date from the insight
            insight_date = None
            if "date" in insight:
                if isinstance(insight["date"], str):
                    insight_date = datetime.fromisoformat(insight["date"].replace("Z", "+00:00")).date()
                elif isinstance(insight["date"], (datetime, date)):
                    insight_date = insight["date"].date() if isinstance(insight["date"], datetime) else insight["date"]
            elif "timestamp" in insight:
                if isinstance(insight["timestamp"], str):
                    insight_date = datetime.fromisoformat(insight["timestamp"].replace("Z", "+00:00")).date()
                elif isinstance(insight["timestamp"], (datetime, date)):
                    insight_date = insight["timestamp"].date() if isinstance(insight["timestamp"], datetime) else insight["timestamp"]
            elif "day" in insight and isinstance(insight["day"], int):
                # If insight has a day index, use today - day
                insight_date = date.today() - timedelta(days=insight["day"])
            
            if not insight_date:
                logger.warning(f"Skipping insight without valid date: {insight}")
                continue
            
            # Extract insight type
            insight_type = insight.get("type", "general")
            
            # Extract insight title and summary
            title = insight.get("title", "Insight")
            
            if "summary" in insight:
                summary = insight["summary"]
            elif "content" in insight:
                summary = insight["content"]
            elif "message" in insight:
                summary = insight["message"]
            else:
                summary = str(insight)
            
            # Extract content
            content = insight.get("content", None)
            
            # Extract importance
            importance = insight.get("importance", 1)
            
            # Extract actionable flag
            is_actionable = insight.get("is_actionable", False)
            
            # Extract related metrics
            related_metrics = insight.get("related_metrics", None)
            
            # Create insight
            try:
                await self.insight_repo.create_insight(
                    user_id=user_id,
                    date_value=insight_date,
                    insight_type=insight_type,
                    title=title,
                    summary=summary,
                    content=content,
                    importance=importance,
                    is_actionable=is_actionable,
                    related_metrics=related_metrics
                )
                insight_count += 1
            except Exception as e:
                logger.error(f"Error creating insight for date {insight_date}: {str(e)}")
        
        return insight_count
    
    async def get_user_metrics_dataframe(self, user_id: str, days: int = 30) -> pd.DataFrame:
        """
        Get metrics for a user as a pandas DataFrame.
        
        Args:
            user_id: The ID of the user
            days: Number of days of data to retrieve
            
        Returns:
            DataFrame with metrics data
        """
        return await self.metrics_repo.get_metrics_as_dataframe(user_id, days)
    
    async def get_user_insights_dataframe(self, user_id: str, days: int = 30) -> pd.DataFrame:
        """
        Get insights for a user as a pandas DataFrame.
        
        Args:
            user_id: The ID of the user
            days: Number of days of data to retrieve
            
        Returns:
            DataFrame with insights data
        """
        return await self.insight_repo.get_insights_as_dataframe(user_id, days)
    
    async def get_user_data_for_visualization(self, user_id: str, days: int = 30) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """
        Get user metrics and insights data for visualization.
        
        Args:
            user_id: The ID of the user
            days: Number of days of data to retrieve
            
        Returns:
            Tuple of (metrics_dataframe, insights_list)
        """
        # Calculate date range
        end_date = date.today()
        start_date = end_date - timedelta(days=days)
        
        # Get metrics
        metrics = await self.metrics_repo.find_by_user_and_date_range(
            user_id=user_id,
            start_date=start_date,
            end_date=end_date
        )
        
        # Get insights
        insights = await self.insight_repo.find_by_user_and_date_range(
            user_id=user_id,
            start_date=start_date,
            end_date=end_date
        )
        
        # Convert metrics to DataFrame
        if metrics:
            metrics_df = pd.DataFrame(metrics)
            # Ensure date column is datetime
            metrics_df['date'] = pd.to_datetime(metrics_df['date'])
            metrics_df.set_index('date', inplace=True)
        else:
            metrics_df = pd.DataFrame()
        
        return metrics_df, insights
    
    async def connect_device(self, user_id: str, device_type: str, device_name: str, 
                            auth_token: str, refresh_token: str, token_expires_at: datetime,
                            sync_settings: Dict[str, Any] = None) -> str:
        """
        Connect a wearable device to a user account.
        
        Args:
            user_id: The ID of the user
            device_type: Type of device (e.g., 'fitbit', 'garmin')
            device_name: Name of the device
            auth_token: OAuth authentication token
            refresh_token: OAuth refresh token
            token_expires_at: Token expiration datetime
            sync_settings: Optional sync settings
            
        Returns:
            ID of the connected device
        """
        from src.config.supabase_config import get_supabase_client
        
        # Get Supabase client
        supabase = get_supabase_client()
        
        # Prepare device data
        device_data = {
            "user_id": user_id,
            "device_type": device_type,
            "device_name": device_name,
            "auth_token": auth_token,
            "refresh_token": refresh_token,
            "token_expires_at": token_expires_at.isoformat(),
            "is_active": True,
            "sync_settings": sync_settings or {}
        }
        
        # Insert device
        result = supabase.table("connected_devices").insert(device_data).execute()
        
        if not result.data:
            raise Exception("Failed to connect device")
        
        return result.data[0]["id"]
    
    async def list_user_devices(self, user_id: str) -> List[Dict[str, Any]]:
        """
        List all devices connected to a user account.
        
        Args:
            user_id: The ID of the user
            
        Returns:
            List of connected devices
        """
        from src.config.supabase_config import get_supabase_client
        
        # Get Supabase client
        supabase = get_supabase_client()
        
        # Query devices
        result = supabase.table("connected_devices").select("*").eq("user_id", user_id).execute()
        
        return result.data
    
    async def save_visualization(self, user_id: str, title: str, description: str,
                                start_date: date, end_date: date, 
                                metrics_included: List[str], insight_types_included: List[str],
                                html_content: str) -> str:
        """
        Save a visualization to Supabase.
        
        Args:
            user_id: The ID of the user
            title: Title of the visualization
            description: Description of the visualization
            start_date: Start date of the data
            end_date: End date of the data
            metrics_included: List of metrics included in the visualization
            insight_types_included: List of insight types included
            html_content: HTML content of the visualization
            
        Returns:
            ID of the saved visualization
        """
        from src.config.supabase_config import get_supabase_client
        
        # Get Supabase client
        supabase = get_supabase_client()
        
        # Prepare visualization data
        visualization_data = {
            "user_id": user_id,
            "title": title,
            "description": description,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "metrics_included": metrics_included,
            "insight_types_included": insight_types_included,
            "visualization_config": {
                "type": "interactive_timeline",
                "version": "1.0"
            },
            "html_content": html_content
        }
        
        # Insert visualization
        result = supabase.table("timeline_visualizations").insert(visualization_data).execute()
        
        if not result.data:
            raise Exception("Failed to save visualization")
        
        return result.data[0]["id"]
    
    async def get_user_health_overview(self, user_id: str) -> Dict[str, Any]:
        """
        Get a health overview for a user.
        
        Args:
            user_id: The ID of the user
            
        Returns:
            Dictionary with health overview data
        """
        # Get recent metrics
        metrics_df = await self.metrics_repo.get_metrics_as_dataframe(user_id, days=7)
        
        if metrics_df.empty:
            return {
                "status": "No recent data available",
                "metrics": {},
                "trends": {},
                "insights": []
            }
        
        # Calculate latest metrics
        latest_metrics = {}
        for metric in ["hrv_rmssd", "resting_hr", "sleep_hours", "sleep_quality", "recovery_score", "strain"]:
            if metric in metrics_df.columns:
                latest_metrics[metric] = metrics_df[metric].iloc[-1] if not metrics_df[metric].iloc[-1:].empty else None
        
        # Calculate trends
        trends = {}
        for metric in ["hrv_rmssd", "resting_hr", "sleep_quality", "recovery_score"]:
            if metric in metrics_df.columns and len(metrics_df) >= 3:
                recent_values = metrics_df[metric].dropna().tail(3).values
                if len(recent_values) >= 2:
                    trends[metric] = "improving" if recent_values[-1] > recent_values[0] else "declining"
                    # For metrics where lower is better
                    if metric == "resting_hr":
                        trends[metric] = "improving" if recent_values[-1] < recent_values[0] else "declining"
        
        # Get recent insights
        recent_insights = await self.insight_repo.get_recent_unread_insights(user_id, days=7)
        
        return {
            "status": "healthy" if latest_metrics.get("recovery_score", 0) > 70 else "needs_attention",
            "metrics": latest_metrics,
            "trends": trends,
            "insights": recent_insights
        }
    
    def run_async(self, coroutine):
        """
        Run an async coroutine and return the result.
        
        Args:
            coroutine: Async coroutine to run
            
        Returns:
            Result of the coroutine
        """
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Create a new event loop if the current one is already running
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                return new_loop.run_until_complete(coroutine)
            finally:
                new_loop.close()
                asyncio.set_event_loop(loop)
        else:
            return loop.run_until_complete(coroutine)
