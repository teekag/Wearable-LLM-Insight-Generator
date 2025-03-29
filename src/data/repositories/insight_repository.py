"""
Insight Repository Module for Wearable Data Insight Generator

This module provides a repository for interacting with the insights table in Supabase.
"""

from typing import Dict, List, Any, Optional
import logging
from datetime import date, datetime, timedelta
import pandas as pd

from src.data.repositories.base_repository import BaseRepository

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class InsightRepository(BaseRepository):
    """Repository for insight database operations."""
    
    def __init__(self):
        """Initialize the repository with the insights table."""
        super().__init__("insights")
    
    async def find_by_user_and_date(self, user_id: str, date_value: date) -> List[Dict[str, Any]]:
        """
        Find insights for a specific user on a specific date.
        
        Args:
            user_id: The ID of the user
            date_value: The date to get insights for
            
        Returns:
            List of insight records
        """
        try:
            response = await self.table.select("*").eq("user_id", user_id).eq("date", date_value.isoformat()).execute()
            return response.data
        except Exception as e:
            logger.error(f"Error finding insights for user {user_id} on date {date_value}: {str(e)}")
            raise
    
    async def find_by_user_and_date_range(self, user_id: str, start_date: date, end_date: date) -> List[Dict[str, Any]]:
        """
        Find insights for a specific user within a date range.
        
        Args:
            user_id: The ID of the user
            start_date: The start date (inclusive)
            end_date: The end date (inclusive)
            
        Returns:
            List of insight records
        """
        try:
            response = await self.table.select("*").eq("user_id", user_id).gte("date", start_date.isoformat()).lte("date", end_date.isoformat()).order("date").execute()
            return response.data
        except Exception as e:
            logger.error(f"Error finding insights for user {user_id} between {start_date} and {end_date}: {str(e)}")
            raise
    
    async def find_by_user_and_type(self, user_id: str, insight_type: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Find insights for a specific user of a specific type.
        
        Args:
            user_id: The ID of the user
            insight_type: The type of insights to find
            limit: Maximum number of insights to return
            
        Returns:
            List of insight records
        """
        try:
            response = await self.table.select("*").eq("user_id", user_id).eq("type", insight_type).order("date", desc=True).limit(limit).execute()
            return response.data
        except Exception as e:
            logger.error(f"Error finding insights for user {user_id} of type {insight_type}: {str(e)}")
            raise
    
    async def create_insight(self, 
                           user_id: str, 
                           date_value: date, 
                           insight_type: str,
                           title: str,
                           summary: str,
                           content: Optional[str] = None,
                           importance: int = 1,
                           is_actionable: bool = False,
                           related_metrics: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create a new insight.
        
        Args:
            user_id: The ID of the user
            date_value: The date for the insight
            insight_type: The type of insight
            title: The insight title
            summary: A brief summary of the insight
            content: Detailed content of the insight
            importance: Importance level (1-5)
            is_actionable: Whether the insight is actionable
            related_metrics: Related metrics data
            
        Returns:
            The created insight record
        """
        insight_data = {
            "user_id": user_id,
            "date": date_value.isoformat(),
            "type": insight_type,
            "title": title,
            "summary": summary,
            "importance": importance,
            "is_actionable": is_actionable
        }
        
        if content:
            insight_data["content"] = content
        
        if related_metrics:
            insight_data["related_metrics"] = related_metrics
        
        return await self.create(insight_data)
    
    async def mark_as_read(self, insight_id: str) -> Dict[str, Any]:
        """
        Mark an insight as read.
        
        Args:
            insight_id: The ID of the insight
            
        Returns:
            The updated insight record
        """
        return await self.update(insight_id, {"is_read": True})
    
    async def mark_action_taken(self, insight_id: str) -> Dict[str, Any]:
        """
        Mark that action was taken on an actionable insight.
        
        Args:
            insight_id: The ID of the insight
            
        Returns:
            The updated insight record
        """
        return await self.update(insight_id, {"action_taken": True})
    
    async def get_insights_as_dataframe(self, user_id: str, days: int = 30) -> pd.DataFrame:
        """
        Get insights for a user as a pandas DataFrame.
        
        Args:
            user_id: The ID of the user
            days: Number of days of data to retrieve
            
        Returns:
            DataFrame with insights data
        """
        end_date = date.today()
        start_date = end_date - timedelta(days=days-1)
        
        insights = await self.find_by_user_and_date_range(user_id, start_date, end_date)
        
        if not insights:
            return pd.DataFrame()
        
        df = pd.DataFrame(insights)
        
        # Convert date strings to datetime objects
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
        
        return df
    
    async def get_recent_unread_insights(self, user_id: str, days: int = 7) -> List[Dict[str, Any]]:
        """
        Get recent unread insights for a user.
        
        Args:
            user_id: The ID of the user
            days: Number of days to look back
            
        Returns:
            List of unread insight records
        """
        end_date = date.today()
        start_date = end_date - timedelta(days=days-1)
        
        try:
            response = await self.table.select("*").eq("user_id", user_id).eq("is_read", False).gte("date", start_date.isoformat()).lte("date", end_date.isoformat()).order("date", desc=True).execute()
            return response.data
        except Exception as e:
            logger.error(f"Error finding unread insights for user {user_id}: {str(e)}")
            raise
