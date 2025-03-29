"""
User Repository Module for Wearable Data Insight Generator

This module provides a repository for interacting with the users table in Supabase.
"""

from typing import Dict, List, Any, Optional
import logging
from datetime import date

from src.data.repositories.base_repository import BaseRepository

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UserRepository(BaseRepository):
    """Repository for user-related database operations."""
    
    def __init__(self):
        """Initialize the repository with the users table."""
        super().__init__("users")
    
    async def find_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """
        Find a user by email.
        
        Args:
            email: The email address to search for
            
        Returns:
            The user record if found, None otherwise
        """
        try:
            response = await self.table.select("*").eq("email", email).execute()
            data = response.data
            
            if data and len(data) > 0:
                return data[0]
            return None
        except Exception as e:
            logger.error(f"Error finding user by email {email}: {str(e)}")
            raise
    
    async def create_user(self, 
                         email: str, 
                         first_name: Optional[str] = None, 
                         last_name: Optional[str] = None,
                         birth_date: Optional[date] = None,
                         gender: Optional[str] = None,
                         height: Optional[float] = None,
                         weight: Optional[float] = None,
                         activity_level: str = "moderate",
                         avatar_url: Optional[str] = None,
                         preferences: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create a new user.
        
        Args:
            email: User's email address
            first_name: User's first name
            last_name: User's last name
            birth_date: User's birth date
            gender: User's gender
            height: User's height
            weight: User's weight
            activity_level: User's activity level
            avatar_url: URL to user's avatar
            preferences: User preferences as a dictionary
            
        Returns:
            The created user record
        """
        user_data = {
            "email": email,
            "activity_level": activity_level
        }
        
        if first_name:
            user_data["first_name"] = first_name
        
        if last_name:
            user_data["last_name"] = last_name
        
        if birth_date:
            user_data["birth_date"] = birth_date.isoformat()
        
        if gender:
            user_data["gender"] = gender
        
        if height:
            user_data["height"] = height
        
        if weight:
            user_data["weight"] = weight
        
        if avatar_url:
            user_data["avatar_url"] = avatar_url
        
        if preferences:
            user_data["preferences"] = preferences
        
        return await self.create(user_data)
    
    async def update_user_profile(self, user_id: str, profile_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update a user's profile.
        
        Args:
            user_id: The ID of the user to update
            profile_data: Dictionary of profile fields to update
            
        Returns:
            The updated user record
        """
        # Convert birth_date to ISO format if it's a date object
        if "birth_date" in profile_data and isinstance(profile_data["birth_date"], date):
            profile_data["birth_date"] = profile_data["birth_date"].isoformat()
        
        return await self.update(user_id, profile_data)
    
    async def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """
        Get a user's preferences.
        
        Args:
            user_id: The ID of the user
            
        Returns:
            The user's preferences as a dictionary
        """
        try:
            response = await self.table.select("preferences").eq("id", user_id).execute()
            data = response.data
            
            if data and len(data) > 0 and "preferences" in data[0]:
                return data[0]["preferences"]
            return {}
        except Exception as e:
            logger.error(f"Error getting preferences for user {user_id}: {str(e)}")
            raise
    
    async def update_user_preferences(self, user_id: str, preferences: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update a user's preferences.
        
        Args:
            user_id: The ID of the user
            preferences: The new preferences to set
            
        Returns:
            The updated user record
        """
        return await self.update(user_id, {"preferences": preferences})
