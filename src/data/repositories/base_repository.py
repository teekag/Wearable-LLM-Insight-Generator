"""
Base Repository Module for Wearable Data Insight Generator

This module provides a base repository class for interacting with Supabase tables.
"""

from typing import Dict, List, Any, Optional, TypeVar, Generic, Union
import logging
from supabase import Client

from src.config.supabase_config import get_supabase_client

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

T = TypeVar('T')

class BaseRepository(Generic[T]):
    """Base repository class for Supabase table operations."""
    
    def __init__(self, table_name: str):
        """
        Initialize the repository with a table name.
        
        Args:
            table_name: Name of the Supabase table
        """
        self.table_name = table_name
        self._client = get_supabase_client()
    
    @property
    def client(self) -> Client:
        """Get the Supabase client instance."""
        return self._client
    
    @property
    def table(self):
        """Get the table reference."""
        return self.client.table(self.table_name)
    
    async def find_by_id(self, id: str) -> Optional[T]:
        """
        Find a record by its ID.
        
        Args:
            id: The ID of the record to find
            
        Returns:
            The record if found, None otherwise
        """
        try:
            response = await self.table.select("*").eq("id", id).execute()
            data = response.data
            
            if data and len(data) > 0:
                return data[0]
            return None
        except Exception as e:
            logger.error(f"Error finding record by ID {id}: {str(e)}")
            raise
    
    async def find_all(self, limit: int = 100, offset: int = 0) -> List[T]:
        """
        Find all records with pagination.
        
        Args:
            limit: Maximum number of records to return
            offset: Number of records to skip
            
        Returns:
            List of records
        """
        try:
            response = await self.table.select("*").range(offset, offset + limit - 1).execute()
            return response.data
        except Exception as e:
            logger.error(f"Error finding all records: {str(e)}")
            raise
    
    async def find_by(self, filters: Dict[str, Any], limit: int = 100) -> List[T]:
        """
        Find records by filter criteria.
        
        Args:
            filters: Dictionary of field-value pairs to filter by
            limit: Maximum number of records to return
            
        Returns:
            List of matching records
        """
        try:
            query = self.table.select("*")
            
            for field, value in filters.items():
                query = query.eq(field, value)
            
            response = await query.limit(limit).execute()
            return response.data
        except Exception as e:
            logger.error(f"Error finding records by filters {filters}: {str(e)}")
            raise
    
    async def create(self, data: Dict[str, Any]) -> T:
        """
        Create a new record.
        
        Args:
            data: Dictionary of field-value pairs for the new record
            
        Returns:
            The created record
        """
        try:
            response = await self.table.insert(data).execute()
            
            if response.data and len(response.data) > 0:
                return response.data[0]
            
            raise ValueError("Failed to create record")
        except Exception as e:
            logger.error(f"Error creating record: {str(e)}")
            raise
    
    async def update(self, id: str, data: Dict[str, Any]) -> T:
        """
        Update an existing record.
        
        Args:
            id: The ID of the record to update
            data: Dictionary of field-value pairs to update
            
        Returns:
            The updated record
        """
        try:
            response = await self.table.update(data).eq("id", id).execute()
            
            if response.data and len(response.data) > 0:
                return response.data[0]
            
            raise ValueError(f"Failed to update record with ID {id}")
        except Exception as e:
            logger.error(f"Error updating record with ID {id}: {str(e)}")
            raise
    
    async def delete(self, id: str) -> bool:
        """
        Delete a record by its ID.
        
        Args:
            id: The ID of the record to delete
            
        Returns:
            True if the record was deleted, False otherwise
        """
        try:
            response = await self.table.delete().eq("id", id).execute()
            
            if response.data is not None:
                return True
            
            return False
        except Exception as e:
            logger.error(f"Error deleting record with ID {id}: {str(e)}")
            raise
    
    async def count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """
        Count records, optionally filtered.
        
        Args:
            filters: Optional dictionary of field-value pairs to filter by
            
        Returns:
            Count of matching records
        """
        try:
            query = self.table.select("id", count="exact")
            
            if filters:
                for field, value in filters.items():
                    query = query.eq(field, value)
            
            response = await query.execute()
            return response.count
        except Exception as e:
            logger.error(f"Error counting records: {str(e)}")
            raise
