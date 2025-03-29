"""
Supabase Configuration Module for Wearable Data Insight Generator

This module handles the configuration and initialization of the Supabase client,
which is used for database operations, authentication, and storage.
"""

import os
from typing import Optional
from dotenv import load_dotenv
from supabase import create_client, Client
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class SupabaseConfig:
    """Singleton class to manage Supabase client configuration and initialization."""
    
    _instance: Optional['SupabaseConfig'] = None
    _client: Optional[Client] = None
    
    def __new__(cls):
        """Ensure only one instance of SupabaseConfig exists."""
        if cls._instance is None:
            cls._instance = super(SupabaseConfig, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize the Supabase client with credentials from environment variables."""
        try:
            supabase_url = os.getenv("SUPABASE_URL")
            supabase_key = os.getenv("SUPABASE_KEY")
            
            if not supabase_url or not supabase_key:
                logger.error("Supabase URL or key not found in environment variables")
                raise ValueError("Supabase URL or key not found in environment variables")
            
            self._client = create_client(supabase_url, supabase_key)
            logger.info("Supabase client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Supabase client: {str(e)}")
            raise
    
    @property
    def client(self) -> Client:
        """Get the Supabase client instance."""
        if self._client is None:
            self._initialize()
        return self._client

# Create a singleton instance for easy import
supabase = SupabaseConfig().client

def get_supabase_client() -> Client:
    """Get the Supabase client instance."""
    return supabase
