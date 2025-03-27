"""
Data Loader Module for Wearable LLM Insight Generator

This module handles the ingestion, normalization, and segmentation of time-series data
from wearable devices in CSV or JSON format.
"""

import pandas as pd
import json
import os
from typing import Dict, List, Union, Optional
from datetime import datetime, timedelta
import logging
import requests
from zipfile import ZipFile
from io import BytesIO

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataLoader:
    """Class to load and preprocess wearable data from various sources."""
    
    def __init__(self, data_dir: str = "../data/raw"):
        """
        Initialize the DataLoader.
        
        Args:
            data_dir: Directory containing raw data files
        """
        self.data_dir = data_dir
        self.supported_extensions = ['.csv', '.json']
    
    def list_available_files(self) -> List[str]:
        """List all supported data files in the data directory."""
        available_files = []
        
        for file in os.listdir(self.data_dir):
            file_path = os.path.join(self.data_dir, file)
            if os.path.isfile(file_path) and any(file.endswith(ext) for ext in self.supported_extensions):
                available_files.append(file)
                
        return available_files
    
    def load_csv(self, file_path: str, date_column: str = 'timestamp') -> pd.DataFrame:
        """
        Load data from a CSV file.
        
        Args:
            file_path: Path to the CSV file
            date_column: Name of the column containing timestamps
            
        Returns:
            DataFrame with the loaded data
        """
        try:
            df = pd.read_csv(file_path)
            
            # Convert timestamp column to datetime if it exists
            if date_column in df.columns:
                df[date_column] = pd.to_datetime(df[date_column])
                
            logger.info(f"Successfully loaded CSV file: {file_path}")
            return df
        
        except Exception as e:
            logger.error(f"Error loading CSV file {file_path}: {str(e)}")
            raise
    
    def load_json(self, file_path: str) -> Dict:
        """
        Load data from a JSON file.
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            Dictionary with the loaded data
        """
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            logger.info(f"Successfully loaded JSON file: {file_path}")
            return data
        
        except Exception as e:
            logger.error(f"Error loading JSON file {file_path}: {str(e)}")
            raise
    
    def load_file(self, file_name: str) -> Union[pd.DataFrame, Dict]:
        """
        Load data from a file based on its extension.
        
        Args:
            file_name: Path to the file to load
            
        Returns:
            DataFrame or Dict containing the loaded data
        """
        # Check if file exists
        if not os.path.exists(file_name):
            raise FileNotFoundError(f"File {file_name} not found")
        
        # Get file extension
        _, ext = os.path.splitext(file_name)
        
        # Load based on extension
        if ext.lower() == '.csv':
            return pd.read_csv(file_name)
        elif ext.lower() == '.json':
            with open(file_name, 'r') as f:
                return json.load(f)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")
    
    def download_diatrend_dataset(self, output_dir: str = '../data/raw/diatrend') -> str:
        """
        Download the DiaTrend dataset from Zenodo.
        
        Args:
            output_dir: Directory to save the dataset
            
        Returns:
            Path to the downloaded dataset
        """
        import requests
        from zipfile import ZipFile
        from io import BytesIO
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Check if dataset already exists
        if os.path.exists(os.path.join(output_dir, 'diatrend_data.csv')):
            print("DiaTrend dataset already downloaded.")
            return os.path.join(output_dir, 'diatrend_data.csv')
        
        # Download dataset
        print("Downloading DiaTrend dataset from Zenodo...")
        zenodo_url = "https://zenodo.org/record/7810922/files/diatrend_dataset.zip"
        
        try:
            response = requests.get(zenodo_url)
            response.raise_for_status()
            
            # Extract zip file
            with ZipFile(BytesIO(response.content)) as zip_file:
                zip_file.extractall(output_dir)
            
            # Find the main data file
            for root, _, files in os.walk(output_dir):
                for file in files:
                    if file.endswith('.csv') and 'glucose' in file.lower():
                        # Rename to a standard name
                        src_path = os.path.join(root, file)
                        dst_path = os.path.join(output_dir, 'diatrend_data.csv')
                        os.rename(src_path, dst_path)
                        return dst_path
            
            raise FileNotFoundError("Could not find DiaTrend data file in the downloaded archive.")
            
        except Exception as e:
            print(f"Error downloading DiaTrend dataset: {e}")
            print("Please download manually from: https://zenodo.org/record/7810922")
            print(f"And place the CSV files in: {output_dir}")
            return ""
    
    def load_diatrend_data(self, file_path: str = None) -> pd.DataFrame:
        """
        Load and preprocess DiaTrend dataset.
        
        Args:
            file_path: Path to the DiaTrend dataset CSV file
            
        Returns:
            DataFrame containing the preprocessed DiaTrend data
        """
        if file_path is None:
            # Try to find the dataset in the default location
            default_path = '../data/raw/diatrend/diatrend_data.csv'
            if os.path.exists(default_path):
                file_path = default_path
            else:
                # Try to download the dataset
                file_path = self.download_diatrend_dataset()
                if not file_path:
                    raise FileNotFoundError("DiaTrend dataset not found and could not be downloaded.")
        
        # Load the dataset
        df = pd.read_csv(file_path)
        
        # Convert timestamps to datetime
        if 'Time' in df.columns:
            df['Time'] = pd.to_datetime(df['Time'])
        
        # Process glucose and insulin data
        if 'GlucoseLevel' in df.columns:
            df['GlucoseLevel'] = pd.to_numeric(df['GlucoseLevel'], errors='coerce')
        
        if 'InsulinDose' in df.columns:
            df['InsulinDose'] = pd.to_numeric(df['InsulinDose'], errors='coerce')
        
        # Add day column for easier grouping
        if 'Time' in df.columns:
            df['Day'] = df['Time'].dt.date
            df['Hour'] = df['Time'].dt.hour
        
        return df
    
    def segment_diatrend_by_day(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Segment DiaTrend data by day.
        
        Args:
            df: DataFrame containing DiaTrend data
            
        Returns:
            Dictionary mapping days to DataFrames
        """
        if 'Day' not in df.columns:
            if 'Time' in df.columns:
                df['Day'] = df['Time'].dt.date
            else:
                raise ValueError("DataFrame must have 'Time' column to segment by day")
        
        # Group by day
        days = df['Day'].unique()
        result = {}
        
        for day in days:
            day_df = df[df['Day'] == day].copy()
            
            # Check if day has enough data
            if len(day_df) < 10:  # Arbitrary threshold, adjust as needed
                continue
                
            # Check if day has both glucose and insulin data
            if 'GlucoseLevel' in day_df.columns and 'InsulinDose' in day_df.columns:
                has_glucose = day_df['GlucoseLevel'].notna().any()
                has_insulin = day_df['InsulinDose'].notna().any()
                
                if not (has_glucose and has_insulin):
                    continue
            
            result[str(day)] = day_df
        
        return result
    
    def normalize_data(self, data: Union[pd.DataFrame, Dict], data_type: str) -> pd.DataFrame:
        """
        Normalize data to a standard format based on data type.
        
        Args:
            data: Raw data as DataFrame or Dictionary
            data_type: Type of data (e.g., 'hrv', 'activity', 'sleep')
            
        Returns:
            Normalized DataFrame
        """
        if data_type == 'hrv':
            return self._normalize_hrv_data(data)
        elif data_type == 'activity':
            return self._normalize_activity_data(data)
        elif data_type == 'sleep':
            return self._normalize_sleep_data(data)
        else:
            raise ValueError(f"Unsupported data type: {data_type}")
    
    def _normalize_hrv_data(self, data: Union[pd.DataFrame, Dict]) -> pd.DataFrame:
        """Normalize HRV data to a standard format."""
        if isinstance(data, dict):
            # Convert JSON structure to DataFrame
            df = pd.DataFrame(data.get('hrv_readings', []))
        else:
            df = data.copy()
        
        # Ensure required columns exist
        required_columns = ['timestamp', 'rmssd']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in HRV data")
        
        # Convert timestamp to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        return df
    
    def _normalize_activity_data(self, data: Union[pd.DataFrame, Dict]) -> pd.DataFrame:
        """Normalize activity data to a standard format."""
        if isinstance(data, dict):
            # Handle JSON structure
            if 'activities' in data:
                df = pd.DataFrame(data['activities'])
            else:
                # Try to convert flat dictionary to DataFrame
                df = pd.DataFrame([data])
        else:
            df = data.copy()
        
        # Ensure timestamp column exists and convert to datetime
        if 'timestamp' in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                df['timestamp'] = pd.to_datetime(df['timestamp'])
        elif 'start_time' in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df['start_time']):
                df['start_time'] = pd.to_datetime(df['start_time'])
            df['timestamp'] = df['start_time']
        
        # Sort by timestamp
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp')
        
        return df
    
    def _normalize_sleep_data(self, data: Union[pd.DataFrame, Dict]) -> pd.DataFrame:
        """Normalize sleep data to a standard format."""
        if isinstance(data, dict):
            # Handle JSON structure
            if 'sleep_sessions' in data:
                df = pd.DataFrame(data['sleep_sessions'])
            else:
                # Try to convert flat dictionary to DataFrame
                df = pd.DataFrame([data])
        else:
            df = data.copy()
        
        # Ensure required columns exist or create them
        if 'start_time' in df.columns and 'end_time' not in df.columns:
            if 'duration' in df.columns:
                # Convert duration to timedelta and calculate end_time
                if isinstance(df['duration'].iloc[0], (int, float)):
                    # Assume duration is in minutes
                    df['end_time'] = df['start_time'] + pd.to_timedelta(df['duration'], unit='m')
        
        # Convert timestamp columns to datetime
        datetime_columns = ['start_time', 'end_time', 'timestamp']
        for col in datetime_columns:
            if col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = pd.to_datetime(df[col])
        
        return df
    
    def segment_by_day(self, df: pd.DataFrame, timestamp_col: str = 'timestamp') -> Dict[str, pd.DataFrame]:
        """
        Segment data by day.
        
        Args:
            df: DataFrame to segment
            timestamp_col: Name of the timestamp column
            
        Returns:
            Dictionary mapping dates to DataFrames
        """
        if timestamp_col not in df.columns:
            raise ValueError(f"Timestamp column '{timestamp_col}' not found in DataFrame")
        
        # Ensure timestamp column is datetime
        if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        
        # Extract date from timestamp
        df['date'] = df[timestamp_col].dt.date
        
        # Group by date
        grouped = {str(date): group.drop('date', axis=1) for date, group in df.groupby('date')}
        
        return grouped
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean data by removing outliers and handling missing values.
        
        Args:
            df: DataFrame to clean
            
        Returns:
            Cleaned DataFrame
        """
        # Make a copy to avoid modifying the original
        cleaned_df = df.copy()
        
        # Handle missing values
        numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            # Fill missing values with median for numeric columns
            if cleaned_df[col].isna().any():
                median_val = cleaned_df[col].median()
                cleaned_df[col] = cleaned_df[col].fillna(median_val)
        
        # Remove outliers using IQR method for numeric columns
        for col in numeric_cols:
            Q1 = cleaned_df[col].quantile(0.25)
            Q3 = cleaned_df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Replace outliers with NaN and then fill with median
            mask = (cleaned_df[col] < lower_bound) | (cleaned_df[col] > upper_bound)
            if mask.any():
                cleaned_df.loc[mask, col] = None
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
        
        return cleaned_df


# Example usage
if __name__ == "__main__":
    loader = DataLoader()
    available_files = loader.list_available_files()
    print(f"Available files: {available_files}")
    
    if available_files:
        # Load the first available file
        sample_file = available_files[0]
        data = loader.load_file(sample_file)
        
        # Determine data type based on filename or content
        data_type = 'hrv' if 'hrv' in sample_file.lower() else 'activity' if 'activity' in sample_file.lower() else 'sleep'
        
        # Normalize and clean the data
        normalized_data = loader.normalize_data(data, data_type)
        cleaned_data = loader.clean_data(normalized_data)
        
        # Segment by day
        daily_data = loader.segment_by_day(cleaned_data)
        
        print(f"Loaded and processed {sample_file}")
        print(f"Number of days: {len(daily_data)}")
        for date, df in list(daily_data.items())[:2]:  # Show first two days
            print(f"Data for {date}: {len(df)} records")
