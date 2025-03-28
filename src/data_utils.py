"""
Data Utilities Module for Wearable Data Insight Generator

This module provides utility functions for data processing, normalization,
and transformation to support the Wearable Data Insight Generator pipeline.
"""

import logging
from typing import Dict, List, Optional, Union, Any

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def normalize_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize wearable data to a standard schema.
    
    Transforms various data formats into a consistent schema with the following columns:
    - timestamp: datetime
    - hrv: float (heart rate variability)
    - sleep_quality: float (0-100)
    - activity_level: float (0-100)
    - subjective_recovery: float (0-100)
    - notes: str (optional)
    
    Args:
        df: Input DataFrame with wearable data
        
    Returns:
        Normalized DataFrame with standardized columns
    """
    try:
        # Create a copy to avoid modifying the original
        normalized_df = df.copy()
        
        # Ensure timestamp column exists and is in datetime format
        if 'timestamp' in normalized_df.columns:
            normalized_df['timestamp'] = pd.to_datetime(normalized_df['timestamp'])
        elif 'date' in normalized_df.columns:
            normalized_df['timestamp'] = pd.to_datetime(normalized_df['date'])
        elif 'time' in normalized_df.columns:
            normalized_df['timestamp'] = pd.to_datetime(normalized_df['time'])
        else:
            # Create a timestamp if none exists
            normalized_df['timestamp'] = pd.to_datetime('now')
            logger.warning("No timestamp column found, using current time")
        
        # Normalize HRV (Heart Rate Variability)
        if 'hrv' in normalized_df.columns:
            normalized_df['hrv'] = pd.to_numeric(normalized_df['hrv'], errors='coerce')
        elif 'rmssd' in normalized_df.columns:  # Common HRV metric
            normalized_df['hrv'] = pd.to_numeric(normalized_df['rmssd'], errors='coerce')
        elif 'heart_rate_variability' in normalized_df.columns:
            normalized_df['hrv'] = pd.to_numeric(normalized_df['heart_rate_variability'], errors='coerce')
        else:
            normalized_df['hrv'] = np.nan
            logger.warning("No HRV column found, using NaN")
        
        # Normalize sleep quality
        if 'sleep_quality' in normalized_df.columns:
            normalized_df['sleep_quality'] = pd.to_numeric(normalized_df['sleep_quality'], errors='coerce')
        elif 'sleep_score' in normalized_df.columns:
            normalized_df['sleep_quality'] = pd.to_numeric(normalized_df['sleep_score'], errors='coerce')
        elif 'sleep_efficiency' in normalized_df.columns:
            normalized_df['sleep_quality'] = pd.to_numeric(normalized_df['sleep_efficiency'], errors='coerce')
        else:
            normalized_df['sleep_quality'] = np.nan
            logger.warning("No sleep quality column found, using NaN")
        
        # Normalize activity level
        if 'activity_level' in normalized_df.columns:
            normalized_df['activity_level'] = pd.to_numeric(normalized_df['activity_level'], errors='coerce')
        elif 'steps' in normalized_df.columns:
            # Convert steps to activity level (0-100)
            steps = pd.to_numeric(normalized_df['steps'], errors='coerce')
            normalized_df['activity_level'] = np.clip(steps / 150, 0, 100)  # Assuming 15000 steps = 100% activity
        elif 'activity_score' in normalized_df.columns:
            normalized_df['activity_level'] = pd.to_numeric(normalized_df['activity_score'], errors='coerce')
        else:
            normalized_df['activity_level'] = np.nan
            logger.warning("No activity level column found, using NaN")
        
        # Normalize subjective recovery
        if 'subjective_recovery' in normalized_df.columns:
            normalized_df['subjective_recovery'] = pd.to_numeric(normalized_df['subjective_recovery'], errors='coerce')
        elif 'recovery_score' in normalized_df.columns:
            normalized_df['subjective_recovery'] = pd.to_numeric(normalized_df['recovery_score'], errors='coerce')
        elif 'readiness' in normalized_df.columns:
            normalized_df['subjective_recovery'] = pd.to_numeric(normalized_df['readiness'], errors='coerce')
        else:
            normalized_df['subjective_recovery'] = np.nan
            logger.warning("No subjective recovery column found, using NaN")
        
        # Ensure notes column exists
        if 'notes' not in normalized_df.columns:
            normalized_df['notes'] = ""
        
        # Select only the normalized columns
        result_df = normalized_df[['timestamp', 'hrv', 'sleep_quality', 'activity_level', 'subjective_recovery', 'notes']]
        
        return result_df
        
    except Exception as e:
        logger.error(f"Error normalizing data: {str(e)}")
        # Return original dataframe if normalization fails
        if 'timestamp' not in df.columns:
            df['timestamp'] = pd.to_datetime('now')
        return df

def generate_synthetic_data(days: int = 30, user_type: str = "athlete") -> pd.DataFrame:
    """
    Generate synthetic wearable data for testing and demonstration.
    
    Args:
        days: Number of days of data to generate
        user_type: Type of user profile ("athlete", "casual", "stressed")
        
    Returns:
        DataFrame with synthetic wearable data
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Base parameters for different user types
    if user_type == "athlete":
        hrv_mean, hrv_std = 65, 10
        sleep_mean, sleep_std = 85, 8
        activity_mean, activity_std = 75, 15
        recovery_mean, recovery_std = 80, 12
    elif user_type == "stressed":
        hrv_mean, hrv_std = 45, 15
        sleep_mean, sleep_std = 60, 12
        activity_mean, activity_std = 50, 20
        recovery_mean, recovery_std = 55, 15
    else:  # casual
        hrv_mean, hrv_std = 55, 8
        sleep_mean, sleep_std = 70, 10
        activity_mean, activity_std = 60, 12
        recovery_mean, recovery_std = 65, 10
    
    # Generate random data with weekly patterns
    data = []
    for i, date in enumerate(date_range):
        # Add weekly patterns (e.g., lower recovery on Mondays, higher activity on weekends)
        day_of_week = date.dayofweek
        
        # Weekend effect
        weekend_factor = 1.0
        if day_of_week >= 5:  # Weekend
            weekend_factor = 1.2 if user_type == "athlete" else 0.8
        
        # Monday effect
        monday_factor = 1.0
        if day_of_week == 0:  # Monday
            monday_factor = 0.9
        
        # Generate values with patterns
        hrv = max(0, np.random.normal(hrv_mean * monday_factor, hrv_std))
        sleep_quality = min(100, max(0, np.random.normal(sleep_mean * monday_factor, sleep_std)))
        activity_level = min(100, max(0, np.random.normal(activity_mean * weekend_factor, activity_std)))
        subjective_recovery = min(100, max(0, np.random.normal(recovery_mean * monday_factor, recovery_std)))
        
        # Add some correlation between metrics
        if i > 0:
            prev_activity = data[i-1]['activity_level']
            # High activity yesterday reduces today's recovery and HRV
            if prev_activity > 80:
                hrv *= 0.9
                subjective_recovery *= 0.95
        
        data.append({
            'timestamp': date,
            'hrv': round(hrv, 1),
            'sleep_quality': round(sleep_quality, 1),
            'activity_level': round(activity_level, 1),
            'subjective_recovery': round(subjective_recovery, 1),
            'notes': ""
        })
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Add some notes for significant events
    significant_days = np.random.choice(range(len(df)), size=5, replace=False)
    events = [
        "Felt unusually tired after workout",
        "Slept poorly due to stress",
        "Great workout, feeling strong",
        "Rest day, focused on recovery",
        "Started new training program"
    ]
    
    for i, event in zip(significant_days, events):
        df.loc[i, 'notes'] = event
    
    return df

def detect_anomalies(df: pd.DataFrame, window_size: int = 7, std_threshold: float = 2.0) -> Dict[str, List[Dict]]:
    """
    Detect anomalies in wearable data metrics.
    
    Args:
        df: DataFrame with normalized wearable data
        window_size: Rolling window size for baseline calculation
        std_threshold: Number of standard deviations to consider as anomaly
        
    Returns:
        Dictionary with anomalies by metric type
    """
    metrics = ['hrv', 'sleep_quality', 'activity_level', 'subjective_recovery']
    anomalies = {metric: [] for metric in metrics}
    
    for metric in metrics:
        # Skip metrics with insufficient data
        if df[metric].isna().sum() > len(df) * 0.5:
            continue
            
        # Calculate rolling mean and std
        rolling_mean = df[metric].rolling(window=window_size, min_periods=3).mean()
        rolling_std = df[metric].rolling(window=window_size, min_periods=3).std()
        
        # Identify anomalies
        for i in range(window_size, len(df)):
            value = df[metric].iloc[i]
            mean = rolling_mean.iloc[i-1]  # Use previous window to avoid data leakage
            std = rolling_std.iloc[i-1]
            
            if std > 0:  # Avoid division by zero
                z_score = abs(value - mean) / std
                
                if z_score > std_threshold:
                    anomalies[metric].append({
                        'timestamp': df['timestamp'].iloc[i],
                        'value': value,
                        'baseline': mean,
                        'z_score': z_score,
                        'direction': 'high' if value > mean else 'low'
                    })
    
    return anomalies

def extract_trends(df: pd.DataFrame, metric: str, window_size: int = 7) -> Dict[str, Any]:
    """
    Extract trends from time series data.
    
    Args:
        df: DataFrame with normalized wearable data
        metric: Metric to analyze
        window_size: Window size for trend calculation
        
    Returns:
        Dictionary with trend information
    """
    if metric not in df.columns or df[metric].isna().sum() > len(df) * 0.5:
        return {'trend': 'unknown', 'slope': 0, 'confidence': 0}
    
    # Ensure data is sorted by timestamp
    df = df.sort_values('timestamp')
    
    # Calculate rolling statistics
    rolling_mean = df[metric].rolling(window=window_size, min_periods=3).mean()
    
    # Simple linear regression for trend
    if len(rolling_mean.dropna()) >= 3:
        y = rolling_mean.dropna().values
        x = np.arange(len(y))
        
        # Calculate slope using least squares
        slope, _ = np.polyfit(x, y, 1)
        
        # Determine trend direction and strength
        if abs(slope) < 0.1:
            trend = 'stable'
            confidence = 0.5
        else:
            trend = 'improving' if slope > 0 else 'declining'
            confidence = min(1.0, abs(slope))
        
        return {
            'trend': trend,
            'slope': round(slope, 3),
            'confidence': round(confidence, 2),
            'start_value': y[0],
            'end_value': y[-1],
            'change_pct': round((y[-1] - y[0]) / y[0] * 100 if y[0] != 0 else 0, 1)
        }
    
    return {'trend': 'unknown', 'slope': 0, 'confidence': 0}
