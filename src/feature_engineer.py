"""
Feature Engineering Module for Wearable LLM Insight Generator

This module extracts meaningful features from wearable time-series data
including HRV metrics, stress windows, activity patterns, and training load.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
from scipy import stats

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Class to extract features from wearable data."""
    
    def __init__(self):
        """Initialize the FeatureEngineer."""
        pass
    
    def extract_hrv_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Extract HRV features from time-series data.
        
        Args:
            df: DataFrame containing HRV data with at least 'rmssd' column
            
        Returns:
            Dictionary of HRV features
        """
        features = {}
        
        # Basic statistics
        if 'rmssd' in df.columns:
            features['rmssd_mean'] = df['rmssd'].mean()
            features['rmssd_median'] = df['rmssd'].median()
            features['rmssd_std'] = df['rmssd'].std()
            features['rmssd_min'] = df['rmssd'].min()
            features['rmssd_max'] = df['rmssd'].max()
            
            # Calculate pNN50 if NN intervals are available
            if 'nn_intervals' in df.columns:
                nn_intervals = np.array(df['nn_intervals'].tolist())
                nn_diffs = np.diff(nn_intervals)
                features['pnn50'] = 100 * np.sum(np.abs(nn_diffs) > 50) / len(nn_diffs)
            
            # Calculate SDNN if available
            if 'nn_intervals' in df.columns:
                nn_intervals = np.array(df['nn_intervals'].tolist())
                features['sdnn'] = np.std(nn_intervals)
            
            # Time-of-day analysis
            if 'timestamp' in df.columns:
                df['hour'] = df['timestamp'].dt.hour
                
                # Morning HRV (5-9 AM)
                morning_mask = (df['hour'] >= 5) & (df['hour'] < 9)
                if morning_mask.any():
                    features['morning_rmssd'] = df.loc[morning_mask, 'rmssd'].mean()
                
                # Evening HRV (8-11 PM)
                evening_mask = (df['hour'] >= 20) & (df['hour'] < 23)
                if evening_mask.any():
                    features['evening_rmssd'] = df.loc[evening_mask, 'rmssd'].mean()
                
                # Calculate HRV trend throughout the day
                hourly_hrv = df.groupby('hour')['rmssd'].mean()
                if len(hourly_hrv) > 3:  # Need at least a few hours of data
                    slope, _, _, _, _ = stats.linregress(hourly_hrv.index, hourly_hrv.values)
                    features['hrv_trend_slope'] = slope
        
        return features
    
    def extract_stress_windows(self, df: pd.DataFrame, 
                              low_hrv_threshold: Optional[float] = None,
                              window_size: int = 30) -> List[Dict]:
        """
        Identify stress windows based on sustained low HRV.
        
        Args:
            df: DataFrame with HRV data
            low_hrv_threshold: Threshold for low HRV (if None, uses 25th percentile)
            window_size: Minimum duration in minutes for a stress window
            
        Returns:
            List of stress windows with start/end times and metrics
        """
        if 'rmssd' not in df.columns or 'timestamp' not in df.columns:
            logger.warning("Required columns 'rmssd' and 'timestamp' not found for stress window analysis")
            return []
        
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        # Determine low HRV threshold if not provided
        if low_hrv_threshold is None:
            low_hrv_threshold = df['rmssd'].quantile(0.25)
        
        # Identify low HRV points
        df['is_low_hrv'] = df['rmssd'] < low_hrv_threshold
        
        # Find continuous windows of low HRV
        df['low_hrv_group'] = (df['is_low_hrv'] != df['is_low_hrv'].shift()).cumsum()
        low_hrv_groups = df[df['is_low_hrv']].groupby('low_hrv_group')
        
        stress_windows = []
        
        for _, group in low_hrv_groups:
            start_time = group['timestamp'].min()
            end_time = group['timestamp'].max()
            duration_min = (end_time - start_time).total_seconds() / 60
            
            # Only include windows that meet the minimum duration
            if duration_min >= window_size:
                window = {
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration_min': duration_min,
                    'avg_rmssd': group['rmssd'].mean(),
                    'min_rmssd': group['rmssd'].min()
                }
                stress_windows.append(window)
        
        return stress_windows
    
    def extract_activity_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Extract activity features from time-series data.
        
        Args:
            df: DataFrame containing activity data
            
        Returns:
            Dictionary of activity features
        """
        features = {}
        
        # Steps analysis
        if 'steps' in df.columns:
            features['total_steps'] = df['steps'].sum()
            features['avg_hourly_steps'] = df['steps'].mean() * 24  # Assuming hourly data
            
            # Calculate active minutes (periods with significant step count)
            if 'timestamp' in df.columns:
                active_threshold = df['steps'].quantile(0.7)  # Adjust threshold as needed
                active_periods = df[df['steps'] > active_threshold]
                features['active_minutes'] = len(active_periods)
        
        # Heart rate analysis during activity
        if 'heart_rate' in df.columns:
            features['avg_heart_rate'] = df['heart_rate'].mean()
            features['max_heart_rate'] = df['heart_rate'].max()
            
            # Calculate heart rate zones if possible
            if df['heart_rate'].max() > 0:
                # Simplified heart rate zones
                max_hr_est = 220 - 30  # Assuming 30 years old; adjust as needed
                
                zone1 = (df['heart_rate'] >= 0.5 * max_hr_est) & (df['heart_rate'] < 0.6 * max_hr_est)
                zone2 = (df['heart_rate'] >= 0.6 * max_hr_est) & (df['heart_rate'] < 0.7 * max_hr_est)
                zone3 = (df['heart_rate'] >= 0.7 * max_hr_est) & (df['heart_rate'] < 0.8 * max_hr_est)
                zone4 = (df['heart_rate'] >= 0.8 * max_hr_est) & (df['heart_rate'] < 0.9 * max_hr_est)
                zone5 = df['heart_rate'] >= 0.9 * max_hr_est
                
                features['time_in_zone1'] = zone1.sum()
                features['time_in_zone2'] = zone2.sum()
                features['time_in_zone3'] = zone3.sum()
                features['time_in_zone4'] = zone4.sum()
                features['time_in_zone5'] = zone5.sum()
        
        # Intensity analysis
        if 'intensity' in df.columns:
            features['avg_intensity'] = df['intensity'].mean()
            features['max_intensity'] = df['intensity'].max()
            
            # Calculate time spent in different intensity levels
            low_intensity = (df['intensity'] > 0) & (df['intensity'] <= 3)
            medium_intensity = (df['intensity'] > 3) & (df['intensity'] <= 7)
            high_intensity = df['intensity'] > 7
            
            features['low_intensity_minutes'] = low_intensity.sum()
            features['medium_intensity_minutes'] = medium_intensity.sum()
            features['high_intensity_minutes'] = high_intensity.sum()
        
        return features
    
    def calculate_training_load(self, activity_data: Dict) -> float:
        """
        Calculate training load based on activity data.
        
        Args:
            activity_data: Dictionary containing activity data
            
        Returns:
            Training load score
        """
        # Simple training load calculation
        # In a real system, this would be more sophisticated
        intensity = activity_data.get('intensity', 0)
        duration = activity_data.get('duration_minutes', 0)
        
        # Training load = intensity * duration
        training_load = intensity * duration
        
        return training_load
    
    def extract_sleep_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Extract sleep features from sleep data.
        
        Args:
            df: DataFrame containing sleep data
            
        Returns:
            Dictionary of sleep features
        """
        features = {}
        
        # Basic sleep duration
        if 'start_time' in df.columns and 'end_time' in df.columns:
            df['sleep_duration'] = (df['end_time'] - df['start_time']).dt.total_seconds() / 3600  # hours
            features['total_sleep_hours'] = df['sleep_duration'].sum()
            
            # Sleep timing
            features['sleep_start'] = df['start_time'].min()
            features['sleep_end'] = df['end_time'].max()
            
            # Sleep consistency
            if len(df) > 1:
                start_times = df['start_time'].dt.hour + df['start_time'].dt.minute / 60
                features['sleep_start_consistency'] = start_times.std()
        
        # Sleep stages if available
        sleep_stages = ['deep_sleep', 'light_sleep', 'rem_sleep', 'awake']
        for stage in sleep_stages:
            if stage in df.columns:
                features[f'{stage}_minutes'] = df[stage].sum()
                if features.get('total_sleep_hours', 0) > 0:
                    features[f'{stage}_percentage'] = df[stage].sum() / (features['total_sleep_hours'] * 60) * 100
        
        # Sleep quality if available
        if 'quality' in df.columns:
            features['sleep_quality'] = df['quality'].mean()
        
        # Calculate sleep efficiency if we have awake time
        if 'awake' in df.columns and features.get('total_sleep_hours', 0) > 0:
            total_minutes = features['total_sleep_hours'] * 60
            awake_minutes = features.get('awake_minutes', 0)
            features['sleep_efficiency'] = (total_minutes - awake_minutes) / total_minutes * 100
        
        return features
    
    def combine_features(self, hrv_features: Dict[str, float],
                        activity_features: Dict[str, float],
                        sleep_features: Dict[str, float],
                        training_load: Dict[str, float]) -> Dict[str, float]:
        """
        Combine all features into a single dictionary.
        
        Args:
            hrv_features: HRV features
            activity_features: Activity features
            sleep_features: Sleep features
            training_load: Training load metrics
            
        Returns:
            Combined dictionary of all features
        """
        combined = {}
        
        # Add all features with prefixes
        for key, value in hrv_features.items():
            combined[f'hrv_{key}'] = value
            
        for key, value in activity_features.items():
            combined[f'activity_{key}'] = value
            
        for key, value in sleep_features.items():
            combined[f'sleep_{key}'] = value
            
        # Add training load metrics directly
        combined.update(training_load)
        
        # Calculate derived cross-domain features
        
        # Recovery score: combination of HRV, sleep quality and training load
        if 'hrv_rmssd_mean' in combined and 'sleep_total_sleep_hours' in combined:
            # Normalize HRV (higher is better)
            norm_hrv = min(1.0, combined.get('hrv_rmssd_mean', 0) / 100)
            
            # Normalize sleep (higher is better)
            norm_sleep = min(1.0, combined.get('sleep_total_sleep_hours', 0) / 8)
            
            # Normalize training load (lower ratio is better for recovery)
            load_ratio = combined.get('load_ratio', 1.0)
            norm_load = max(0, min(1.0, 2 - load_ratio)) if load_ratio > 0 else 0.5
            
            # Calculate weighted recovery score
            combined['recovery_score'] = (0.4 * norm_hrv + 0.4 * norm_sleep + 0.2 * norm_load) * 100
        
        return combined
    
    def extract_diatrend_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Extract features from DiaTrend dataset.
        
        Args:
            df: DataFrame containing DiaTrend data for a single day
            
        Returns:
            Dictionary of extracted features
        """
        features = {}
        
        # Basic glucose statistics
        if 'GlucoseLevel' in df.columns:
            glucose_values = df['GlucoseLevel'].dropna()
            if len(glucose_values) > 0:
                features['avg_glucose'] = glucose_values.mean()
                features['min_glucose'] = glucose_values.min()
                features['max_glucose'] = glucose_values.max()
                features['glucose_range'] = features['max_glucose'] - features['min_glucose']
                features['glucose_std'] = glucose_values.std()
                
                # Count hypoglycemic and hyperglycemic events
                hypo_threshold = 70  # mg/dL
                hyper_threshold = 180  # mg/dL
                features['hypo_events'] = (glucose_values < hypo_threshold).sum()
                features['hyper_events'] = (glucose_values > hyper_threshold).sum()
                
                # Time in range (70-180 mg/dL)
                in_range = ((glucose_values >= hypo_threshold) & (glucose_values <= hyper_threshold)).sum()
                features['time_in_range_percent'] = (in_range / len(glucose_values)) * 100
        
        # Insulin statistics
        if 'InsulinDose' in df.columns:
            insulin_doses = df['InsulinDose'].dropna()
            if len(insulin_doses) > 0:
                features['total_daily_insulin'] = insulin_doses.sum()
                features['max_insulin_dose'] = insulin_doses.max()
                features['insulin_doses_count'] = len(insulin_doses)
        
        # Extract time window volatility
        features.update(self.calculate_glucose_volatility(df))
        
        # Extract meal-related features
        features.update(self.extract_meal_related_features(df))
        
        # Extract comment sentiment
        features.update(self.analyze_comment_sentiment(df))
        
        return features
    
    def calculate_glucose_volatility(self, df: pd.DataFrame, window_hours: int = 2) -> Dict[str, float]:
        """
        Calculate glucose volatility in time windows.
        
        Args:
            df: DataFrame containing DiaTrend data
            window_hours: Size of the time window in hours
            
        Returns:
            Dictionary of volatility features
        """
        features = {}
        
        if 'GlucoseLevel' not in df.columns or 'Time' not in df.columns:
            return features
        
        # Make sure time is sorted
        df = df.sort_values('Time')
        
        # Filter to only glucose readings
        glucose_df = df[df['GlucoseLevel'].notna()].copy()
        if len(glucose_df) < 3:
            return features
        
        # Calculate rolling standard deviation
        glucose_df = glucose_df.set_index('Time')
        window_size = f'{window_hours}H'
        rolling_std = glucose_df['GlucoseLevel'].rolling(window_size).std()
        
        features['glucose_volatility_2hr_mean'] = rolling_std.mean()
        features['glucose_volatility_2hr_max'] = rolling_std.max()
        
        # Calculate rate of change (mg/dL per minute)
        glucose_df = glucose_df.reset_index()
        glucose_df['time_diff'] = glucose_df['Time'].diff().dt.total_seconds() / 60  # in minutes
        glucose_df['glucose_diff'] = glucose_df['GlucoseLevel'].diff()
        glucose_df['rate_of_change'] = glucose_df['glucose_diff'] / glucose_df['time_diff']
        
        # Remove infinite values
        valid_roc = glucose_df['rate_of_change'].replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(valid_roc) > 0:
            features['glucose_rate_of_change_mean'] = valid_roc.mean()
            features['glucose_rate_of_change_max'] = valid_roc.max()
            features['glucose_rate_of_change_min'] = valid_roc.min()
        
        return features
    
    def extract_meal_related_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Extract pre/post-meal glucose features.
        
        Args:
            df: DataFrame containing DiaTrend data
            
        Returns:
            Dictionary of meal-related features
        """
        features = {}
        
        if 'GlucoseLevel' not in df.columns or 'Time' not in df.columns or 'Comment' not in df.columns:
            return features
        
        # Identify potential meal times from comments or insulin doses
        meal_indicators = ['meal', 'breakfast', 'lunch', 'dinner', 'eating', 'food', 'carbs', 'snack']
        
        # Find rows that might indicate meals
        meal_rows = df[df['Comment'].str.lower().str.contains('|'.join(meal_indicators), na=False)]
        
        # Also consider insulin doses as potential meal indicators
        if 'InsulinDose' in df.columns:
            insulin_rows = df[(df['InsulinDose'] > 0) & df['InsulinDose'].notna()]
            meal_rows = pd.concat([meal_rows, insulin_rows]).drop_duplicates()
        
        if len(meal_rows) == 0:
            return features
        
        # Calculate pre and post meal glucose values
        pre_meal_slopes = []
        post_meal_slopes = []
        
        for _, meal_row in meal_rows.iterrows():
            meal_time = meal_row['Time']
            
            # Get glucose values 2 hours before meal
            pre_meal = df[(df['Time'] >= meal_time - pd.Timedelta(hours=2)) & 
                          (df['Time'] < meal_time) & 
                          df['GlucoseLevel'].notna()]
            
            # Get glucose values 2 hours after meal
            post_meal = df[(df['Time'] > meal_time) & 
                           (df['Time'] <= meal_time + pd.Timedelta(hours=2)) & 
                           df['GlucoseLevel'].notna()]
            
            # Calculate pre-meal slope if enough data points
            if len(pre_meal) >= 2:
                pre_meal = pre_meal.sort_values('Time')
                # Convert to minutes since first reading
                pre_meal['minutes'] = (pre_meal['Time'] - pre_meal['Time'].iloc[0]).dt.total_seconds() / 60
                if len(pre_meal) > 1:  # Need at least 2 points for linear regression
                    slope, _ = np.polyfit(pre_meal['minutes'], pre_meal['GlucoseLevel'], 1)
                    pre_meal_slopes.append(slope)
            
            # Calculate post-meal slope if enough data points
            if len(post_meal) >= 2:
                post_meal = post_meal.sort_values('Time')
                # Convert to minutes since meal
                post_meal['minutes'] = (post_meal['Time'] - meal_time).dt.total_seconds() / 60
                if len(post_meal) > 1:  # Need at least 2 points for linear regression
                    slope, _ = np.polyfit(post_meal['minutes'], post_meal['GlucoseLevel'], 1)
                    post_meal_slopes.append(slope)
        
        # Add features if we have enough data
        if pre_meal_slopes:
            features['pre_meal_glucose_slope_mean'] = np.mean(pre_meal_slopes)
            features['pre_meal_glucose_slope_min'] = np.min(pre_meal_slopes)
            features['pre_meal_glucose_slope_max'] = np.max(pre_meal_slopes)
        
        if post_meal_slopes:
            features['post_meal_glucose_slope_mean'] = np.mean(post_meal_slopes)
            features['post_meal_glucose_slope_min'] = np.min(post_meal_slopes)
            features['post_meal_glucose_slope_max'] = np.max(post_meal_slopes)
        
        return features
    
    def analyze_comment_sentiment(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Analyze sentiment and extract tags from user comments.
        
        Args:
            df: DataFrame containing DiaTrend data
            
        Returns:
            Dictionary of comment analysis features
        """
        features = {}
        
        if 'Comment' not in df.columns:
            return features
        
        # Get all non-empty comments
        comments = df['Comment'].dropna().astype(str).tolist()
        if not comments:
            return features
        
        # Combine all comments for the day
        all_comments = ' '.join(comments)
        features['comment_count'] = len(comments)
        features['comment_word_count'] = len(all_comments.split())
        
        # Simple keyword-based sentiment analysis
        positive_keywords = ['good', 'great', 'better', 'improved', 'happy', 'stable', 'steady', 'normal']
        negative_keywords = ['bad', 'worse', 'difficult', 'problem', 'issue', 'tired', 'fatigue', 'sick', 
                            'dizzy', 'shaky', 'low', 'high', 'hypo', 'hyper', 'stress', 'pain']
        
        # Count occurrences
        positive_count = sum(1 for word in positive_keywords if word in all_comments.lower())
        negative_count = sum(1 for word in negative_keywords if word in all_comments.lower())
        
        # Calculate simple sentiment score (-1 to 1)
        total_count = positive_count + negative_count
        if total_count > 0:
            features['comment_sentiment_score'] = (positive_count - negative_count) / total_count
        else:
            features['comment_sentiment_score'] = 0
        
        # Extract common tags
        tag_keywords = {
            'exercise': ['exercise', 'workout', 'run', 'running', 'walk', 'walking', 'gym', 'training'],
            'stress': ['stress', 'stressed', 'anxiety', 'anxious', 'worried', 'tension'],
            'sleep': ['sleep', 'tired', 'fatigue', 'rest', 'insomnia', 'nap'],
            'illness': ['sick', 'ill', 'fever', 'cold', 'flu', 'infection'],
            'medication': ['medicine', 'medication', 'drug', 'pill', 'prescription']
        }
        
        for tag, keywords in tag_keywords.items():
            tag_present = any(keyword in all_comments.lower() for keyword in keywords)
            features[f'tag_{tag}'] = 1 if tag_present else 0
        
        return features


# Example usage
if __name__ == "__main__":
    import pandas as pd
    from datetime import datetime, timedelta
    
    # Create sample HRV data
    hrv_data = {
        'timestamp': [datetime(2023, 1, 1, h, 0) for h in range(24)],
        'rmssd': [50 + 10 * np.sin(h/3) + np.random.normal(0, 5) for h in range(24)]
    }
    hrv_df = pd.DataFrame(hrv_data)
    
    # Create sample activity data
    activity_data = {
        'timestamp': [datetime(2023, 1, 1, h, 0) for h in range(8, 20)],
        'steps': [np.random.randint(100, 1000) for _ in range(12)],
        'heart_rate': [80 + 20 * np.sin(h/3) + np.random.normal(0, 10) for h in range(12)],
        'intensity': [np.random.uniform(1, 10) for _ in range(12)],
        'duration': [60 for _ in range(12)]  # 60 minutes per entry
    }
    activity_df = pd.DataFrame(activity_data)
    
    # Create sample sleep data
    sleep_data = {
        'start_time': [datetime(2023, 1, 1, 22, 0)],
        'end_time': [datetime(2023, 1, 2, 6, 30)],
        'deep_sleep': [90],  # minutes
        'light_sleep': [180],
        'rem_sleep': [90],
        'awake': [30]
    }
    sleep_df = pd.DataFrame(sleep_data)
    
    # Initialize feature engineer
    feature_eng = FeatureEngineer()
    
    # Extract features
    hrv_features = feature_eng.extract_hrv_features(hrv_df)
    activity_features = feature_eng.extract_activity_features(activity_df)
    sleep_features = feature_eng.extract_sleep_features(sleep_df)
    training_load = feature_eng.calculate_training_load([activity_df])
    
    # Combine features
    combined_features = feature_eng.combine_features(
        hrv_features, activity_features, sleep_features, training_load
    )
    
    print("Combined Features:")
    for key, value in combined_features.items():
        print(f"{key}: {value}")
