"""
Timeline Visualizer Module for Wearable LLM Insight Generator

This module provides core visualization capabilities for displaying time-series wearable data
along with insights annotations, creating a comprehensive timeline view.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Tuple
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define insight type colors
INSIGHT_COLORS = {
    "recovery": "#3498db",  # Blue
    "sleep": "#9b59b6",     # Purple
    "activity": "#2ecc71",  # Green
    "strain": "#e74c3c",    # Red
    "nutrition": "#f39c12", # Orange
    "stress": "#e67e22",    # Dark Orange
    "general": "#7f8c8d",   # Gray
    "warning": "#c0392b",   # Dark Red
    "recommendation": "#27ae60"  # Dark Green
}

# Define metric colors for consistency
METRIC_COLORS = {
    "hrv": "#3498db",       # Blue
    "resting_hr": "#e74c3c", # Red
    "sleep_hours": "#9b59b6", # Purple
    "sleep_quality": "#8e44ad", # Dark Purple
    "recovery_score": "#2ecc71", # Green
    "strain": "#e67e22",    # Orange
    "steps": "#f1c40f",     # Yellow
    "active_minutes": "#16a085" # Teal
}

class TimelineVisualizer:
    """Class to visualize time-series wearable data with insight annotations."""
    
    def __init__(self, metrics: Optional[List[str]] = None):
        """
        Initialize the Timeline Visualizer.
        
        Args:
            metrics: Optional list of metrics to visualize
        """
        self.metrics = metrics or ["hrv_rmssd", "sleep_hours", "recovery_score", "strain"]
        
        # Create output directory
        os.makedirs("outputs/visualizations", exist_ok=True)
    
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare and normalize data for visualization.
        
        Args:
            df: DataFrame with time-series data
            
        Returns:
            Prepared DataFrame
        """
        # Make a copy to avoid modifying the original
        prepared_df = df.copy()
        
        # Ensure date column is datetime
        if 'date' in prepared_df.columns and not pd.api.types.is_datetime64_any_dtype(prepared_df['date']):
            prepared_df['date'] = pd.to_datetime(prepared_df['date'])
        
        # Sort by date
        if 'date' in prepared_df.columns:
            prepared_df = prepared_df.sort_values('date')
        
        # Fill missing values with forward fill then backward fill
        prepared_df = prepared_df.ffill().bfill()
        
        # Ensure all requested metrics are present
        for metric in self.metrics:
            if metric not in prepared_df.columns:
                logger.warning(f"Metric '{metric}' not found in data, will be excluded from visualization")
        
        # Filter to only include available metrics
        self.available_metrics = [m for m in self.metrics if m in prepared_df.columns]
        
        return prepared_df
    
    def add_insights(self, df: pd.DataFrame, insights: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Add insight annotations to the data.
        
        Args:
            df: DataFrame with time-series data
            insights: List of insight dictionaries
            
        Returns:
            DataFrame with insight annotations
        """
        # Make a copy to avoid modifying the original
        df_with_insights = df.copy()
        
        # Create columns for insight annotations if they don't exist
        if 'has_insight' not in df_with_insights.columns:
            df_with_insights['has_insight'] = False
            df_with_insights['insight_types'] = None
            df_with_insights['insight_summaries'] = None
            df_with_insights['insight_ids'] = None
        
        # Process each insight
        for insight in insights:
            # Extract date from insight
            insight_date = None
            
            # Handle different date formats in insights
            if 'date' in insight:
                insight_date = pd.to_datetime(insight['date'])
            elif 'timestamp' in insight:
                insight_date = pd.to_datetime(insight['timestamp'])
            elif 'day' in insight and isinstance(insight['day'], int) and 0 <= insight['day'] < len(df):
                # If insight has a day index, use the date from that index
                insight_date = df.iloc[insight['day']]['date']
            
            if insight_date is None:
                logger.warning(f"Could not determine date for insight: {insight}")
                continue
            
            # Find the closest date in the DataFrame
            closest_idx = (df_with_insights['date'] - insight_date).abs().idxmin()
            
            # Mark this date as having an insight
            df_with_insights.at[closest_idx, 'has_insight'] = True
            
            # Extract insight type
            insight_type = insight.get('type', 'general')
            
            # Extract insight summary
            if 'summary' in insight:
                insight_summary = insight['summary']
            elif 'content' in insight:
                insight_summary = insight['content']
            elif 'message' in insight:
                insight_summary = insight['message']
            else:
                insight_summary = str(insight)
            
            # Extract insight ID
            insight_id = insight.get('id', f"insight_{closest_idx}")
            
            # Update insight information
            if df_with_insights.at[closest_idx, 'insight_types'] is None:
                df_with_insights.at[closest_idx, 'insight_types'] = [insight_type]
                df_with_insights.at[closest_idx, 'insight_summaries'] = [insight_summary]
                df_with_insights.at[closest_idx, 'insight_ids'] = [insight_id]
            else:
                df_with_insights.at[closest_idx, 'insight_types'].append(insight_type)
                df_with_insights.at[closest_idx, 'insight_summaries'].append(insight_summary)
                df_with_insights.at[closest_idx, 'insight_ids'].append(insight_id)
        
        return df_with_insights
    
    def plot_timeline(self, 
                     df: pd.DataFrame, 
                     insights: Optional[List[Dict[str, Any]]] = None,
                     title: str = "Wearable Data Timeline",
                     output_path: Optional[str] = None,
                     figsize: Tuple[int, int] = (12, 10),
                     date_range: Optional[Tuple[datetime, datetime]] = None) -> plt.Figure:
        """
        Plot a timeline visualization of wearable data with optional insight annotations.
        
        Args:
            df: DataFrame with time-series data
            insights: Optional list of insight dictionaries
            title: Plot title
            output_path: Optional path to save the visualization
            figsize: Figure size as (width, height)
            date_range: Optional tuple of (start_date, end_date) to filter data
            
        Returns:
            Matplotlib Figure object
        """
        # Prepare data
        prepared_df = self.prepare_data(df)
        
        # Add insights if provided
        if insights:
            prepared_df = self.add_insights(prepared_df, insights)
        
        # Filter by date range if provided
        if date_range:
            start_date, end_date = date_range
            prepared_df = prepared_df[(prepared_df['date'] >= start_date) & (prepared_df['date'] <= end_date)]
        
        # Determine number of metrics to plot
        n_metrics = len(self.available_metrics)
        
        if n_metrics == 0:
            logger.error("No valid metrics available to plot")
            return None
        
        # Create figure and axes
        fig, axs = plt.subplots(n_metrics, 1, figsize=figsize, sharex=True)
        
        # Handle case where there's only one metric
        if n_metrics == 1:
            axs = [axs]
        
        # Plot each metric
        for i, metric in enumerate(self.available_metrics):
            ax = axs[i]
            
            # Get metric color
            color = METRIC_COLORS.get(metric, f"C{i}")
            
            # Plot the metric
            ax.plot(prepared_df['date'], prepared_df[metric], color=color, linewidth=2)
            
            # Set y-label
            metric_label = metric.replace('_', ' ').title()
            ax.set_ylabel(metric_label, fontsize=10)
            
            # Set grid
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Add insight markers if available
            if 'has_insight' in prepared_df.columns:
                insight_dates = prepared_df[prepared_df['has_insight']]['date']
                
                for idx, insight_date in zip(prepared_df[prepared_df['has_insight']].index, insight_dates):
                    # Get y-value at this date
                    y_value = prepared_df.loc[idx, metric]
                    
                    # Get insight types for this date
                    insight_types = prepared_df.loc[idx, 'insight_types']
                    
                    if insight_types:
                        # Use the first insight type for the marker color
                        insight_type = insight_types[0]
                        marker_color = INSIGHT_COLORS.get(insight_type, 'gray')
                        
                        # Plot marker
                        ax.scatter(insight_date, y_value, color=marker_color, s=100, zorder=5, 
                                  marker='o', edgecolor='black')
                        
                        # Add vertical line across all subplots
                        if i == 0:  # Only add line on first subplot
                            for j in range(n_metrics):
                                axs[j].axvline(x=insight_date, color=marker_color, linestyle='--', 
                                              alpha=0.5, zorder=1)
        
        # Format x-axis
        plt.xlabel('Date', fontsize=12)
        
        # Format dates on x-axis
        date_format = mdates.DateFormatter('%Y-%m-%d')
        plt.gca().xaxis.set_major_formatter(date_format)
        plt.xticks(rotation=45)
        
        # Add title
        plt.suptitle(title, fontsize=16)
        
        # Add legend for insight types
        if insights and len(insights) > 0:
            # Collect unique insight types
            all_insight_types = []
            if 'insight_types' in prepared_df.columns:
                for types in prepared_df['insight_types'].dropna():
                    if types:
                        all_insight_types.extend(types)
            
            unique_insight_types = list(set(all_insight_types))
            
            # Create legend elements
            legend_elements = []
            for insight_type in unique_insight_types:
                color = INSIGHT_COLORS.get(insight_type, 'gray')
                legend_elements.append(
                    Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                          markersize=10, label=insight_type.title())
                )
            
            # Add legend to the figure
            if legend_elements:
                fig.legend(handles=legend_elements, loc='upper right', 
                          bbox_to_anchor=(0.99, 0.99), fontsize=10)
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        # Save if path provided
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved timeline visualization to {output_path}")
        
        return fig
    
    def plot_metric_comparison(self, 
                              df: pd.DataFrame,
                              metric_pairs: List[Tuple[str, str]],
                              title: str = "Metric Correlation Analysis",
                              output_path: Optional[str] = None) -> plt.Figure:
        """
        Plot correlation scatter plots between pairs of metrics.
        
        Args:
            df: DataFrame with time-series data
            metric_pairs: List of metric name pairs to compare
            title: Plot title
            output_path: Optional path to save the visualization
            
        Returns:
            Matplotlib Figure object
        """
        # Prepare data
        prepared_df = self.prepare_data(df)
        
        # Determine number of pairs to plot
        n_pairs = len(metric_pairs)
        
        if n_pairs == 0:
            logger.error("No metric pairs provided for comparison")
            return None
        
        # Calculate grid dimensions
        n_cols = min(3, n_pairs)
        n_rows = (n_pairs + n_cols - 1) // n_cols
        
        # Create figure and axes
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        
        # Handle case where there's only one pair
        if n_pairs == 1:
            axs = np.array([axs])
        
        # Flatten axes array for easier indexing
        axs = axs.flatten()
        
        # Plot each metric pair
        for i, (metric1, metric2) in enumerate(metric_pairs):
            if i < len(axs):
                ax = axs[i]
                
                # Check if both metrics are available
                if metric1 in prepared_df.columns and metric2 in prepared_df.columns:
                    # Get colors
                    color1 = METRIC_COLORS.get(metric1, f"C{i}")
                    
                    # Create scatter plot
                    ax.scatter(prepared_df[metric1], prepared_df[metric2], 
                              alpha=0.7, color=color1, edgecolor='black')
                    
                    # Add trend line
                    z = np.polyfit(prepared_df[metric1], prepared_df[metric2], 1)
                    p = np.poly1d(z)
                    ax.plot(prepared_df[metric1], p(prepared_df[metric1]), 
                           linestyle='--', color='red')
                    
                    # Calculate correlation
                    corr = prepared_df[metric1].corr(prepared_df[metric2])
                    
                    # Add correlation text
                    ax.text(0.05, 0.95, f"r = {corr:.2f}", transform=ax.transAxes, 
                           fontsize=10, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    
                    # Set labels
                    metric1_label = metric1.replace('_', ' ').title()
                    metric2_label = metric2.replace('_', ' ').title()
                    ax.set_xlabel(metric1_label)
                    ax.set_ylabel(metric2_label)
                    
                    # Set grid
                    ax.grid(True, linestyle='--', alpha=0.3)
                else:
                    # Handle missing metrics
                    missing = []
                    if metric1 not in prepared_df.columns:
                        missing.append(metric1)
                    if metric2 not in prepared_df.columns:
                        missing.append(metric2)
                    
                    ax.text(0.5, 0.5, f"Missing metrics: {', '.join(missing)}", 
                           ha='center', va='center', transform=ax.transAxes)
                    ax.set_xticks([])
                    ax.set_yticks([])
        
        # Hide any unused subplots
        for i in range(n_pairs, len(axs)):
            axs[i].set_visible(False)
        
        # Add title
        plt.suptitle(title, fontsize=16)
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        # Save if path provided
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved metric comparison visualization to {output_path}")
        
        return fig
    
    def plot_insight_distribution(self, 
                                 insights: List[Dict[str, Any]],
                                 title: str = "Insight Distribution Analysis",
                                 output_path: Optional[str] = None) -> plt.Figure:
        """
        Plot distribution of insights by type and time.
        
        Args:
            insights: List of insight dictionaries
            title: Plot title
            output_path: Optional path to save the visualization
            
        Returns:
            Matplotlib Figure object
        """
        if not insights:
            logger.error("No insights provided for distribution analysis")
            return None
        
        # Extract dates and types from insights
        dates = []
        types = []
        
        for insight in insights:
            # Extract date
            insight_date = None
            
            if 'date' in insight:
                insight_date = pd.to_datetime(insight['date'])
            elif 'timestamp' in insight:
                insight_date = pd.to_datetime(insight['timestamp'])
            
            if insight_date:
                dates.append(insight_date)
                
                # Extract type
                insight_type = insight.get('type', 'general')
                types.append(insight_type)
        
        if not dates:
            logger.error("Could not extract dates from insights")
            return None
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Plot 1: Insights by type (pie chart)
        type_counts = pd.Series(types).value_counts()
        
        # Get colors for each type
        colors = [INSIGHT_COLORS.get(t, 'gray') for t in type_counts.index]
        
        ax1.pie(type_counts, labels=type_counts.index, autopct='%1.1f%%', 
               startangle=90, colors=colors)
        ax1.set_title('Insights by Type')
        
        # Plot 2: Insights over time (histogram)
        ax2.hist(dates, bins=min(20, len(dates)), color='skyblue', edgecolor='black')
        ax2.set_title('Insights Over Time')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Number of Insights')
        
        # Format dates on x-axis
        date_format = mdates.DateFormatter('%Y-%m-%d')
        ax2.xaxis.set_major_formatter(date_format)
        plt.xticks(rotation=45)
        
        # Add overall title
        plt.suptitle(title, fontsize=16)
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        
        # Save if path provided
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved insight distribution visualization to {output_path}")
        
        return fig


# Example usage
if __name__ == "__main__":
    # Create sample data
    dates = pd.date_range(start='2025-01-01', periods=30)
    data = {
        'date': dates,
        'hrv_rmssd': np.random.normal(65, 10, 30),
        'sleep_hours': np.random.normal(7.5, 1, 30),
        'recovery_score': np.random.normal(75, 15, 30),
        'strain': np.random.normal(12, 4, 30)
    }
    df = pd.DataFrame(data)
    
    # Create sample insights
    insights = [
        {
            'date': '2025-01-05',
            'type': 'recovery',
            'summary': 'Your recovery is trending upward, indicating good adaptation to recent training.'
        },
        {
            'date': '2025-01-10',
            'type': 'sleep',
            'summary': 'Your sleep duration has decreased over the past 3 days, which may impact recovery.'
        },
        {
            'date': '2025-01-15',
            'type': 'strain',
            'summary': 'High strain detected. Consider a recovery day tomorrow.'
        },
        {
            'date': '2025-01-20',
            'type': 'warning',
            'summary': 'HRV trending downward for 5 consecutive days, indicating possible overtraining.'
        },
        {
            'date': '2025-01-25',
            'type': 'recommendation',
            'summary': 'Based on your recovery trends, consider increasing training intensity next week.'
        }
    ]
    
    # Create visualizer
    visualizer = TimelineVisualizer()
    
    # Plot timeline
    visualizer.plot_timeline(df, insights, output_path="outputs/visualizations/sample_timeline.png")
    
    # Plot metric comparisons
    metric_pairs = [('hrv_rmssd', 'recovery_score'), ('sleep_hours', 'recovery_score'), ('strain', 'hrv_rmssd')]
    visualizer.plot_metric_comparison(df, metric_pairs, output_path="outputs/visualizations/sample_correlations.png")
    
    # Plot insight distribution
    visualizer.plot_insight_distribution(insights, output_path="outputs/visualizations/sample_insight_distribution.png")
    
    print("Sample visualizations created in outputs/visualizations/ directory")
