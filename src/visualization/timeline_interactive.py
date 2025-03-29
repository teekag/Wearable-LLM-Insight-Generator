"""
Interactive Timeline Module for Wearable LLM Insight Generator

This module extends the core timeline visualizer with interactive capabilities
using Plotly for web-based interactive visualizations.
"""

import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Tuple
import logging
import json
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define insight type colors (matching the static visualizer)
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
    "hrv_rmssd": "#3498db", # Blue
    "resting_hr": "#e74c3c", # Red
    "sleep_hours": "#9b59b6", # Purple
    "sleep_quality": "#8e44ad", # Dark Purple
    "recovery_score": "#2ecc71", # Green
    "strain": "#e67e22",    # Orange
    "steps": "#f1c40f",     # Yellow
    "active_minutes": "#16a085" # Teal
}

class InteractiveTimeline:
    """Class to create interactive timeline visualizations of wearable data with insights."""
    
    def __init__(self, metrics: Optional[List[str]] = None):
        """
        Initialize the Interactive Timeline.
        
        Args:
            metrics: Optional list of metrics to visualize
        """
        self.metrics = metrics or ["hrv_rmssd", "sleep_hours", "recovery_score", "strain"]
        
        # Create output directory
        os.makedirs("outputs/interactive", exist_ok=True)
    
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
    
    def create_interactive_timeline(self, 
                                  df: pd.DataFrame, 
                                  insights: Optional[List[Dict[str, Any]]] = None,
                                  title: str = "Interactive Wearable Data Timeline",
                                  output_path: Optional[str] = None,
                                  height: int = 800,
                                  date_range: Optional[Tuple[datetime, datetime]] = None) -> go.Figure:
        """
        Create an interactive timeline visualization using Plotly.
        
        Args:
            df: DataFrame with time-series data
            insights: Optional list of insight dictionaries
            title: Plot title
            output_path: Optional path to save the HTML visualization
            height: Height of the figure in pixels
            date_range: Optional tuple of (start_date, end_date) to filter data
            
        Returns:
            Plotly Figure object
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
        
        # Create subplot figure
        fig = make_subplots(rows=n_metrics, cols=1, shared_xaxes=True, 
                           vertical_spacing=0.05,
                           subplot_titles=[m.replace('_', ' ').title() for m in self.available_metrics])
        
        # Plot each metric
        for i, metric in enumerate(self.available_metrics):
            # Get metric color
            color = METRIC_COLORS.get(metric, f"rgb({(i*50)%255}, {(i*100)%255}, {(i*150)%255})")
            
            # Add line trace for the metric
            fig.add_trace(
                go.Scatter(
                    x=prepared_df['date'],
                    y=prepared_df[metric],
                    mode='lines',
                    name=metric.replace('_', ' ').title(),
                    line=dict(color=color, width=2),
                    hovertemplate=f"{metric.replace('_', ' ').title()}: %{{y:.1f}}<br>Date: %{{x|%Y-%m-%d}}<extra></extra>"
                ),
                row=i+1, col=1
            )
            
            # Add insight markers if available
            if 'has_insight' in prepared_df.columns and any(prepared_df['has_insight']):
                insight_df = prepared_df[prepared_df['has_insight']]
                
                for idx, row in insight_df.iterrows():
                    # Get insight types and summaries
                    insight_types = row['insight_types']
                    insight_summaries = row['insight_summaries']
                    
                    if insight_types and insight_summaries:
                        # Create hover text with all insights for this date
                        hover_text = "<br>".join([
                            f"<b>{t.title()}:</b> {s}" 
                            for t, s in zip(insight_types, insight_summaries)
                        ])
                        
                        # Use the first insight type for the marker color
                        insight_type = insight_types[0]
                        marker_color = INSIGHT_COLORS.get(insight_type, 'gray')
                        
                        # Add marker
                        fig.add_trace(
                            go.Scatter(
                                x=[row['date']],
                                y=[row[metric]],
                                mode='markers',
                                marker=dict(
                                    color=marker_color,
                                    size=12,
                                    line=dict(color='black', width=1)
                                ),
                                name=f"{insight_type.title()} Insight",
                                text=hover_text,
                                hoverinfo='text',
                                showlegend=False
                            ),
                            row=i+1, col=1
                        )
                
                # Add vertical lines for insights
                if i == 0:  # Only add once
                    for idx, row in insight_df.iterrows():
                        insight_types = row['insight_types']
                        if insight_types:
                            # Use the first insight type for the line color
                            insight_type = insight_types[0]
                            line_color = INSIGHT_COLORS.get(insight_type, 'gray')
                            
                            # Add vertical line across all subplots
                            for j in range(n_metrics):
                                fig.add_shape(
                                    type="line",
                                    x0=row['date'],
                                    y0=0,
                                    x1=row['date'],
                                    y1=1,
                                    yref="paper",
                                    line=dict(color=line_color, width=1, dash="dash"),
                                    row=j+1, col=1
                                )
        
        # Create a custom legend for insight types
        if insights and len(insights) > 0:
            # Collect unique insight types
            all_insight_types = []
            if 'insight_types' in prepared_df.columns:
                for types in prepared_df['insight_types'].dropna():
                    if types:
                        all_insight_types.extend(types)
            
            unique_insight_types = list(set(all_insight_types))
            
            # Add a trace for each insight type (for legend only)
            for insight_type in unique_insight_types:
                color = INSIGHT_COLORS.get(insight_type, 'gray')
                
                fig.add_trace(
                    go.Scatter(
                        x=[None],
                        y=[None],
                        mode='markers',
                        marker=dict(color=color, size=10),
                        name=f"{insight_type.title()} Insight"
                    )
                )
        
        # Update layout
        fig.update_layout(
            title=title,
            height=height,
            hovermode="closest",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=60, r=30, t=80, b=60)
        )
        
        # Add range slider
        fig.update_layout(
            xaxis=dict(
                rangeslider=dict(visible=True),
                type="date"
            )
        )
        
        # Add buttons for time range selection
        fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    direction="right",
                    buttons=[
                        dict(
                            args=[{"xaxis.range": [prepared_df['date'].min(), prepared_df['date'].max()]}],
                            label="All Time",
                            method="relayout"
                        ),
                        dict(
                            args=[{"xaxis.range": [(prepared_df['date'].max() - timedelta(days=7)), prepared_df['date'].max()]}],
                            label="Last 7 Days",
                            method="relayout"
                        ),
                        dict(
                            args=[{"xaxis.range": [(prepared_df['date'].max() - timedelta(days=30)), prepared_df['date'].max()]}],
                            label="Last 30 Days",
                            method="relayout"
                        ),
                        dict(
                            args=[{"xaxis.range": [(prepared_df['date'].max() - timedelta(days=90)), prepared_df['date'].max()]}],
                            label="Last 90 Days",
                            method="relayout"
                        )
                    ],
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.1,
                    xanchor="left",
                    y=1.1,
                    yanchor="top"
                )
            ]
        )
        
        # Save if path provided
        if output_path:
            fig.write_html(output_path)
            logger.info(f"Saved interactive timeline to {output_path}")
        
        return fig
    
    def create_metric_correlation_dashboard(self, 
                                          df: pd.DataFrame,
                                          title: str = "Metric Correlation Dashboard",
                                          output_path: Optional[str] = None) -> go.Figure:
        """
        Create an interactive dashboard showing correlations between metrics.
        
        Args:
            df: DataFrame with time-series data
            title: Dashboard title
            output_path: Optional path to save the HTML visualization
            
        Returns:
            Plotly Figure object
        """
        # Prepare data
        prepared_df = self.prepare_data(df)
        
        # Get available metrics
        metrics = self.available_metrics
        
        if len(metrics) < 2:
            logger.error("Need at least 2 metrics for correlation analysis")
            return None
        
        # Create scatter matrix
        fig = px.scatter_matrix(
            prepared_df,
            dimensions=metrics,
            color_discrete_sequence=px.colors.qualitative.Plotly,
            title=title,
            labels={col: col.replace('_', ' ').title() for col in metrics},
            hover_data=['date']
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            width=900,
            title=dict(
                text=title,
                x=0.5,
                xanchor='center'
            ),
            margin=dict(l=50, r=50, t=100, b=50)
        )
        
        # Update traces
        fig.update_traces(
            diagonal_visible=False,
            showupperhalf=False
        )
        
        # Save if path provided
        if output_path:
            fig.write_html(output_path)
            logger.info(f"Saved correlation dashboard to {output_path}")
        
        return fig
    
    def create_insight_timeline(self, 
                              insights: List[Dict[str, Any]],
                              title: str = "Insight Timeline",
                              output_path: Optional[str] = None) -> go.Figure:
        """
        Create an interactive timeline focused on insights.
        
        Args:
            insights: List of insight dictionaries
            title: Timeline title
            output_path: Optional path to save the HTML visualization
            
        Returns:
            Plotly Figure object
        """
        if not insights:
            logger.error("No insights provided for timeline")
            return None
        
        # Extract data from insights
        insight_data = []
        
        for i, insight in enumerate(insights):
            # Extract date
            insight_date = None
            
            if 'date' in insight:
                insight_date = pd.to_datetime(insight['date'])
            elif 'timestamp' in insight:
                insight_date = pd.to_datetime(insight['timestamp'])
            
            if not insight_date:
                logger.warning(f"Could not determine date for insight {i}, skipping")
                continue
            
            # Extract type and summary
            insight_type = insight.get('type', 'general')
            
            if 'summary' in insight:
                insight_summary = insight['summary']
            elif 'content' in insight:
                insight_summary = insight['content']
            elif 'message' in insight:
                insight_summary = insight['message']
            else:
                insight_summary = str(insight)
            
            # Add to data
            insight_data.append({
                'date': insight_date,
                'type': insight_type,
                'summary': insight_summary,
                'id': insight.get('id', f"insight_{i}")
            })
        
        if not insight_data:
            logger.error("Could not extract valid data from insights")
            return None
        
        # Convert to DataFrame
        insight_df = pd.DataFrame(insight_data)
        
        # Sort by date
        insight_df = insight_df.sort_values('date')
        
        # Create figure
        fig = go.Figure()
        
        # Group insights by type
        for insight_type, group in insight_df.groupby('type'):
            color = INSIGHT_COLORS.get(insight_type, 'gray')
            
            fig.add_trace(
                go.Scatter(
                    x=group['date'],
                    y=[insight_type] * len(group),
                    mode='markers',
                    name=insight_type.title(),
                    marker=dict(
                        color=color,
                        size=15,
                        line=dict(color='black', width=1)
                    ),
                    text=group['summary'],
                    hovertemplate="<b>%{y}</b><br>%{text}<br>Date: %{x|%Y-%m-%d}<extra></extra>"
                )
            )
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Insight Type",
            height=500,
            hovermode="closest",
            yaxis=dict(
                categoryorder='category ascending'
            ),
            margin=dict(l=100, r=30, t=80, b=60)
        )
        
        # Add range slider
        fig.update_layout(
            xaxis=dict(
                rangeslider=dict(visible=True),
                type="date"
            )
        )
        
        # Save if path provided
        if output_path:
            fig.write_html(output_path)
            logger.info(f"Saved insight timeline to {output_path}")
        
        return fig
    
    def filter_by_metric(self, 
                       fig: go.Figure, 
                       metrics_to_show: List[str]) -> go.Figure:
        """
        Filter an existing figure to show only specific metrics.
        
        Args:
            fig: Plotly Figure object
            metrics_to_show: List of metric names to show
            
        Returns:
            Updated Plotly Figure object
        """
        # Create a new figure with the same layout
        new_fig = go.Figure(layout=fig.layout)
        
        # Copy only the traces for the specified metrics
        for trace in fig.data:
            if trace.name in [m.replace('_', ' ').title() for m in metrics_to_show]:
                new_fig.add_trace(trace)
        
        return new_fig
    
    def filter_by_date(self, 
                     fig: go.Figure, 
                     start_date: datetime, 
                     end_date: datetime) -> go.Figure:
        """
        Filter an existing figure to show only a specific date range.
        
        Args:
            fig: Plotly Figure object
            start_date: Start date
            end_date: End date
            
        Returns:
            Updated Plotly Figure object
        """
        # Update the x-axis range
        fig.update_layout(
            xaxis_range=[start_date, end_date]
        )
        
        return fig
    
    def enable_zoom(self, fig: go.Figure) -> go.Figure:
        """
        Enable zoom functionality on a figure.
        
        Args:
            fig: Plotly Figure object
            
        Returns:
            Updated Plotly Figure object
        """
        fig.update_layout(
            dragmode='zoom',
            xaxis=dict(
                rangeslider=dict(visible=True),
                type="date"
            )
        )
        
        return fig
    
    def enable_tooltips(self, fig: go.Figure) -> go.Figure:
        """
        Enable enhanced tooltips on a figure.
        
        Args:
            fig: Plotly Figure object
            
        Returns:
            Updated Plotly Figure object
        """
        fig.update_layout(
            hovermode="closest"
        )
        
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
    
    # Create interactive timeline
    timeline = InteractiveTimeline()
    
    # Create and save visualizations
    timeline.create_interactive_timeline(df, insights, output_path="outputs/interactive/sample_interactive_timeline.html")
    timeline.create_metric_correlation_dashboard(df, output_path="outputs/interactive/sample_correlation_dashboard.html")
    timeline.create_insight_timeline(insights, output_path="outputs/interactive/sample_insight_timeline.html")
    
    print("Sample interactive visualizations created in outputs/interactive/ directory")
