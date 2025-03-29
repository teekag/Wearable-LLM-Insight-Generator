"""
Timeline Integration Module for Wearable LLM Insight Generator

This module integrates the timeline visualization components with the insight engine,
user profiles, and simulator to create comprehensive visualizations of wearable data and insights.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Tuple
import logging
import json
from pathlib import Path

# Import visualization components
from src.visualization.timeline_visualizer import TimelineVisualizer
from src.visualization.timeline_interactive import InteractiveTimeline

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TimelineIntegration:
    """Class to integrate timeline visualization with other system components."""
    
    def __init__(self, insight_engine=None, user_profile_manager=None):
        """
        Initialize the Timeline Integration.
        
        Args:
            insight_engine: Optional insight engine instance
            user_profile_manager: Optional user profile manager instance
        """
        self.insight_engine = insight_engine
        self.user_profile_manager = user_profile_manager
        self.static_visualizer = TimelineVisualizer()
        self.interactive_visualizer = InteractiveTimeline()
        
        # Create output directories
        os.makedirs("outputs/visualizations", exist_ok=True)
        os.makedirs("outputs/interactive", exist_ok=True)
    
    def visualize_user_data(self, 
                           user_id: str, 
                           data: pd.DataFrame,
                           date_range: Optional[Tuple[datetime, datetime]] = None,
                           metrics: Optional[List[str]] = None,
                           include_insights: bool = True,
                           interactive: bool = True,
                           output_dir: Optional[str] = None) -> Dict[str, str]:
        """
        Create visualizations for a user's wearable data.
        
        Args:
            user_id: User ID
            data: DataFrame with wearable data
            date_range: Optional tuple of (start_date, end_date) to filter data
            metrics: Optional list of metrics to include
            include_insights: Whether to include insights in the visualization
            interactive: Whether to create interactive visualizations
            output_dir: Optional output directory
            
        Returns:
            Dictionary of output file paths
        """
        # Set output directory
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        else:
            output_dir = f"outputs/visualizations/{user_id}"
            os.makedirs(output_dir, exist_ok=True)
        
        # Set metrics if provided
        if metrics:
            self.static_visualizer.metrics = metrics
            self.interactive_visualizer.metrics = metrics
        
        # Get insights if requested and insight engine is available
        insights = []
        if include_insights and self.insight_engine:
            try:
                # Filter data by date range if provided
                if date_range:
                    start_date, end_date = date_range
                    filtered_data = data[(data['date'] >= start_date) & (data['date'] <= end_date)]
                else:
                    filtered_data = data
                
                # Generate insights
                insights = self.insight_engine.generate_insight(
                    data=filtered_data,
                    user_id=user_id
                )
                
                logger.info(f"Generated {len(insights)} insights for user {user_id}")
            except Exception as e:
                logger.error(f"Error generating insights: {str(e)}")
        
        # Create output paths
        output_paths = {}
        
        # Create static visualization
        static_path = os.path.join(output_dir, "timeline_static.png")
        try:
            self.static_visualizer.plot_timeline(
                data, 
                insights=insights,
                title=f"Wearable Data Timeline for User {user_id}",
                output_path=static_path,
                date_range=date_range
            )
            output_paths["static_timeline"] = static_path
            logger.info(f"Created static timeline visualization at {static_path}")
        except Exception as e:
            logger.error(f"Error creating static visualization: {str(e)}")
        
        # Create metric correlation visualization
        if len(self.static_visualizer.available_metrics) >= 2:
            correlation_path = os.path.join(output_dir, "metric_correlation.png")
            try:
                # Create pairs of metrics
                metric_pairs = []
                metrics = self.static_visualizer.available_metrics
                for i in range(len(metrics)):
                    for j in range(i+1, len(metrics)):
                        metric_pairs.append((metrics[i], metrics[j]))
                
                self.static_visualizer.plot_metric_comparison(
                    data,
                    metric_pairs=metric_pairs,
                    title=f"Metric Correlations for User {user_id}",
                    output_path=correlation_path
                )
                output_paths["metric_correlation"] = correlation_path
                logger.info(f"Created metric correlation visualization at {correlation_path}")
            except Exception as e:
                logger.error(f"Error creating correlation visualization: {str(e)}")
        
        # Create insight distribution visualization if insights are available
        if insights:
            insight_dist_path = os.path.join(output_dir, "insight_distribution.png")
            try:
                self.static_visualizer.plot_insight_distribution(
                    insights,
                    title=f"Insight Distribution for User {user_id}",
                    output_path=insight_dist_path
                )
                output_paths["insight_distribution"] = insight_dist_path
                logger.info(f"Created insight distribution visualization at {insight_dist_path}")
            except Exception as e:
                logger.error(f"Error creating insight distribution visualization: {str(e)}")
        
        # Create interactive visualizations if requested
        if interactive:
            # Create interactive timeline
            interactive_path = os.path.join(output_dir, "timeline_interactive.html")
            try:
                self.interactive_visualizer.create_interactive_timeline(
                    data,
                    insights=insights,
                    title=f"Interactive Wearable Data Timeline for User {user_id}",
                    output_path=interactive_path,
                    date_range=date_range
                )
                output_paths["interactive_timeline"] = interactive_path
                logger.info(f"Created interactive timeline visualization at {interactive_path}")
            except Exception as e:
                logger.error(f"Error creating interactive timeline: {str(e)}")
            
            # Create interactive correlation dashboard
            if len(self.interactive_visualizer.available_metrics) >= 2:
                interactive_corr_path = os.path.join(output_dir, "correlation_dashboard.html")
                try:
                    self.interactive_visualizer.create_metric_correlation_dashboard(
                        data,
                        title=f"Interactive Metric Correlation Dashboard for User {user_id}",
                        output_path=interactive_corr_path
                    )
                    output_paths["correlation_dashboard"] = interactive_corr_path
                    logger.info(f"Created interactive correlation dashboard at {interactive_corr_path}")
                except Exception as e:
                    logger.error(f"Error creating interactive correlation dashboard: {str(e)}")
            
            # Create interactive insight timeline if insights are available
            if insights:
                insight_timeline_path = os.path.join(output_dir, "insight_timeline.html")
                try:
                    self.interactive_visualizer.create_insight_timeline(
                        insights,
                        title=f"Interactive Insight Timeline for User {user_id}",
                        output_path=insight_timeline_path
                    )
                    output_paths["insight_timeline"] = insight_timeline_path
                    logger.info(f"Created interactive insight timeline at {insight_timeline_path}")
                except Exception as e:
                    logger.error(f"Error creating interactive insight timeline: {str(e)}")
        
        return output_paths
    
    def visualize_simulation(self, 
                           simulation_data: pd.DataFrame,
                           insights: List[Dict[str, Any]],
                           scenario_type: str,
                           user_profile: Optional[Dict[str, Any]] = None,
                           output_dir: Optional[str] = None) -> Dict[str, str]:
        """
        Create visualizations for simulation results.
        
        Args:
            simulation_data: DataFrame with simulation data
            insights: List of insights from the simulation
            scenario_type: Type of scenario that was simulated
            user_profile: Optional user profile used in the simulation
            output_dir: Optional output directory
            
        Returns:
            Dictionary of output file paths
        """
        # Set output directory
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        else:
            output_dir = f"outputs/simulations/{scenario_type}"
            os.makedirs(output_dir, exist_ok=True)
        
        # Create title with scenario information
        title_prefix = f"Simulation: {scenario_type.replace('_', ' ').title()}"
        if user_profile and 'name' in user_profile:
            title_prefix += f" for {user_profile['name']}"
        
        # Create output paths
        output_paths = {}
        
        # Create static visualization
        static_path = os.path.join(output_dir, f"{scenario_type}_timeline.png")
        try:
            self.static_visualizer.plot_timeline(
                simulation_data, 
                insights=insights,
                title=f"{title_prefix} - Timeline",
                output_path=static_path
            )
            output_paths["static_timeline"] = static_path
            logger.info(f"Created static simulation timeline at {static_path}")
        except Exception as e:
            logger.error(f"Error creating static simulation visualization: {str(e)}")
        
        # Create interactive visualization
        interactive_path = os.path.join(output_dir, f"{scenario_type}_interactive.html")
        try:
            self.interactive_visualizer.create_interactive_timeline(
                simulation_data,
                insights=insights,
                title=f"{title_prefix} - Interactive Timeline",
                output_path=interactive_path
            )
            output_paths["interactive_timeline"] = interactive_path
            logger.info(f"Created interactive simulation timeline at {interactive_path}")
        except Exception as e:
            logger.error(f"Error creating interactive simulation visualization: {str(e)}")
        
        # Create insight timeline if insights are available
        if insights:
            insight_timeline_path = os.path.join(output_dir, f"{scenario_type}_insights.html")
            try:
                self.interactive_visualizer.create_insight_timeline(
                    insights,
                    title=f"{title_prefix} - Insights",
                    output_path=insight_timeline_path
                )
                output_paths["insight_timeline"] = insight_timeline_path
                logger.info(f"Created interactive insight timeline at {insight_timeline_path}")
            except Exception as e:
                logger.error(f"Error creating interactive insight timeline: {str(e)}")
        
        return output_paths
    
    def compare_scenarios(self, 
                        scenario_results: Dict[str, Dict[str, Any]],
                        output_dir: Optional[str] = None) -> Dict[str, str]:
        """
        Create visualizations comparing different simulation scenarios.
        
        Args:
            scenario_results: Dictionary of scenario results, where keys are scenario names
                             and values are dictionaries with 'simulation_data' and 'insights'
            output_dir: Optional output directory
            
        Returns:
            Dictionary of output file paths
        """
        # Set output directory
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        else:
            output_dir = "outputs/simulations/comparison"
            os.makedirs(output_dir, exist_ok=True)
        
        # Check if there are scenarios to compare
        if not scenario_results or len(scenario_results) < 2:
            logger.error("Need at least 2 scenarios to compare")
            return {}
        
        # Create output paths
        output_paths = {}
        
        # Extract key metrics for comparison
        comparison_data = {}
        
        for scenario, data in scenario_results.items():
            if 'simulation_data' in data and isinstance(data['simulation_data'], pd.DataFrame):
                df = data['simulation_data']
                
                # Calculate average metrics
                metrics = {}
                for metric in self.static_visualizer.metrics:
                    if metric in df.columns:
                        metrics[f"avg_{metric}"] = df[metric].mean()
                        metrics[f"min_{metric}"] = df[metric].min()
                        metrics[f"max_{metric}"] = df[metric].max()
                
                # Add insight count
                if 'insights' in data and isinstance(data['insights'], list):
                    metrics["insight_count"] = len(data['insights'])
                
                comparison_data[scenario] = metrics
        
        # Create comparison visualization
        if comparison_data:
            # Convert to DataFrame
            comparison_df = pd.DataFrame.from_dict(comparison_data, orient='index')
            
            # Create bar charts for each metric
            for metric in self.static_visualizer.metrics:
                if f"avg_{metric}" in comparison_df.columns:
                    metric_path = os.path.join(output_dir, f"comparison_{metric}.png")
                    
                    try:
                        # Create figure
                        import matplotlib.pyplot as plt
                        
                        plt.figure(figsize=(10, 6))
                        
                        # Plot average value
                        comparison_df[f"avg_{metric}"].plot(kind='bar', color='skyblue')
                        
                        # Add min/max as error bars if available
                        if f"min_{metric}" in comparison_df.columns and f"max_{metric}" in comparison_df.columns:
                            plt.errorbar(
                                x=range(len(comparison_df)),
                                y=comparison_df[f"avg_{metric}"],
                                yerr=[
                                    comparison_df[f"avg_{metric}"] - comparison_df[f"min_{metric}"],
                                    comparison_df[f"max_{metric}"] - comparison_df[f"avg_{metric}"]
                                ],
                                fmt='none',
                                color='black',
                                capsize=5
                            )
                        
                        # Set labels and title
                        plt.xlabel('Scenario')
                        plt.ylabel(metric.replace('_', ' ').title())
                        plt.title(f'Comparison of {metric.replace("_", " ").title()} Across Scenarios')
                        plt.xticks(rotation=45, ha='right')
                        plt.tight_layout()
                        
                        # Save
                        plt.savefig(metric_path)
                        plt.close()
                        
                        output_paths[f"comparison_{metric}"] = metric_path
                        logger.info(f"Created scenario comparison for {metric} at {metric_path}")
                    except Exception as e:
                        logger.error(f"Error creating comparison for {metric}: {str(e)}")
            
            # Create insight count comparison if available
            if "insight_count" in comparison_df.columns:
                insight_path = os.path.join(output_dir, "comparison_insights.png")
                
                try:
                    # Create figure
                    import matplotlib.pyplot as plt
                    
                    plt.figure(figsize=(10, 6))
                    comparison_df["insight_count"].plot(kind='bar', color='purple')
                    
                    # Set labels and title
                    plt.xlabel('Scenario')
                    plt.ylabel('Number of Insights')
                    plt.title('Comparison of Insight Count Across Scenarios')
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    
                    # Save
                    plt.savefig(insight_path)
                    plt.close()
                    
                    output_paths["comparison_insights"] = insight_path
                    logger.info(f"Created insight count comparison at {insight_path}")
                except Exception as e:
                    logger.error(f"Error creating insight count comparison: {str(e)}")
            
            # Save comparison data as JSON
            json_path = os.path.join(output_dir, "scenario_comparison.json")
            try:
                with open(json_path, 'w') as f:
                    # Convert DataFrame to dict for JSON serialization
                    json.dump(comparison_df.to_dict(), f, indent=2)
                
                output_paths["comparison_json"] = json_path
                logger.info(f"Saved scenario comparison data to {json_path}")
            except Exception as e:
                logger.error(f"Error saving comparison data: {str(e)}")
        
        return output_paths
    
    def get_insight_annotations(self, 
                              user_id: str, 
                              date_range: Optional[Tuple[datetime, datetime]] = None) -> List[Dict[str, Any]]:
        """
        Get insight annotations for a specific user and date range.
        
        Args:
            user_id: User ID
            date_range: Optional tuple of (start_date, end_date) to filter insights
            
        Returns:
            List of insight dictionaries
        """
        insights = []
        
        # Check if insight engine is available
        if not self.insight_engine:
            logger.warning("Insight engine not available")
            return insights
        
        try:
            # Get insights from the insight engine
            # This assumes the insight engine has a method to retrieve existing insights
            if hasattr(self.insight_engine, 'get_user_insights'):
                insights = self.insight_engine.get_user_insights(user_id)
            
            # Filter by date range if provided
            if date_range and insights:
                start_date, end_date = date_range
                
                filtered_insights = []
                for insight in insights:
                    insight_date = None
                    
                    if 'date' in insight:
                        insight_date = pd.to_datetime(insight['date'])
                    elif 'timestamp' in insight:
                        insight_date = pd.to_datetime(insight['timestamp'])
                    
                    if insight_date and start_date <= insight_date <= end_date:
                        filtered_insights.append(insight)
                
                insights = filtered_insights
            
            logger.info(f"Retrieved {len(insights)} insights for user {user_id}")
        except Exception as e:
            logger.error(f"Error retrieving insights: {str(e)}")
        
        return insights


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
    
    # Create timeline integration
    integration = TimelineIntegration()
    
    # Visualize user data
    output_paths = integration.visualize_user_data(
        user_id="test_user_001",
        data=df,
        include_insights=True,
        interactive=True
    )
    
    print("Created visualizations:")
    for key, path in output_paths.items():
        print(f"- {key}: {path}")
    
    # Simulate scenario comparison
    scenario_results = {
        "overtraining": {
            "simulation_data": df.copy(),
            "insights": insights
        },
        "recovery": {
            "simulation_data": df.copy(),
            "insights": insights[:2]
        }
    }
    
    comparison_paths = integration.compare_scenarios(scenario_results)
    
    print("\nCreated scenario comparisons:")
    for key, path in comparison_paths.items():
        print(f"- {key}: {path}")
