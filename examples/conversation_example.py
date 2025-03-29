#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Conversation Example for Wearable Data Insight Generator

This script demonstrates the conversational interface for the Wearable Data Insight Generator,
showing how users can interact with the system through natural language to get personalized
insights and recommendations based on their wearable data.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Any, Tuple, Optional

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.demo_data_generator import DemoDataGenerator
from src.simulator_engine import SimulatorEngine
from src.visualization.timeline_interactive import InteractiveTimeline
from src.visualization.timeline_integration import TimelineIntegration

# Create output directories if they don't exist
os.makedirs('outputs/examples/conversations', exist_ok=True)

class ConversationalAgent:
    """
    A simple conversational agent that simulates interaction with the Wearable Data Insight Generator.
    """
    
    def __init__(self):
        """Initialize the conversational agent with necessary components."""
        self.data_generator = DemoDataGenerator()
        self.simulator = SimulatorEngine()
        self.user_profile = None
        self.user_data = None
        self.insights = []
        self.conversation_history = []
        
    def set_user_data(self, profile_type: str = "athlete", days: int = 30):
        """Generate and set user data for the conversation."""
        self.user_profile, self.user_data, _ = self.data_generator.generate_demo_dataset(
            profile_type=profile_type,
            days=days
        )
        self.simulator.set_user_profile(self.user_profile)
        
        # The SimulatorEngine doesn't have a set_user_data method
        # Instead, we'll use the simulation parameters and run a baseline simulation
        self.simulator.set_simulation_parameters("baseline", days=days)
        # The simulation data will be generated when we run the simulation
        
    def process_user_query(self, query: str) -> Dict[str, Any]:
        """
        Process a user query and generate a response.
        
        Args:
            query: The user's natural language query
            
        Returns:
            A dictionary containing the response and any relevant visualizations
        """
        # Add query to conversation history
        self.conversation_history.append({"role": "user", "content": query})
        
        # Simple keyword-based response generation
        # In a real system, this would use an LLM or more sophisticated NLP
        response = {}
        
        # Check for different query types
        if any(keyword in query.lower() for keyword in ["overview", "summary", "status"]):
            response = self._generate_overview_response()
        elif any(keyword in query.lower() for keyword in ["sleep", "rest"]):
            response = self._generate_sleep_insights()
        elif any(keyword in query.lower() for keyword in ["train", "workout", "exercise"]):
            response = self._generate_training_insights()
        elif any(keyword in query.lower() for keyword in ["recover", "recovery"]):
            response = self._generate_recovery_insights()
        elif any(keyword in query.lower() for keyword in ["compare", "trend", "progress"]):
            response = self._generate_trend_analysis()
        elif any(keyword in query.lower() for keyword in ["simulate", "scenario", "what if"]):
            scenario = "overtraining" if "overtrain" in query.lower() else "recovery"
            response = self._simulate_scenario(scenario)
        else:
            # Default response
            response = {
                "text": "I can help you understand your wearable data and provide personalized insights. "
                        "Try asking about your sleep, training, recovery, or overall health status. "
                        "I can also simulate different scenarios to help you understand potential outcomes.",
                "visualization": None
            }
        
        # Add response to conversation history
        self.conversation_history.append({"role": "assistant", "content": response["text"]})
        
        return response
    
    def _generate_overview_response(self) -> Dict[str, Any]:
        """Generate an overview of the user's health status."""
        # Calculate averages for key metrics
        hrv_avg = self.user_data['hrv_rmssd'].mean()
        sleep_avg = self.user_data['sleep_quality'].mean()
        recovery_avg = self.user_data['recovery_score'].mean()
        
        # Determine overall status based on averages
        if hrv_avg > 70 and sleep_avg > 0.8 and recovery_avg > 80:
            status = "excellent"
        elif hrv_avg > 60 and sleep_avg > 0.7 and recovery_avg > 70:
            status = "good"
        elif hrv_avg > 50 and sleep_avg > 0.6 and recovery_avg > 60:
            status = "fair"
        else:
            status = "needs improvement"
        
        # Generate response text
        response_text = (
            f"Based on your recent data, your overall health status is {status}.\n\n"
            f"• Heart Rate Variability: {hrv_avg:.1f} ms (RMSSD)\n"
            f"• Sleep Quality: {sleep_avg:.2f}/1.0\n"
            f"• Recovery Score: {recovery_avg:.1f}/100\n\n"
            "Your HRV is a key indicator of autonomic nervous system balance. "
            "Sleep quality reflects how restorative your sleep is. "
            "Recovery score combines multiple metrics to assess your body's readiness for activity."
        )
        
        return {
            "text": response_text,
            "visualization": None
        }
    
    def _generate_sleep_insights(self) -> Dict[str, Any]:
        """Generate insights about the user's sleep patterns."""
        # Run a simulation focused on sleep
        self.simulator.set_simulation_parameters("sleep_optimization", days=7)
        sim_data, sim_insights = self.simulator.run_simulation()
        
        # Create a visualization
        fig = self._create_sleep_visualization(sim_data)
        
        # Save the visualization
        output_path = 'outputs/examples/conversations/sleep_insights.html'
        fig.write_html(output_path)
        
        # Extract sleep-related insights
        sleep_insights = [i for i in sim_insights if i.get('category') == 'sleep']
        
        if sleep_insights:
            insight = sleep_insights[0]
            response_text = (
                f"Sleep Insight: {insight.get('summary', 'Sleep Analysis')}\n\n"
                f"{insight.get('detail', '')}\n\n"
                "Recommendations:\n"
            )
            for i, rec in enumerate(insight.get('recommendations', []), 1):
                response_text += f"{i}. {rec}\n"
        else:
            response_text = (
                "Based on your sleep data, I've identified some patterns:\n\n"
                "• Your sleep quality varies significantly throughout the week\n"
                "• There's a strong correlation between your sleep quality and next-day recovery score\n"
                "• Your best sleep typically occurs after moderate activity days\n\n"
                "Try maintaining a consistent sleep schedule and limiting screen time before bed to improve your sleep quality."
            )
        
        return {
            "text": response_text,
            "visualization": output_path
        }
    
    def _generate_training_insights(self) -> Dict[str, Any]:
        """Generate insights about the user's training patterns."""
        # Run a simulation focused on training
        self.simulator.set_simulation_parameters("training_optimization", days=14)
        sim_data, sim_insights = self.simulator.run_simulation()
        
        # Create a visualization
        fig = self._create_training_visualization(sim_data)
        
        # Save the visualization
        output_path = 'outputs/examples/conversations/training_insights.html'
        fig.write_html(output_path)
        
        # Extract training-related insights
        training_insights = [i for i in sim_insights if i.get('category') == 'training']
        
        if training_insights:
            insight = training_insights[0]
            response_text = (
                f"Training Insight: {insight.get('summary', 'Training Analysis')}\n\n"
                f"{insight.get('detail', '')}\n\n"
                "Recommendations:\n"
            )
            for i, rec in enumerate(insight.get('recommendations', []), 1):
                response_text += f"{i}. {rec}\n"
        else:
            response_text = (
                "I've analyzed your training patterns and found some interesting insights:\n\n"
                "• Your high-intensity days are often followed by significant drops in recovery metrics\n"
                "• Your body typically requires 48-72 hours to fully recover from intense sessions\n"
                "• You perform best when you alternate between high and low-intensity days\n\n"
                "Consider implementing a more structured training plan with proper periodization to optimize your performance."
            )
        
        return {
            "text": response_text,
            "visualization": output_path
        }
    
    def _generate_recovery_insights(self) -> Dict[str, Any]:
        """Generate insights about the user's recovery patterns."""
        # Run a simulation focused on recovery
        self.simulator.set_simulation_parameters("recovery_phase", days=10)
        sim_data, sim_insights = self.simulator.run_simulation()
        
        # Create a visualization
        fig = self._create_recovery_visualization(sim_data)
        
        # Save the visualization
        output_path = 'outputs/examples/conversations/recovery_insights.html'
        fig.write_html(output_path)
        
        # Extract recovery-related insights
        recovery_insights = [i for i in sim_insights if i.get('category') == 'recovery']
        
        if recovery_insights:
            insight = recovery_insights[0]
            response_text = (
                f"Recovery Insight: {insight.get('summary', 'Recovery Analysis')}\n\n"
                f"{insight.get('detail', '')}\n\n"
                "Recommendations:\n"
            )
            for i, rec in enumerate(insight.get('recommendations', []), 1):
                response_text += f"{i}. {rec}\n"
        else:
            response_text = (
                "Looking at your recovery patterns, I've noticed:\n\n"
                "• Your HRV is a strong predictor of your recovery status\n"
                "• You recover faster when you prioritize sleep and hydration\n"
                "• Active recovery days (light activity) seem to accelerate your recovery process\n\n"
                "To optimize your recovery, focus on quality sleep, proper nutrition, and strategic active recovery sessions."
            )
        
        return {
            "text": response_text,
            "visualization": output_path
        }
    
    def _generate_trend_analysis(self) -> Dict[str, Any]:
        """Generate a trend analysis of the user's data over time."""
        # Create an interactive timeline visualization
        output_path = 'outputs/examples/conversations/trend_analysis.html'
        
        # Use the InteractiveTimeline class directly
        timeline = InteractiveTimeline()
        timeline.create_interactive_timeline(
            df=self.user_data,
            metrics=['hrv_rmssd', 'sleep_quality', 'recovery_score', 'strain'],
            title="Your Health Metrics Over Time",
            output_path=output_path
        )
        
        # Calculate trends
        hrv_trend = self._calculate_trend(self.user_data['hrv_rmssd'])
        sleep_trend = self._calculate_trend(self.user_data['sleep_quality'])
        recovery_trend = self._calculate_trend(self.user_data['recovery_score'])
        
        response_text = (
            "I've analyzed your data trends over time:\n\n"
            f"• HRV: {hrv_trend['direction']} trend ({hrv_trend['percentage']:.1f}% {hrv_trend['direction']})\n"
            f"• Sleep Quality: {sleep_trend['direction']} trend ({sleep_trend['percentage']:.1f}% {sleep_trend['direction']})\n"
            f"• Recovery: {recovery_trend['direction']} trend ({recovery_trend['percentage']:.1f}% {recovery_trend['direction']})\n\n"
            "The interactive visualization shows how your metrics have changed over time. "
            "You can hover over data points for details and use the range slider to focus on specific time periods."
        )
        
        return {
            "text": response_text,
            "visualization": output_path
        }
    
    def _simulate_scenario(self, scenario_type: str) -> Dict[str, Any]:
        """Simulate a specific scenario and provide insights."""
        # Run the simulation
        self.simulator.set_simulation_parameters(scenario_type, days=14)
        sim_data, sim_insights = self.simulator.run_simulation()
        
        # Create a visualization
        output_path = f'outputs/examples/conversations/{scenario_type}_simulation.html'
        
        # Use the TimelineIntegration class to visualize the simulation
        timeline_integration = TimelineIntegration()
        result_paths = timeline_integration.visualize_simulation(
            simulation_data=sim_data,
            insights=sim_insights,
            scenario_type=scenario_type,
            output_dir='outputs/examples/conversations'
        )
        
        # Use the interactive visualization path
        if 'interactive' in result_paths:
            output_path = result_paths['interactive']
        
        # Format the response
        if scenario_type == "overtraining":
            response_text = (
                "I've simulated an overtraining scenario based on your data. Here's what would likely happen:\n\n"
                "• Initial performance gains in the first 3-4 days\n"
                "• Significant drop in HRV and recovery scores after day 5\n"
                "• Sleep quality deterioration starting around day 7\n"
                "• Potential performance plateau or decline after day 10\n\n"
                "To avoid overtraining, ensure you're balancing intense workouts with adequate recovery periods. "
                "Monitor your HRV closely - a consistent downward trend is an early warning sign."
            )
        else:  # recovery scenario
            response_text = (
                "I've simulated a focused recovery phase based on your data. Here's what you could expect:\n\n"
                "• Gradual increase in HRV over the first 5-7 days\n"
                "• Sleep quality improvements starting around day 3\n"
                "• Recovery score reaching optimal levels by day 8-10\n"
                "• Readiness for high-performance training after the recovery phase\n\n"
                "This simulation shows the benefits of a structured recovery period. Consider implementing "
                "a recovery week every 4-6 weeks of training to prevent burnout and optimize performance."
            )
        
        return {
            "text": response_text,
            "visualization": output_path
        }
    
    def _create_overview_visualization(self) -> go.Figure:
        """Create an overview visualization of the user's health metrics."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("HRV Trend", "Sleep Quality", "Recovery Score", "Activity Level"),
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "scatter"}]]
        )
        
        # Add traces for each metric
        fig.add_trace(
            go.Scatter(x=self.user_data.index, y=self.user_data['hrv_rmssd'], mode='lines+markers', name='HRV'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=self.user_data.index, y=self.user_data['sleep_quality'], mode='lines+markers', name='Sleep Quality'),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatter(x=self.user_data.index, y=self.user_data['recovery_score'], mode='lines+markers', name='Recovery Score'),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=self.user_data.index, y=self.user_data['strain'], mode='lines+markers', name='Activity Level'),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title="Your Health Metrics Overview",
            height=800,
            showlegend=False,
            template="plotly_white"
        )
        
        return fig
    
    def _create_sleep_visualization(self, data: pd.DataFrame) -> go.Figure:
        """Create a sleep-focused visualization."""
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Sleep Quality vs. Recovery", "Sleep Duration and Efficiency"),
            specs=[[{"type": "scatter"}], [{"type": "bar"}]],
            row_heights=[0.6, 0.4]
        )
        
        # Add traces
        fig.add_trace(
            go.Scatter(x=data.index, y=data['sleep_quality'], mode='lines+markers', name='Sleep Quality'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=data.index, y=data['recovery_score'], mode='lines+markers', name='Recovery Score'),
            row=1, col=1
        )
        
        # Generate some synthetic sleep duration and efficiency data
        np.random.seed(42)
        sleep_duration = np.random.normal(7.5, 0.5, len(data))
        sleep_efficiency = np.random.normal(85, 5, len(data))
        
        # Add bar chart for sleep duration and efficiency
        fig.add_trace(
            go.Bar(x=data.index, y=sleep_duration, name='Sleep Duration (hours)'),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=data.index, y=sleep_efficiency, mode='lines+markers', name='Sleep Efficiency (%)'),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            title="Sleep Analysis",
            height=800,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            template="plotly_white"
        )
        
        return fig
    
    def _create_training_visualization(self, data: pd.DataFrame) -> go.Figure:
        """Create a visualization of training patterns."""
        # Create figure with subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Activity Level Over Time", 
                "Recovery vs. Activity",
                "Activity Distribution", 
                "Weekly Activity Pattern"
            ),
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "bar"}, {"type": "bar"}]
            ]
        )
        
        # Add traces for each subplot
        fig.add_trace(
            go.Scatter(x=data.index, y=data['strain'], mode='lines+markers', name='Activity Level'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=data['strain'], y=data['recovery_score'], mode='markers', name='Recovery vs. Activity',
                      marker=dict(size=10, color=data.index, colorscale='Viridis', showscale=True)),
            row=1, col=2
        )
        
        # Create activity distribution
        activity_bins = pd.cut(data['strain'], bins=5, labels=['Very Low', 'Low', 'Moderate', 'High', 'Very High'])
        activity_counts = activity_bins.value_counts().sort_index()
        
        fig.add_trace(
            go.Bar(x=activity_counts.index, y=activity_counts.values, name='Activity Distribution'),
            row=2, col=1
        )
        
        # Create weekly pattern - check if we have date information
        if 'date' in data.columns:
            # Convert date column to datetime if it's not already
            if not pd.api.types.is_datetime64_any_dtype(data['date']):
                data['date'] = pd.to_datetime(data['date'])
            
            # Get day of week
            data['day_of_week'] = data['date'].dt.day_name()
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            weekly_activity = data.groupby('day_of_week')['strain'].mean().reindex(day_order)
            
            fig.add_trace(
                go.Bar(x=weekly_activity.index, y=weekly_activity.values, name='Weekly Pattern'),
                row=2, col=2
            )
        else:
            # If no date column, just show a simple bar chart of strain values
            fig.add_trace(
                go.Bar(x=list(range(1, len(data) + 1)), y=data['strain'], name='Activity Levels'),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=800,
            width=1000,
            title_text="Your Training Patterns Analysis",
            showlegend=False
        )
        
        return fig
    
    def _create_recovery_visualization(self, data: pd.DataFrame) -> go.Figure:
        """Create a recovery-focused visualization."""
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Recovery Metrics Over Time", "Recovery Score vs. HRV"),
            specs=[[{"type": "scatter"}], [{"type": "scatter"}]],
            row_heights=[0.5, 0.5]
        )
        
        # Add traces for recovery metrics
        fig.add_trace(
            go.Scatter(x=data.index, y=data['recovery_score'], mode='lines+markers', name='Recovery Score'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=data.index, y=data['hrv_rmssd'], mode='lines+markers', name='HRV'),
            row=1, col=1
        )
        
        # Add scatter plot for recovery score vs. HRV
        fig.add_trace(
            go.Scatter(x=data['hrv_rmssd'], y=data['recovery_score'], mode='markers', name='Recovery vs. HRV',
                      marker=dict(size=10, color=data.index, colorscale='Viridis', showscale=True)),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            title="Recovery Analysis",
            height=800,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            template="plotly_white"
        )
        
        # Update axis labels for the scatter plot
        fig.update_xaxes(title_text="HRV (ms)", row=2, col=1)
        fig.update_yaxes(title_text="Recovery Score (%)", row=2, col=1)
        
        return fig
    
    def _calculate_trend(self, data_series: pd.Series) -> Dict[str, Any]:
        """Calculate the trend direction and percentage change for a data series."""
        # Use a simple linear regression to determine trend
        x = np.arange(len(data_series))
        y = data_series.values
        
        # Calculate slope
        slope = np.polyfit(x, y, 1)[0]
        
        # Calculate percentage change
        start_value = data_series.iloc[0]
        end_value = data_series.iloc[-1]
        percentage_change = ((end_value - start_value) / start_value) * 100
        
        return {
            "direction": "increasing" if slope > 0 else "decreasing",
            "percentage": abs(percentage_change)
        }
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get the conversation history."""
        return self.conversation_history
    
    def simulate_conversation(self) -> None:
        """Simulate a conversation with predefined queries."""
        queries = [
            "Can you give me an overview of my health status?",
            "How's my sleep quality looking?",
            "What can you tell me about my training patterns?",
            "I'm feeling tired. Any recovery insights?",
            "How have my metrics been trending over the past month?",
            "What would happen if I overtrain for the next two weeks?",
            "Can you simulate a recovery phase for me?"
        ]
        
        print("=== Simulating User Conversation ===\n")
        
        for query in queries:
            print(f"User: {query}")
            response = self.process_user_query(query)
            print(f"Assistant: {response['text']}")
            print(f"[Visualization available at: {response['visualization']}]")
            print("\n" + "-"*50 + "\n")


def main():
    """Main function to demonstrate the conversational interface."""
    print("Initializing Conversational Agent for Wearable Data Insight Generator...")
    
    # Create the conversational agent
    agent = ConversationalAgent()
    
    # Set up user data
    print("Generating user data...")
    agent.set_user_data(profile_type="athlete", days=30)
    
    # Simulate a conversation
    agent.simulate_conversation()
    
    print("\nConversation simulation complete. Visualizations are available in the outputs/examples/conversations directory.")


if __name__ == "__main__":
    main()
