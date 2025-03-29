"""
Supabase Integration Example for Wearable Data Insight Generator

This example demonstrates how to use the Supabase integration with the
Wearable Data Insight Generator for data storage, user management,
and interactive visualizations.
"""

import os
import sys
import asyncio
import pandas as pd
from datetime import date, datetime, timedelta
import logging
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config.supabase_config import get_supabase_client
from src.services.supabase_data_service import SupabaseDataService
from src.adapters.supabase_adapter import SupabaseSimulatorAdapter, SupabaseInsightAdapter
from src.demo_data_generator import DemoDataGenerator
from src.visualization.timeline_interactive import InteractiveTimeline

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

async def create_demo_user():
    """Create a demo user in Supabase and return the user ID."""
    data_service = SupabaseDataService()
    
    # Generate a unique email for the demo user
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    email = f"demo_user_{timestamp}@example.com"
    
    # Create user profile data
    profile_data = {
        "first_name": "Demo",
        "last_name": "User",
        "birth_date": date(1990, 1, 1),
        "gender": "other",
        "height": 175,
        "weight": 70,
        "activity_level": "active",
        "preferences": {
            "theme": "light",
            "metrics_to_show": ["hrv_rmssd", "sleep_quality", "recovery_score", "strain"],
            "notification_preferences": {
                "email": True,
                "push": False
            }
        }
    }
    
    # Create or get user
    user = await data_service.get_or_create_user(email, profile_data)
    logger.info(f"Demo user created with ID: {user['id']}")
    
    return user['id']

async def generate_and_store_demo_data(user_id):
    """Generate demo data and store it in Supabase."""
    # Create demo data generator
    demo_generator = DemoDataGenerator()
    
    # Generate demo dataset
    user_profile, time_series_data, scenario = demo_generator.generate_demo_dataset(
        profile_type="athlete",
        days=30,
        scenario="recovery_phase"
    )
    
    # Convert DataFrame to list of dictionaries
    metrics_records = time_series_data.to_dict('records')
    
    # Create data service
    data_service = SupabaseDataService()
    
    # Sync metrics data
    sync_count = await data_service.sync_metrics_data(user_id, metrics_records)
    logger.info(f"Synced {sync_count} metrics records for user {user_id}")
    
    return time_series_data

async def generate_insights_from_metrics(user_id):
    """Generate insights from metrics data and store them in Supabase."""
    # Create insight adapter
    insight_adapter = SupabaseInsightAdapter(user_id)
    
    # Generate insights
    insights = insight_adapter.generate_insights_from_supabase_data(days=30)
    
    logger.info(f"Generated and stored {len(insights)} insights for user {user_id}")
    
    return insights

async def create_interactive_visualization(user_id):
    """Create an interactive visualization of user data from Supabase."""
    # Create data service
    data_service = SupabaseDataService()
    
    # Get user data for visualization
    metrics_df, insights = await data_service.get_user_data_for_visualization(user_id, days=30)
    
    if metrics_df.empty:
        logger.error("No metrics data found for visualization")
        return None
    
    # Create output directory
    os.makedirs("outputs/supabase_example", exist_ok=True)
    
    # Create interactive timeline
    timeline = InteractiveTimeline()
    fig = timeline.create_interactive_timeline(
        df=metrics_df,
        insights=insights,
        title="Wearable Data Insights from Supabase",
        output_path="outputs/supabase_example/supabase_timeline.html"
    )
    
    logger.info("Created interactive visualization at outputs/supabase_example/supabase_timeline.html")
    
    return fig

async def run_simulation_with_supabase(user_id):
    """Run a simulation and store results in Supabase."""
    # Create simulator adapter
    simulator_adapter = SupabaseSimulatorAdapter(user_id)
    
    # Initialize simulator from Supabase user data
    success = simulator_adapter.initialize_simulator_from_supabase()
    
    if not success:
        logger.error("Failed to initialize simulator from Supabase")
        return None, None
    
    # Run simulation and store results
    metrics_df, insights = simulator_adapter.run_simulation_and_store_results(
        scenario_type="training_peak",
        days=14
    )
    
    logger.info(f"Simulation completed with {len(metrics_df)} metrics records and {len(insights)} insights")
    
    return metrics_df, insights

async def get_user_health_overview(user_id):
    """Get a health overview for a user from Supabase."""
    # Create data service
    data_service = SupabaseDataService()
    
    # Get health overview
    overview = await data_service.get_user_health_overview(user_id)
    
    logger.info(f"Retrieved health overview for user {user_id}")
    
    return overview

async def main():
    """Main function to run the example."""
    try:
        # Check if Supabase is configured
        supabase = get_supabase_client()
        logger.info("Supabase client initialized successfully")
        
        # Create demo user
        user_id = await create_demo_user()
        
        # Generate and store demo data
        time_series_data = await generate_and_store_demo_data(user_id)
        
        # Generate insights
        insights = await generate_insights_from_metrics(user_id)
        
        # Create visualization
        fig = await create_interactive_visualization(user_id)
        
        # Run simulation
        sim_metrics, sim_insights = await run_simulation_with_supabase(user_id)
        
        # Get health overview
        overview = await get_user_health_overview(user_id)
        
        # Print health overview
        print("\n=== User Health Overview ===")
        print(f"Status: {overview['status']}")
        print("\nLatest Metrics:")
        for metric, value in overview['metrics'].items():
            if value is not None:
                print(f"  • {metric}: {value}")
        
        print("\nTrends:")
        for metric, trend in overview['trends'].items():
            print(f"  • {metric}: {trend}")
        
        print("\nRecent Insights:")
        for insight in overview['insights'][:3]:  # Show top 3 insights
            print(f"  • {insight['title']}: {insight['summary']}")
        
        print("\nVisualization available at: outputs/supabase_example/supabase_timeline.html")
        
    except Exception as e:
        logger.error(f"Error in Supabase example: {str(e)}")
        raise

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
