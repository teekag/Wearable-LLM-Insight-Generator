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

async def run_simulation_with_supabase(user_id):
    """Run a simulation and store results in Supabase."""
    # Create simulator adapter
    simulator_adapter = SupabaseSimulatorAdapter(user_id)
    
    # Run simulation and store results
    metrics_df, insights = simulator_adapter.run_simulation_and_store_results(
        scenario_type="overtraining",
        days=14
    )
    
    logger.info(f"Ran simulation with {len(metrics_df)} metrics and {len(insights)} insights")
    
    return metrics_df, insights

async def get_user_health_overview(user_id):
    """Get a health overview for a user from Supabase."""
    # Create data service
    data_service = SupabaseDataService()
    
    # Get user data for visualization
    metrics_df, insights = await data_service.get_user_data_for_visualization(user_id, days=30)
    
    logger.info(f"Retrieved {len(metrics_df)} metrics and {len(insights)} insights for user {user_id}")
    
    return metrics_df, insights

async def create_and_save_visualization(user_id, metrics_df, insights):
    """Create an interactive visualization and save it to Supabase."""
    # Create interactive timeline
    timeline = InteractiveTimeline()
    
    # Create visualization
    fig = timeline.create_insight_timeline(
        metrics_df=metrics_df,
        insights=insights,
        user_id=user_id,
        metrics_to_show=["hrv_rmssd", "sleep_quality", "recovery_score", "strain"],
        insight_types_to_show=["recovery", "sleep", "activity", "strain"]
    )
    
    # Convert to HTML
    html_content = fig.to_html(include_plotlyjs='cdn')
    
    # Save visualization to Supabase
    data_service = SupabaseDataService()
    visualization_id = await data_service.save_visualization(
        user_id=user_id,
        title="30-Day Health Overview",
        description="Interactive visualization of health metrics and insights",
        start_date=date.today() - timedelta(days=30),
        end_date=date.today(),
        metrics_included=["hrv_rmssd", "sleep_quality", "recovery_score", "strain"],
        insight_types_included=["recovery", "sleep", "activity", "strain"],
        html_content=html_content
    )
    
    logger.info(f"Saved visualization with ID: {visualization_id}")
    
    return visualization_id

async def demonstrate_oauth_integration(user_id):
    """Demonstrate OAuth integration with wearable platforms."""
    # In a real application, this would redirect to the OAuth provider's authorization page
    # For this example, we'll just simulate the process
    
    data_service = SupabaseDataService()
    
    # Simulate connecting a Fitbit device
    device_id = await data_service.connect_device(
        user_id=user_id,
        device_type="fitbit",
        device_name="Fitbit Sense",
        auth_token="simulated_auth_token",
        refresh_token="simulated_refresh_token",
        token_expires_at=datetime.now() + timedelta(days=30),
        sync_settings={
            "metrics_to_sync": ["heart_rate", "sleep", "activity"],
            "sync_frequency": "daily"
        }
    )
    
    logger.info(f"Connected Fitbit device with ID: {device_id}")
    
    # List connected devices
    devices = await data_service.list_user_devices(user_id)
    logger.info(f"User has {len(devices)} connected devices")
    
    return devices

async def demonstrate_repository_pattern(user_id):
    """Demonstrate the repository pattern with direct repository access."""
    from src.data.repositories.user_repository import UserRepository
    from src.data.repositories.wearable_metrics_repository import WearableMetricsRepository
    from src.data.repositories.insight_repository import InsightRepository
    
    # Create repositories
    user_repo = UserRepository()
    metrics_repo = WearableMetricsRepository()
    insight_repo = InsightRepository()
    
    # Get user by ID
    user = await user_repo.find_by_id(user_id)
    logger.info(f"Retrieved user: {user['first_name']} {user['last_name']}")
    
    # Get latest metrics
    latest_metrics = await metrics_repo.find_latest_by_user_id(user_id, limit=5)
    logger.info(f"Retrieved {len(latest_metrics)} latest metrics")
    
    # Get insights by type
    recovery_insights = await insight_repo.find_by_user_and_type(
        user_id=user_id,
        insight_type="recovery",
        limit=3
    )
    logger.info(f"Retrieved {len(recovery_insights)} recovery insights")
    
    # Update user preferences
    updated_user = await user_repo.update_user(
        user_id=user_id,
        updates={
            "preferences": {
                **user.get("preferences", {}),
                "last_viewed_insight": recovery_insights[0]["id"] if recovery_insights else None
            }
        }
    )
    
    logger.info(f"Updated user preferences")
    
    return {
        "user": user,
        "latest_metrics": latest_metrics,
        "recovery_insights": recovery_insights
    }

async def main():
    """Main function to run the example."""
    try:
        logger.info("Starting Supabase integration example")
        
        # Step 1: Create a demo user
        logger.info("Step 1: Creating demo user")
        user_id = await create_demo_user()
        
        # Step 2: Generate and store demo data
        logger.info("Step 2: Generating and storing demo data")
        metrics_df = await generate_and_store_demo_data(user_id)
        
        # Step 3: Generate insights from metrics
        logger.info("Step 3: Generating insights from metrics")
        insights = await generate_insights_from_metrics(user_id)
        
        # Step 4: Run a simulation with Supabase
        logger.info("Step 4: Running simulation with Supabase")
        sim_metrics, sim_insights = await run_simulation_with_supabase(user_id)
        
        # Step 5: Get user health overview
        logger.info("Step 5: Getting user health overview")
        overview_metrics, overview_insights = await get_user_health_overview(user_id)
        
        # Step 6: Create and save visualization
        logger.info("Step 6: Creating and saving visualization")
        visualization_id = await create_and_save_visualization(
            user_id, overview_metrics, overview_insights
        )
        
        # Step 7: Demonstrate OAuth integration
        logger.info("Step 7: Demonstrating OAuth integration")
        devices = await demonstrate_oauth_integration(user_id)
        
        # Step 8: Demonstrate repository pattern
        logger.info("Step 8: Demonstrating repository pattern")
        repo_results = await demonstrate_repository_pattern(user_id)
        
        logger.info("Supabase integration example completed successfully")
        
        # Print summary
        print("\n=== Supabase Integration Example Summary ===")
        print(f"Demo User ID: {user_id}")
        print(f"Metrics Generated: {len(metrics_df)}")
        print(f"Insights Generated: {len(insights)}")
        print(f"Simulation Metrics: {len(sim_metrics)}")
        print(f"Simulation Insights: {len(sim_insights)}")
        print(f"Visualization Created: {visualization_id}")
        print(f"Connected Devices: {len(devices)}")
        print("===========================================\n")
        
        # Instructions for next steps
        print("Next Steps:")
        print("1. Create a Supabase project at https://supabase.com")
        print("2. Run the schema SQL in the Supabase SQL Editor")
        print("3. Configure your .env file with Supabase credentials")
        print("4. Run this example again to see the full integration in action")
        
    except Exception as e:
        logger.error(f"Error in main function: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
