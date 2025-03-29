import os
import json
from datetime import datetime, date
from typing import Dict, List, Optional, Any

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.data_loader import DataLoader
from src.feature_engineer import FeatureEngineer
from src.insight_prompt_builder import InsightPromptBuilder
from src.llm_engine import LLMEngine
from src.agent_simulator import AgentSimulator
from src.services.supabase_data_service import SupabaseDataService
from src.adapters.supabase_adapter import SupabaseSimulatorAdapter, SupabaseInsightAdapter
from src.visualization.timeline_interactive import InteractiveTimeline

# Initialize FastAPI app
app = FastAPI(
    title="Wearable LLM Insight Generator",
    description="API for generating insights from wearable device data using LLMs",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Default user ID for demo purposes
DEFAULT_USER_ID = "demo_user_001"

# Initialize services
try:
    supabase_service = SupabaseDataService()
    simulator_adapter = SupabaseSimulatorAdapter(user_id=DEFAULT_USER_ID)
    insight_adapter = SupabaseInsightAdapter(user_id=DEFAULT_USER_ID)
    print("Supabase services initialized successfully")
except Exception as e:
    print(f"Warning: Supabase services could not be initialized: {e}")
    supabase_service = None
    simulator_adapter = None
    insight_adapter = None

# Initialize LLM Engine with local model
llm_engine = LLMEngine()
print(f"LLM Engine initialized with default provider: {llm_engine.config.get('default_provider', 'local')}")

# Pydantic models
class ChatMessage(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    timestamp: str

class DashboardData(BaseModel):
    metrics: Dict[str, Any]
    daily_insight: str
    weekly_trends: List[str]
    
class DeviceConnection(BaseModel):
    device_type: str
    device_id: str
    sync_frequency: str
    metrics_to_sync: List[str]

class VisualizationSettings(BaseModel):
    metrics: List[str]
    time_range: str
    chart_type: str
    show_insights: bool

# Routes
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Redirect to dashboard page"""
    return RedirectResponse(url="/dashboard")

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Render the dashboard page"""
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request):
    """Render the chat interface page"""
    return templates.TemplateResponse("chat.html", {"request": request})

@app.get("/device-sync", response_class=HTMLResponse)
async def device_sync_page(request: Request):
    """Render the device sync page"""
    return templates.TemplateResponse("device_sync.html", {"request": request})

@app.get("/visualizations", response_class=HTMLResponse)
async def visualizations_page(request: Request):
    """Render the visualizations page"""
    return templates.TemplateResponse("visualizations.html", {"request": request})

@app.post("/api/chat", response_model=ChatResponse)
async def chat(message: ChatMessage):
    """
    Process a chat message and return a response.
    """
    try:
        # Create prompt for LLM
        prompt_builder = InsightPromptBuilder()
        prompt = prompt_builder.create_chat_prompt(message.message)
        
        # Generate response using local LLM
        response_text, metadata = llm_engine.generate_insight(prompt, provider="local")
        
        # Format response
        response = ChatResponse(
            response=response_text,
            timestamp=datetime.now().isoformat()
        )
        
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat message: {str(e)}")

@app.get("/api/dashboard/data", response_model=DashboardData)
async def get_dashboard_data():
    """
    Get data for the dashboard.
    """
    try:
        # Mock data for demonstration
        metrics = {
            "sleep": {
                "value": 7.5,
                "unit": "hours",
                "trend": "+0.5",
                "status": "good"
            },
            "hrv": {
                "value": 65,
                "unit": "ms",
                "trend": "+3",
                "status": "good"
            },
            "steps": {
                "value": 8742,
                "unit": "steps",
                "trend": "-523",
                "status": "neutral"
            },
            "recovery": {
                "value": 85,
                "unit": "%",
                "trend": "+5",
                "status": "excellent"
            }
        }
        
        # Generate daily insight using local LLM
        prompt_builder = InsightPromptBuilder()
        prompt = prompt_builder.create_daily_insight_prompt(metrics)
        daily_insight, _ = llm_engine.generate_insight(prompt, provider="local")
        
        # Mock weekly trends
        weekly_trends = [
            "Your sleep consistency has improved by 12% this week",
            "HRV is trending upward, indicating improved recovery",
            "Activity levels are consistent with your goals"
        ]
        
        return DashboardData(
            metrics=metrics,
            daily_insight=daily_insight,
            weekly_trends=weekly_trends
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting dashboard data: {str(e)}")

@app.post("/api/device/connect")
async def connect_device(connection: DeviceConnection):
    """
    Connect a wearable device.
    """
    try:
        # In a real implementation, this would initiate OAuth flow
        # and store device connection details
        
        # For demo, just return success with mock data
        return {
            "status": "connected",
            "device": {
                "type": connection.device_type,
                "id": connection.device_id,
                "name": f"{connection.device_type} Device",
                "last_sync": datetime.now().isoformat(),
                "battery": "92%",
                "metrics_available": connection.metrics_to_sync
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error connecting device: {str(e)}")

@app.post("/api/device/sync/{device_id}")
async def sync_data(device_id: str):
    """
    Trigger a data sync from the connected device.
    """
    try:
        # In a real implementation, this would fetch data from the device API
        
        # For demo, return mock sync results
        return {
            "status": "completed",
            "device_id": device_id,
            "sync_time": datetime.now().isoformat(),
            "metrics_synced": ["sleep", "hrv", "steps", "recovery"],
            "days_synced": 7,
            "new_insights": 3
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error syncing data: {str(e)}")

@app.post("/api/visualizations/settings")
async def save_visualization(settings: VisualizationSettings):
    """
    Save visualization settings.
    """
    try:
        # In a real implementation, this would save user preferences
        
        # For demo, just acknowledge the settings
        return {
            "status": "saved",
            "settings": settings.dict(),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving visualization settings: {str(e)}")

@app.get("/api/visualizations/data")
async def get_visualization_data(metrics: List[str], time_range: str):
    """
    Get data for visualizations.
    """
    try:
        # In a real implementation, this would fetch data from the database
        
        # For demo, generate mock time series data
        end_date = date.today()
        
        if time_range == "week":
            days = 7
        elif time_range == "month":
            days = 30
        elif time_range == "year":
            days = 365
        else:
            days = 7  # Default to week
        
        # Generate mock data points
        data_points = []
        for i in range(days):
            current_date = end_date - datetime.timedelta(days=days-i-1)
            data_point = {
                "date": current_date.isoformat(),
                "metrics": {}
            }
            
            # Add requested metrics
            for metric in metrics:
                if metric == "sleep":
                    data_point["metrics"][metric] = round(6.5 + 2 * (0.5 - (i % 3) * 0.1), 1)
                elif metric == "hrv":
                    data_point["metrics"][metric] = 55 + (i % 5) * 3
                elif metric == "steps":
                    data_point["metrics"][metric] = 7500 + (i % 7) * 500
                elif metric == "recovery":
                    data_point["metrics"][metric] = 70 + (i % 10) * 2
            
            data_points.append(data_point)
        
        return {
            "time_range": time_range,
            "metrics": metrics,
            "data_points": data_points
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting visualization data: {str(e)}")


# Run the application
if __name__ == "__main__":
    # Check if local LLM is configured
    print(f"Using LLM provider: {llm_engine.config.get('default_provider', 'local')}")
    print(f"Local model: {llm_engine.config.get('local', {}).get('model', 'Not configured')}")
    
    # Run the FastAPI app
    uvicorn.run(app, host="0.0.0.0", port=8000)
