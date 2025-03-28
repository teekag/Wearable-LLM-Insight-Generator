"""
FastAPI Application for Wearable Data Insight Generator

This module provides a REST API for generating insights from wearable device data.
It exposes endpoints for data upload, insight generation, and agent-based coaching.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Union, Any
from datetime import datetime

import pandas as pd
from fastapi import FastAPI, HTTPException, Depends, Body, Query, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Import core modules
import sys
sys.path.append("..")  # Add parent directory to path
from src.insight_engine import InsightEngine, UserConfig, InsightType, Insight
from src.data_utils import generate_synthetic_data

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Wearable Data Insight Generator API",
    description="API for generating personalized insights from wearable device data",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Initialize insight engine
insight_engine = InsightEngine()
memory_store = {}  # Simple in-memory store for user data

# Define API models
class DataUploadRequest(BaseModel):
    """Request model for data upload"""
    data: Dict[str, Any] = Field(..., description="Wearable data in JSON format")
    user_id: str = Field(..., description="User identifier")
    data_source: str = Field(default="manual", description="Source of the data")

class InsightRequest(BaseModel):
    """Request model for insight generation"""
    user_id: str = Field(..., description="User identifier")
    data: Optional[Dict[str, Any]] = Field(default=None, description="Wearable data (optional)")
    persona: str = Field(default="athlete", description="User persona")
    goals: List[str] = Field(default_factory=list, description="User goals")
    insight_categories: List[str] = Field(default=["recovery", "training", "sleep"], 
                                         description="Categories of insights to generate")

class ChatRequest(BaseModel):
    """Request model for agent chat"""
    user_id: str = Field(..., description="User identifier")
    message: str = Field(..., description="User message")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Additional context")

class ChatResponse(BaseModel):
    """Response model for agent chat"""
    response: str = Field(..., description="Agent response")
    insights: List[Dict[str, Any]] = Field(default_factory=list, description="Related insights")
    suggestions: List[str] = Field(default_factory=list, description="Suggested actions or questions")

# API endpoints
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Wearable Data Insight Generator API",
        "version": "1.0.0",
        "endpoints": [
            "/api/upload-data",
            "/api/generate-insight",
            "/api/chat-agent",
            "/api/demo-data"
        ]
    }

@app.post("/api/upload-data")
async def upload_data(request: DataUploadRequest):
    """
    Upload wearable data for a user
    
    This endpoint accepts wearable data in JSON format and stores it for future insight generation.
    """
    try:
        # Convert data to DataFrame
        df = pd.DataFrame.from_dict(request.data)
        
        # Store in memory (in production, would use a database)
        if request.user_id not in memory_store:
            memory_store[request.user_id] = {"data": [], "insights": []}
        
        memory_store[request.user_id]["data"].append({
            "timestamp": datetime.now().isoformat(),
            "source": request.data_source,
            "data": request.data
        })
        
        return {"status": "success", "message": "Data uploaded successfully"}
        
    except Exception as e:
        logger.error(f"Error uploading data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error uploading data: {str(e)}")

@app.post("/api/generate-insight")
async def generate_insight(request: InsightRequest):
    """
    Generate insights from wearable data
    
    This endpoint processes wearable data and generates personalized insights based on
    user preferences and goals.
    """
    try:
        # Get data from request or memory store
        if request.data:
            data = request.data
        elif request.user_id in memory_store and memory_store[request.user_id]["data"]:
            # Use the most recent data
            data = memory_store[request.user_id]["data"][-1]["data"]
        else:
            # Generate synthetic data if no real data is available
            synthetic_df = generate_synthetic_data(days=30, user_type=request.persona)
            data = synthetic_df.to_dict(orient="records")
        
        # Convert insight categories to InsightType objects
        insight_types = [
            InsightType(category=category, priority=i+1)
            for i, category in enumerate(request.insight_categories)
        ]
        
        # Create user config
        user_config = UserConfig(
            persona=request.persona,
            goals=request.goals,
            preferences={"user_id": request.user_id}
        )
        
        # Generate insights
        insights = insight_engine.generate_insight(
            data=data,
            user_config=user_config,
            insight_types=insight_types,
            data_format="dict"
        )
        
        # Convert insights to dict for JSON response
        insight_dicts = [insight.dict() for insight in insights]
        
        # Store insights in memory
        if request.user_id not in memory_store:
            memory_store[request.user_id] = {"data": [], "insights": []}
        memory_store[request.user_id]["insights"].extend(insight_dicts)
        
        return {"insights": insight_dicts}
        
    except Exception as e:
        logger.error(f"Error generating insights: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating insights: {str(e)}")

@app.post("/api/chat-agent")
async def chat_agent(request: ChatRequest):
    """
    Interact with the coaching agent
    
    This endpoint provides a conversational interface to the insight system,
    allowing users to ask questions and get personalized coaching.
    """
    try:
        # Get user history if available
        user_history = []
        if request.user_id in memory_store:
            user_history = memory_store[request.user_id].get("insights", [])
        
        # Simple response generation (in production, would use a more sophisticated agent)
        if "how am i doing" in request.message.lower():
            response = "Based on your recent data, your recovery looks good. Your HRV is trending upward, which is a positive sign."
            suggestions = ["Tell me more about my sleep", "What should I focus on today?", "How can I improve my recovery?"]
        elif "sleep" in request.message.lower():
            response = "Your sleep quality has been averaging 85% this week, which is excellent. Keep maintaining your consistent sleep schedule."
            suggestions = ["How does this affect my training?", "What's my optimal sleep duration?", "Show me my sleep trends"]
        elif "train" in request.message.lower() or "workout" in request.message.lower():
            response = "Given your current recovery status, you could increase training intensity by 10-15% today. Focus on strength training as your HRV indicates good readiness."
            suggestions = ["What exercises do you recommend?", "Should I do cardio today?", "When should I rest next?"]
        else:
            response = "I'm here to help you understand your wearable data and provide personalized insights. What would you like to know about your health and fitness?"
            suggestions = ["How am I doing overall?", "What does my sleep data show?", "Should I train hard today?"]
        
        # Find relevant insights
        relevant_insights = []
        if user_history:
            # Simple keyword matching (would use semantic search in production)
            keywords = request.message.lower().split()
            for insight in user_history:
                if any(keyword in insight.get("summary", "").lower() for keyword in keywords):
                    relevant_insights.append(insight)
        
        return ChatResponse(
            response=response,
            insights=relevant_insights[:2],  # Limit to 2 most relevant insights
            suggestions=suggestions
        )
        
    except Exception as e:
        logger.error(f"Error in chat agent: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in chat agent: {str(e)}")

@app.get("/api/demo-data")
async def get_demo_data(user_type: str = Query("athlete", description="Type of user profile")):
    """
    Get synthetic demo data
    
    This endpoint generates synthetic wearable data for demonstration purposes.
    """
    try:
        # Generate synthetic data
        df = generate_synthetic_data(days=30, user_type=user_type)
        
        # Convert to dict for JSON response
        data = df.to_dict(orient="records")
        
        return {"data": data}
        
    except Exception as e:
        logger.error(f"Error generating demo data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating demo data: {str(e)}")

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
