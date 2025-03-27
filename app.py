import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Any

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.data_loader import DataLoader
from src.feature_engineer import FeatureEngineer
from src.insight_prompt_builder import InsightPromptBuilder
from src.llm_engine import LLMEngine
from src.agent_simulator import AgentSimulator

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
os.makedirs("templates", exist_ok=True)
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Create HTML template if it doesn't exist
if not os.path.exists("templates/index.html"):
    with open("templates/index.html", "w") as f:
        f.write("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Wearable LLM Insight Generator</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f8f9fa;
            padding-top: 20px;
        }
        .chat-container {
            height: 400px;
            overflow-y: auto;
            border: 1px solid #dee2e6;
            border-radius: 5px;
            padding: 15px;
            background-color: white;
        }
        .user-message {
            background-color: #e9ecef;
            padding: 10px 15px;
            border-radius: 15px;
            margin-bottom: 10px;
            max-width: 80%;
            margin-left: auto;
            text-align: right;
        }
        .agent-message {
            background-color: #007bff;
            color: white;
            padding: 10px 15px;
            border-radius: 15px;
            margin-bottom: 10px;
            max-width: 80%;
        }
        .dashboard-card {
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            padding: 20px;
            margin-bottom: 20px;
        }
        .metric-value {
            font-size: 2rem;
            font-weight: bold;
        }
        .metric-label {
            color: #6c757d;
            font-size: 0.9rem;
        }
        .insights-container {
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            padding: 20px;
            margin-bottom: 20px;
        }
        .nav-tabs .nav-link.active {
            font-weight: bold;
            border-bottom: 3px solid #007bff;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1 class="text-center mb-4">Wearable LLM Insight Generator</h1>
        
        <ul class="nav nav-tabs mb-4" id="myTab" role="tablist">
            <li class="nav-item" role="presentation">
                <button class="nav-link active" id="dashboard-tab" data-bs-toggle="tab" data-bs-target="#dashboard" type="button" role="tab" aria-controls="dashboard" aria-selected="true">Dashboard</button>
            </li>
            <li class="nav-item" role="presentation">
                <button class="nav-link" id="chat-tab" data-bs-toggle="tab" data-bs-target="#chat" type="button" role="tab" aria-controls="chat" aria-selected="false">Chat with Coach</button>
            </li>
            <li class="nav-item" role="presentation">
                <button class="nav-link" id="data-tab" data-bs-toggle="tab" data-bs-target="#data" type="button" role="tab" aria-controls="data" aria-selected="false">Your Data</button>
            </li>
        </ul>
        
        <div class="tab-content" id="myTabContent">
            <!-- Dashboard Tab -->
            <div class="tab-pane fade show active" id="dashboard" role="tabpanel" aria-labelledby="dashboard-tab">
                <div class="row">
                    <div class="col-md-4">
                        <div class="dashboard-card text-center">
                            <div class="metric-value" id="recovery-score">85</div>
                            <div class="metric-label">Recovery Score</div>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="dashboard-card text-center">
                            <div class="metric-value" id="hrv-score">65</div>
                            <div class="metric-label">HRV (ms)</div>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="dashboard-card text-center">
                            <div class="metric-value" id="sleep-score">7.5</div>
                            <div class="metric-label">Sleep (hours)</div>
                        </div>
                    </div>
                </div>
                
                <div class="insights-container mt-4">
                    <h3>Today's Insights</h3>
                    <div id="daily-insights">
                        <p>Loading insights...</p>
                    </div>
                </div>
                
                <div class="insights-container mt-4">
                    <h3>Weekly Trends</h3>
                    <div id="weekly-trends">
                        <p>Loading trends...</p>
                    </div>
                </div>
            </div>
            
            <!-- Chat Tab -->
            <div class="tab-pane fade" id="chat" role="tabpanel" aria-labelledby="chat-tab">
                <div class="chat-container mb-3" id="chat-messages">
                    <div class="agent-message">
                        Hello! I'm your wearable data coach. How can I help you today?
                    </div>
                </div>
                <div class="input-group">
                    <input type="text" class="form-control" id="user-input" placeholder="Type your message...">
                    <button class="btn btn-primary" id="send-button">Send</button>
                </div>
                <div class="mt-3">
                    <p><strong>Suggested questions:</strong></p>
                    <div class="d-flex flex-wrap gap-2" id="suggested-questions">
                        <button class="btn btn-sm btn-outline-secondary">What does my recovery score mean?</button>
                        <button class="btn btn-sm btn-outline-secondary">How can I improve my sleep?</button>
                        <button class="btn btn-sm btn-outline-secondary">Should I train hard today?</button>
                        <button class="btn btn-sm btn-outline-secondary">What trends do you see in my data?</button>
                    </div>
                </div>
            </div>
            
            <!-- Data Tab -->
            <div class="tab-pane fade" id="data" role="tabpanel" aria-labelledby="data-tab">
                <div class="dashboard-card">
                    <h3>Your Wearable Data</h3>
                    <p>This tab would display detailed charts and tables of your wearable data over time.</p>
                    <div id="data-visualization">
                        <p>Loading data visualization...</p>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        document.addEventListener('DOMContentLoaded', function() {
            // Load initial data
            fetchDashboardData();
            
            // Set up chat functionality
            const chatMessages = document.getElementById('chat-messages');
            const userInput = document.getElementById('user-input');
            const sendButton = document.getElementById('send-button');
            
            sendButton.addEventListener('click', sendMessage);
            userInput.addEventListener('keypress', function(e) {
                if (e.key === 'Enter') {
                    sendMessage();
                }
            });
            
            // Set up suggested questions
            const suggestedQuestions = document.querySelectorAll('#suggested-questions button');
            suggestedQuestions.forEach(button => {
                button.addEventListener('click', function() {
                    userInput.value = this.textContent;
                    sendMessage();
                });
            });
            
            function sendMessage() {
                const message = userInput.value.trim();
                if (message === '') return;
                
                // Add user message to chat
                const userMessageDiv = document.createElement('div');
                userMessageDiv.className = 'user-message';
                userMessageDiv.textContent = message;
                chatMessages.appendChild(userMessageDiv);
                
                // Clear input
                userInput.value = '';
                
                // Scroll to bottom
                chatMessages.scrollTop = chatMessages.scrollHeight;
                
                // Send to API and get response
                fetch('/api/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ message: message }),
                })
                .then(response => response.json())
                .then(data => {
                    // Add agent response to chat
                    const agentMessageDiv = document.createElement('div');
                    agentMessageDiv.className = 'agent-message';
                    agentMessageDiv.textContent = data.response;
                    chatMessages.appendChild(agentMessageDiv);
                    
                    // Scroll to bottom
                    chatMessages.scrollTop = chatMessages.scrollHeight;
                })
                .catch(error => {
                    console.error('Error:', error);
                    // Add error message
                    const errorMessageDiv = document.createElement('div');
                    errorMessageDiv.className = 'agent-message';
                    errorMessageDiv.textContent = 'Sorry, there was an error processing your request.';
                    chatMessages.appendChild(errorMessageDiv);
                    
                    // Scroll to bottom
                    chatMessages.scrollTop = chatMessages.scrollHeight;
                });
            }
            
            function fetchDashboardData() {
                fetch('/api/dashboard')
                .then(response => response.json())
                .then(data => {
                    // Update dashboard metrics
                    document.getElementById('recovery-score').textContent = data.metrics.recovery_score;
                    document.getElementById('hrv-score').textContent = data.metrics.hrv_rmssd_mean;
                    document.getElementById('sleep-score').textContent = data.metrics.sleep_total_sleep_hours;
                    
                    // Update insights
                    document.getElementById('daily-insights').innerHTML = `<p>${data.daily_insight}</p>`;
                    
                    // Update weekly trends
                    let trendsHtml = '<ul>';
                    data.weekly_trends.forEach(trend => {
                        trendsHtml += `<li>${trend}</li>`;
                    });
                    trendsHtml += '</ul>';
                    document.getElementById('weekly-trends').innerHTML = trendsHtml;
                    
                    // Update data visualization placeholder
                    document.getElementById('data-visualization').innerHTML = '<p>Data visualization would be displayed here.</p>';
                })
                .catch(error => {
                    console.error('Error:', error);
                });
            }
        });
    </script>
</body>
</html>
        """)

# Create CSS file if it doesn't exist
if not os.path.exists("static/styles.css"):
    with open("static/styles.css", "w") as f:
        f.write("""
body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background-color: #f8f9fa;
    padding-top: 20px;
}
.chat-container {
    height: 400px;
    overflow-y: auto;
    border: 1px solid #dee2e6;
    border-radius: 5px;
    padding: 15px;
    background-color: white;
}
.user-message {
    background-color: #e9ecef;
    padding: 10px 15px;
    border-radius: 15px;
    margin-bottom: 10px;
    max-width: 80%;
    margin-left: auto;
    text-align: right;
}
.agent-message {
    background-color: #007bff;
    color: white;
    padding: 10px 15px;
    border-radius: 15px;
    margin-bottom: 10px;
    max-width: 80%;
}
.dashboard-card {
    background-color: white;
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    padding: 20px;
    margin-bottom: 20px;
}
.metric-value {
    font-size: 2rem;
    font-weight: bold;
}
.metric-label {
    color: #6c757d;
    font-size: 0.9rem;
}
.insights-container {
    background-color: white;
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    padding: 20px;
    margin-bottom: 20px;
}
        """)

# Initialize components
data_loader = DataLoader()
feature_engineer = FeatureEngineer()
prompt_builder = InsightPromptBuilder()
llm_engine = LLMEngine()

# Load sample data
try:
    # Check if processed data exists
    processed_dir = 'data/processed'
    if os.path.exists(os.path.join(processed_dir, 'combined_features.json')):
        with open(os.path.join(processed_dir, 'combined_features.json'), 'r') as f:
            combined_features = json.load(f)
        
        # Get the latest date
        dates = list(combined_features.keys())
        dates.sort()
        latest_date = dates[-1]
        latest_features = combined_features[latest_date]
    else:
        # Use sample data
        latest_features = {
            "recovery_score": 85,
            "hrv_rmssd_mean": 65,
            "hrv_sdnn": 78,
            "sleep_total_sleep_hours": 7.5,
            "sleep_deep_sleep_percentage": 22,
            "sleep_rem_sleep_percentage": 18,
            "activity_total_steps": 8500,
            "activity_active_minutes": 45,
            "stress_score": 35
        }
        combined_features = {
            datetime.now().strftime("%Y-%m-%d"): latest_features
        }
except Exception as e:
    print(f"Error loading data: {e}")
    # Use default data
    latest_features = {
        "recovery_score": 85,
        "hrv_rmssd_mean": 65,
        "hrv_sdnn": 78,
        "sleep_total_sleep_hours": 7.5,
        "sleep_deep_sleep_percentage": 22,
        "sleep_rem_sleep_percentage": 18,
        "activity_total_steps": 8500,
        "activity_active_minutes": 45,
        "stress_score": 35
    }
    combined_features = {
        datetime.now().strftime("%Y-%m-%d"): latest_features
    }

# Load user goals
try:
    if os.path.exists(os.path.join(processed_dir, 'user_goals.json')):
        with open(os.path.join(processed_dir, 'user_goals.json'), 'r') as f:
            user_goals = json.load(f)
    else:
        user_goals = {
            "name": "Alex",
            "primary_goals": [
                {"area": "fitness", "goal": "Complete a half marathon in under 2 hours"},
                {"area": "sleep", "goal": "Improve sleep quality and consistency"},
                {"area": "stress", "goal": "Reduce daily stress levels"}
            ]
        }
except Exception as e:
    print(f"Error loading user goals: {e}")
    user_goals = {
        "name": "Alex",
        "primary_goals": [
            {"area": "fitness", "goal": "Complete a half marathon in under 2 hours"},
            {"area": "sleep", "goal": "Improve sleep quality and consistency"},
            {"area": "stress", "goal": "Reduce daily stress levels"}
        ]
    }

# Initialize agent
agent = AgentSimulator(llm_engine, latest_features)

# Set user goals
user_goals_dict = {}
for goal in user_goals['primary_goals']:
    user_goals_dict[goal['area']] = goal['goal']
    
agent.set_user_goals(user_goals_dict)

# Set agent persona
agent.set_agent_persona({
    "name": "FitCoach",
    "role": "Wearable Data Coach",
    "tone": "coach",
    "expertise": ["running", "recovery", "sleep optimization", "stress management"]
})

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

# Routes
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/chat", response_model=ChatResponse)
async def chat(message: ChatMessage):
    try:
        # Check if API key is set
        api_key_set = os.environ.get("OPENAI_API_KEY") is not None
        
        if api_key_set:
            # Process message with LLM
            response, _ = agent.process_message(message.message)
        else:
            # Simulate response if no API key
            if "recovery" in message.message.lower():
                response = "Your recovery score is 85, which is good. Based on this, you could handle a moderate intensity workout today. Make sure to monitor how you feel during the session."
            elif "sleep" in message.message.lower():
                response = "You slept 7.5 hours with 22% deep sleep, which is within the healthy range. To improve further, try to maintain a consistent sleep schedule and limit screen time before bed."
            elif "train" in message.message.lower() or "workout" in message.message.lower():
                response = "Based on your recovery score and recent training load, today would be good for a moderate intensity workout. Focus on maintaining good form and listen to your body."
            elif "stress" in message.message.lower():
                response = "Your stress score is relatively low at 35. Your HRV data suggests good autonomic balance. Continue with your current stress management practices."
            else:
                response = "I'm here to help you understand your wearable data and provide personalized recommendations. What specific aspect of your health or fitness would you like to discuss?"
        
        return ChatResponse(
            response=response,
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/dashboard", response_model=DashboardData)
async def get_dashboard_data():
    try:
        # Generate daily insight
        daily_insight_prompt = prompt_builder.build_prompt(
            latest_features,
            tone="coach",
            user_goal="general health"
        )
        
        # Check if API key is set
        api_key_set = os.environ.get("OPENAI_API_KEY") is not None
        
        if api_key_set:
            # Generate insight with LLM
            daily_insight, _ = llm_engine.generate_insight(daily_insight_prompt)
        else:
            # Use simulated insight
            if latest_features["recovery_score"] > 80:
                daily_insight = "Your recovery is excellent today. This is a good opportunity for a challenging workout if it aligns with your training plan."
            elif latest_features["recovery_score"] > 60:
                daily_insight = "Your recovery is moderate today. Consider a maintenance workout or focus on technique rather than intensity."
            else:
                daily_insight = "Your recovery is low today. Consider active recovery activities like walking, yoga, or light mobility work."
        
        # Generate weekly trends
        weekly_trends = []
        
        # Check if we have enough historical data
        if len(combined_features) > 1:
            # Sort dates
            dates = list(combined_features.keys())
            dates.sort()
            
            # Compare first and last day
            first_day = combined_features[dates[0]]
            last_day = combined_features[dates[-1]]
            
            # Recovery trend
            recovery_change = last_day.get("recovery_score", 0) - first_day.get("recovery_score", 0)
            if recovery_change > 5:
                weekly_trends.append("Your recovery score has improved over the past week.")
            elif recovery_change < -5:
                weekly_trends.append("Your recovery score has declined over the past week.")
            else:
                weekly_trends.append("Your recovery score has remained stable over the past week.")
            
            # Sleep trend
            sleep_change = last_day.get("sleep_total_sleep_hours", 0) - first_day.get("sleep_total_sleep_hours", 0)
            if sleep_change > 0.5:
                weekly_trends.append("Your sleep duration has increased over the past week.")
            elif sleep_change < -0.5:
                weekly_trends.append("Your sleep duration has decreased over the past week.")
            else:
                weekly_trends.append("Your sleep duration has remained consistent over the past week.")
            
            # HRV trend
            hrv_change = last_day.get("hrv_rmssd_mean", 0) - first_day.get("hrv_rmssd_mean", 0)
            if hrv_change > 5:
                weekly_trends.append("Your HRV has improved, indicating better recovery capacity.")
            elif hrv_change < -5:
                weekly_trends.append("Your HRV has decreased, which may indicate accumulated fatigue.")
            else:
                weekly_trends.append("Your HRV has remained stable, indicating consistent recovery capacity.")
        else:
            # Default trends if not enough data
            weekly_trends = [
                "Insufficient historical data to generate trends.",
                "Continue tracking your metrics to see patterns emerge.",
                "Focus on consistency in your sleep and recovery routines."
            ]
        
        return DashboardData(
            metrics=latest_features,
            daily_insight=daily_insight,
            weekly_trends=weekly_trends
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the application
if __name__ == "__main__":
    # Check if API key is set
    api_key_set = os.environ.get("OPENAI_API_KEY") is not None
    if not api_key_set:
        print("WARNING: OpenAI API key not set. Responses will be simulated.")
        print("To set the API key, run: export OPENAI_API_KEY='your-api-key'")
    
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
