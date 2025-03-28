"""
Streamlit Demo App for Wearable Data Insight Generator

This interactive demo showcases the capabilities of the Wearable Data Insight Generator,
allowing users to upload data, visualize insights, and interact with the coaching agent.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
from PIL import Image

# Add parent directory to path for imports
sys.path.append("..")
from src.data_utils import generate_synthetic_data, normalize_data, detect_anomalies, extract_trends

# Set page config
st.set_page_config(
    page_title="Wearable Data Insight Generator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-bottom: 1rem;
    }
    .insight-card {
        background-color: #f5f5f5;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        border-left: 5px solid #1E88E5;
    }
    .metric-card {
        background-color: #e3f2fd;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 0.5rem;
    }
    .system-flow {
        background-color: #fafafa;
        border-radius: 10px;
        padding: 1rem;
        margin-top: 1rem;
        border: 1px solid #e0e0e0;
    }
    .footer {
        margin-top: 3rem;
        text-align: center;
        color: #9e9e9e;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'user_id' not in st.session_state:
    st.session_state.user_id = f"demo_user_{datetime.now().strftime('%Y%m%d%H%M%S')}"
if 'data' not in st.session_state:
    st.session_state.data = None
if 'insights' not in st.session_state:
    st.session_state.insights = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'api_url' not in st.session_state:
    # Default to local API if running locally
    st.session_state.api_url = "http://localhost:8000"

# Helper functions
def call_api(endpoint, data=None, method="GET"):
    """Call the API with the given endpoint and data"""
    url = f"{st.session_state.api_url}/{endpoint}"
    try:
        if method == "GET":
            response = requests.get(url, params=data)
        else:
            response = requests.post(url, json=data)
        return response.json()
    except Exception as e:
        st.error(f"Error calling API: {str(e)}")
        return None

def generate_insights(data=None):
    """Generate insights from the API"""
    request_data = {
        "user_id": st.session_state.user_id,
        "persona": st.session_state.persona,
        "goals": st.session_state.goals,
        "insight_categories": ["recovery", "training", "sleep"]
    }
    
    if data is not None:
        request_data["data"] = data
    
    response = call_api("api/generate-insight", request_data, method="POST")
    if response and "insights" in response:
        st.session_state.insights = response["insights"]
        return response["insights"]
    return []

def format_timestamp(timestamp):
    """Format timestamp for display"""
    if isinstance(timestamp, str):
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            return dt.strftime("%b %d, %Y")
        except:
            return timestamp
    return timestamp

def create_metric_chart(df, metric, title, color):
    """Create a line chart for a metric"""
    fig = px.line(
        df, 
        x='timestamp', 
        y=metric, 
        title=title,
        markers=True
    )
    fig.update_traces(line=dict(color=color, width=3))
    fig.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis_title="",
        yaxis_title="",
        hovermode="x unified"
    )
    return fig

def create_radar_chart(data):
    """Create a radar chart for multiple metrics"""
    categories = list(data.keys())
    values = list(data.values())
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Current Status',
        line=dict(color='#1E88E5', width=3),
        fillcolor='rgba(30, 136, 229, 0.3)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        ),
        showlegend=False,
        height=400,
        margin=dict(l=10, r=10, t=10, b=10)
    )
    
    return fig

# Sidebar
with st.sidebar:
    st.image("https://raw.githubusercontent.com/teekag/Wearable-LLM-Insight-Generator/main/diagrams/logo.png", width=100)
    st.markdown("## User Configuration")
    
    # User persona selection
    st.session_state.persona = st.selectbox(
        "Select User Persona",
        ["athlete", "casual", "stressed"],
        index=0,
        help="Choose a persona that best matches your profile"
    )
    
    # User goals
    goals_options = [
        "Improve endurance",
        "Increase strength",
        "Better recovery",
        "Weight management",
        "Stress reduction",
        "Sleep improvement",
        "General health"
    ]
    st.session_state.goals = st.multiselect(
        "Select Your Goals",
        goals_options,
        default=["Improve endurance", "Better recovery"],
        help="Choose one or more goals"
    )
    
    # Data source
    data_source = st.radio(
        "Data Source",
        ["Generate Synthetic Data", "Upload Your Own Data"],
        index=0,
        help="Choose where to get the data from"
    )
    
    if data_source == "Generate Synthetic Data":
        days = st.slider("Days of Data", min_value=7, max_value=90, value=30, step=1)
        if st.button("Generate Data"):
            with st.spinner("Generating synthetic data..."):
                df = generate_synthetic_data(days=days, user_type=st.session_state.persona)
                st.session_state.data = df
                st.success(f"Generated {len(df)} days of synthetic data!")
    else:
        uploaded_file = st.file_uploader("Upload CSV or JSON", type=["csv", "json"])
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.DataFrame(json.load(uploaded_file))
                
                # Normalize the data
                df = normalize_data(df)
                st.session_state.data = df
                st.success(f"Uploaded and processed {len(df)} records!")
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    
    # API connection
    st.markdown("---")
    st.markdown("## API Connection")
    api_url = st.text_input("API URL", value=st.session_state.api_url)
    if api_url != st.session_state.api_url:
        st.session_state.api_url = api_url
    
    # About
    st.markdown("---")
    st.markdown("## About")
    st.markdown("""
    This demo showcases the Wearable Data Insight Generator, a system that transforms raw wearable device data into personalized health and fitness insights using Large Language Models.
    
    [View on GitHub](https://github.com/teekag/Wearable-LLM-Insight-Generator)
    """)

# Main content
st.markdown('<h1 class="main-header">Wearable Data Insight Generator</h1>', unsafe_allow_html=True)
st.markdown("Transform raw wearable data into personalized, actionable insights using AI")

# Tabs for different views
tab1, tab2, tab3, tab4 = st.tabs(["Dashboard", "Insight Generation", "System Flow", "Agent Chat"])

with tab1:
    st.markdown('<h2 class="sub-header">Health & Fitness Dashboard</h2>', unsafe_allow_html=True)
    
    if st.session_state.data is not None:
        df = st.session_state.data
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            latest_hrv = df['hrv'].iloc[-1] if not df['hrv'].isna().all() else 0
            st.metric(
                "Latest HRV", 
                f"{latest_hrv:.1f}",
                delta=f"{latest_hrv - df['hrv'].iloc[-7] if len(df) > 7 and not df['hrv'].isna().all() else 0:.1f}"
            )
        
        with col2:
            latest_sleep = df['sleep_quality'].iloc[-1] if not df['sleep_quality'].isna().all() else 0
            st.metric(
                "Sleep Quality", 
                f"{latest_sleep:.1f}%",
                delta=f"{latest_sleep - df['sleep_quality'].iloc[-7] if len(df) > 7 and not df['sleep_quality'].isna().all() else 0:.1f}%"
            )
        
        with col3:
            latest_activity = df['activity_level'].iloc[-1] if not df['activity_level'].isna().all() else 0
            st.metric(
                "Activity Level", 
                f"{latest_activity:.1f}%",
                delta=f"{latest_activity - df['activity_level'].iloc[-7] if len(df) > 7 and not df['activity_level'].isna().all() else 0:.1f}%"
            )
        
        with col4:
            latest_recovery = df['subjective_recovery'].iloc[-1] if not df['subjective_recovery'].isna().all() else 0
            st.metric(
                "Recovery Score", 
                f"{latest_recovery:.1f}%",
                delta=f"{latest_recovery - df['subjective_recovery'].iloc[-7] if len(df) > 7 and not df['subjective_recovery'].isna().all() else 0:.1f}%"
            )
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(
                create_metric_chart(df, 'hrv', 'Heart Rate Variability (HRV)', '#1E88E5'),
                use_container_width=True
            )
            st.plotly_chart(
                create_metric_chart(df, 'sleep_quality', 'Sleep Quality (%)', '#43A047'),
                use_container_width=True
            )
        
        with col2:
            st.plotly_chart(
                create_metric_chart(df, 'activity_level', 'Activity Level (%)', '#FB8C00'),
                use_container_width=True
            )
            st.plotly_chart(
                create_metric_chart(df, 'subjective_recovery', 'Recovery Score (%)', '#8E24AA'),
                use_container_width=True
            )
        
        # Anomaly detection
        st.markdown("### Anomaly Detection")
        anomalies = detect_anomalies(df)
        
        has_anomalies = False
        for metric, anomaly_list in anomalies.items():
            if anomaly_list:
                has_anomalies = True
                break
        
        if has_anomalies:
            for metric, anomaly_list in anomalies.items():
                if anomaly_list:
                    st.markdown(f"**{metric.replace('_', ' ').title()}**")
                    for anomaly in anomaly_list[-3:]:  # Show only the 3 most recent anomalies
                        st.markdown(
                            f"<div class='metric-card'>"
                            f"Date: {format_timestamp(anomaly['timestamp'])}<br>"
                            f"Value: {anomaly['value']:.1f} ({anomaly['direction'].upper()}) - "
                            f"Baseline: {anomaly['baseline']:.1f}<br>"
                            f"Deviation: {anomaly['z_score']:.1f} standard deviations"
                            f"</div>",
                            unsafe_allow_html=True
                        )
        else:
            st.info("No significant anomalies detected in your data.")
        
        # Trends
        st.markdown("### Trends Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            hrv_trend = extract_trends(df, 'hrv')
            sleep_trend = extract_trends(df, 'sleep_quality')
            
            st.markdown(
                f"<div class='metric-card'>"
                f"<strong>HRV Trend:</strong> {hrv_trend['trend'].title()} "
                f"({hrv_trend['change_pct']}% over period)<br>"
                f"<strong>Sleep Quality Trend:</strong> {sleep_trend['trend'].title()} "
                f"({sleep_trend['change_pct']}% over period)"
                f"</div>",
                unsafe_allow_html=True
            )
        
        with col2:
            activity_trend = extract_trends(df, 'activity_level')
            recovery_trend = extract_trends(df, 'subjective_recovery')
            
            st.markdown(
                f"<div class='metric-card'>"
                f"<strong>Activity Trend:</strong> {activity_trend['trend'].title()} "
                f"({activity_trend['change_pct']}% over period)<br>"
                f"<strong>Recovery Trend:</strong> {recovery_trend['trend'].title()} "
                f"({recovery_trend['change_pct']}% over period)"
                f"</div>",
                unsafe_allow_html=True
            )
        
        # Current status radar chart
        st.markdown("### Current Status Overview")
        
        # Get latest values
        latest_metrics = {
            "HRV": min(100, max(0, df['hrv'].iloc[-1] if not df['hrv'].isna().all() else 0)),
            "Sleep": df['sleep_quality'].iloc[-1] if not df['sleep_quality'].isna().all() else 0,
            "Activity": df['activity_level'].iloc[-1] if not df['activity_level'].isna().all() else 0,
            "Recovery": df['subjective_recovery'].iloc[-1] if not df['subjective_recovery'].isna().all() else 0,
            "Balance": (df['sleep_quality'].iloc[-1] + df['subjective_recovery'].iloc[-1]) / 2 if not df['sleep_quality'].isna().all() and not df['subjective_recovery'].isna().all() else 0
        }
        
        st.plotly_chart(create_radar_chart(latest_metrics), use_container_width=True)
    else:
        st.info("No data available. Please generate synthetic data or upload your own data from the sidebar.")
        
        # Sample dashboard image
        st.image("https://raw.githubusercontent.com/teekag/Wearable-LLM-Insight-Generator/main/diagrams/dashboard_preview.png", caption="Sample Dashboard Preview")

with tab2:
    st.markdown('<h2 class="sub-header">Insight Generation</h2>', unsafe_allow_html=True)
    
    if st.session_state.data is not None:
        if st.button("Generate Insights"):
            with st.spinner("Generating insights from your data..."):
                # Convert DataFrame to records for API
                data_records = st.session_state.data.to_dict(orient="records")
                insights = generate_insights(data_records)
                st.success(f"Generated {len(insights)} insights!")
        
        if st.session_state.insights:
            for insight in st.session_state.insights:
                # Format timestamp
                timestamp = format_timestamp(insight.get("timestamp", ""))
                
                # Get insight type
                insight_type = insight.get("insight_type", {})
                category = insight_type.get("category", "").title()
                
                # Get metrics
                metrics = insight.get("metrics", {})
                metrics_str = ", ".join([f"{k}: {v:.1f}" for k, v in metrics.items() if v is not None])
                
                # Get recommendations
                recommendations = insight.get("recommendations", [])
                
                # Display insight card
                st.markdown(
                    f"<div class='insight-card'>"
                    f"<strong>{category} Insight</strong> - {timestamp}<br>"
                    f"<h3>{insight.get('summary', '')}</h3>"
                    f"<p>{insight.get('detail', '')}</p>"
                    f"<p><strong>Metrics:</strong> {metrics_str}</p>"
                    f"<p><strong>Recommendations:</strong></p>"
                    f"<ul>{''.join([f'<li>{rec}</li>' for rec in recommendations])}</ul>"
                    f"<p><em>Confidence: {insight.get('confidence', 0) * 100:.0f}%</em></p>"
                    f"</div>",
                    unsafe_allow_html=True
                )
        else:
            st.info("No insights generated yet. Click the 'Generate Insights' button to analyze your data.")
            
            # Sample insight card
            st.markdown(
                f"<div class='insight-card'>"
                f"<strong>Recovery Insight</strong> - Sample<br>"
                f"<h3>Your recovery is trending positively</h3>"
                f"<p>Your HRV has increased by 15% over the past week, indicating improved recovery capacity. Sleep quality has also been consistent.</p>"
                f"<p><strong>Metrics:</strong> hrv: 65.0, sleep_quality: 85.0, recovery: 78.0</p>"
                f"<p><strong>Recommendations:</strong></p>"
                f"<ul>"
                f"<li>Consider increasing training intensity by 10%</li>"
                f"<li>Maintain current sleep schedule</li>"
                f"<li>Monitor HRV response to increased load</li>"
                f"</ul>"
                f"<p><em>Confidence: 85%</em></p>"
                f"</div>",
                unsafe_allow_html=True
            )
    else:
        st.info("No data available. Please generate synthetic data or upload your own data from the sidebar.")

with tab3:
    st.markdown('<h2 class="sub-header">System Architecture & Flow</h2>', unsafe_allow_html=True)
    
    # System flow diagram
    st.markdown(
        """
        The Wearable Data Insight Generator processes raw wearable device data through a sophisticated pipeline
        to generate personalized insights. Here's how the system works:
        """
    )
    
    # Animated system flow
    st.markdown(
        """
        <div class="system-flow">
            <div style="text-align: center;">
                <h3>Data Processing Pipeline</h3>
                <img src="https://raw.githubusercontent.com/teekag/Wearable-LLM-Insight-Generator/main/diagrams/system_flow.gif" width="100%">
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # System components
    st.markdown("### Key Components")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(
            """
            **1. Data Ingestion & Normalization**
            - Supports multiple wearable data sources
            - Normalizes data to standard schema
            - Handles missing values and outliers
            
            **2. Feature Engineering**
            - Extracts temporal patterns
            - Calculates derived metrics
            - Identifies anomalies and trends
            """
        )
    
    with col2:
        st.markdown(
            """
            **3. Prompt Engineering & LLM Integration**
            - Constructs context-rich prompts
            - Integrates with OpenAI/Mistral/Gemini
            - Ensures consistent output format
            
            **4. Insight Generation & Personalization**
            - Tailors insights to user persona
            - Provides actionable recommendations
            - Adapts to user goals and preferences
            """
        )
    
    # Technical architecture
    st.markdown("### Technical Architecture")
    st.image("https://raw.githubusercontent.com/teekag/Wearable-LLM-Insight-Generator/main/diagrams/technical_architecture.png", caption="System Architecture Diagram")
    
    # LLM comparison
    st.markdown("### LLM Model Comparison")
    
    model_comparison = pd.DataFrame({
        "Model": ["GPT-4", "GPT-3.5-Turbo", "Mistral 7B", "Llama 2 7B"],
        "Accuracy": [92, 85, 78, 72],
        "Latency (ms)": [450, 120, 80, 90],
        "Cost": ["High", "Medium", "Low", "Low"],
        "Hallucination Risk": ["Low", "Medium", "Medium-High", "High"]
    })
    
    st.dataframe(model_comparison, use_container_width=True)

with tab4:
    st.markdown('<h2 class="sub-header">Coaching Agent</h2>', unsafe_allow_html=True)
    
    st.markdown(
        """
        Interact with the AI coaching agent to get personalized advice based on your wearable data.
        Ask questions about your health, fitness, recovery, or training recommendations.
        """
    )
    
    # Display chat history
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(f"<div style='background-color: #e3f2fd; padding: 10px; border-radius: 10px; margin-bottom: 10px;'><strong>You:</strong> {message['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='background-color: #f5f5f5; padding: 10px; border-radius: 10px; margin-bottom: 10px;'><strong>Coach:</strong> {message['content']}</div>", unsafe_allow_html=True)
    
    # Chat input
    user_input = st.text_input("Ask your coach a question:", key="user_message")
    
    if st.button("Send") and user_input:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # Call chat API
        if st.session_state.data is not None:
            with st.spinner("Coach is thinking..."):
                response = call_api(
                    "api/chat-agent",
                    {
                        "user_id": st.session_state.user_id,
                        "message": user_input,
                        "context": {
                            "persona": st.session_state.persona,
                            "goals": st.session_state.goals
                        }
                    },
                    method="POST"
                )
                
                if response:
                    # Add coach response to chat history
                    st.session_state.chat_history.append({"role": "assistant", "content": response.get("response", "I'm sorry, I couldn't process your request.")})
                    
                    # Display suggestions if available
                    if "suggestions" in response and response["suggestions"]:
                        suggestion_buttons = st.columns(len(response["suggestions"]))
                        for i, suggestion in enumerate(response["suggestions"]):
                            with suggestion_buttons[i]:
                                st.button(suggestion, key=f"suggestion_{i}", on_click=lambda s=suggestion: st.session_state.update({"user_message": s}))
                else:
                    st.session_state.chat_history.append({"role": "assistant", "content": "I'm sorry, I couldn't connect to the coaching service."})
        else:
            st.session_state.chat_history.append({"role": "assistant", "content": "I need your wearable data to provide personalized coaching. Please generate or upload data from the sidebar first."})
        
        # Rerun to update the chat display
        st.experimental_rerun()
    
    # Sample questions
    st.markdown("### Sample Questions")
    sample_questions = [
        "How am I doing overall?",
        "What does my sleep data show?",
        "Should I train hard today?",
        "What's causing my low HRV?",
        "How can I improve my recovery?"
    ]
    
    sample_cols = st.columns(len(sample_questions))
    for i, question in enumerate(sample_questions):
        with sample_cols[i]:
            st.button(question, key=f"sample_{i}", on_click=lambda q=question: st.session_state.update({"user_message": q}))

# Footer
st.markdown(
    """
    <div class="footer">
        <p>Wearable Data Insight Generator | Created by Tejas Agnihotri | <a href="https://github.com/teekag/Wearable-LLM-Insight-Generator">GitHub Repository</a></p>
    </div>
    """,
    unsafe_allow_html=True
)

# Run the Streamlit app
if __name__ == "__main__":
    # This is handled by the Streamlit CLI
    pass
