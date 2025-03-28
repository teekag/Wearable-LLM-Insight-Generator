"""
Insight Engine Module for Wearable Data Insight Generator

This module serves as the core engine for generating insights from wearable device data.
It combines feature engineering, prompt building, and LLM integration to provide
personalized health and fitness insights based on biometric data.
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Union, Any

import pandas as pd
from pydantic import BaseModel, Field

from src.feature_engineer import FeatureEngineer
from src.insight_prompt_builder import InsightPromptBuilder
from src.llm_engine import LLMEngine
from src.data_utils import normalize_data

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UserConfig(BaseModel):
    """User configuration for personalized insights"""
    persona: str = Field(default="athlete", description="User persona (athlete, casual, health-focused)")
    goals: List[str] = Field(default_factory=list, description="User fitness/health goals")
    preferences: Dict[str, Any] = Field(default_factory=dict, description="User preferences for insights")
    sensitivity: Dict[str, float] = Field(default_factory=dict, description="Sensitivity thresholds for metrics")

class InsightType(BaseModel):
    """Type of insight to be generated"""
    category: str = Field(..., description="Category of insight (recovery, training, sleep, etc.)")
    subcategory: str = Field(default="", description="Subcategory for more specific insights")
    priority: int = Field(default=1, description="Priority level (1-5)")

class Insight(BaseModel):
    """Structured insight generated from wearable data"""
    timestamp: datetime = Field(default_factory=datetime.now)
    insight_type: InsightType
    summary: str = Field(..., description="Short summary of the insight")
    detail: str = Field(..., description="Detailed explanation of the insight")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Relevant metrics that led to this insight")
    confidence: float = Field(default=0.0, description="Confidence score (0-1)")
    recommendations: List[str] = Field(default_factory=list, description="Actionable recommendations")
    sources: List[str] = Field(default_factory=list, description="Data sources used for this insight")

class InsightEngine:
    """
    Core engine for generating insights from wearable data.
    
    This class orchestrates the process of:
    1. Processing raw wearable data
    2. Extracting relevant features
    3. Building prompts for the LLM
    4. Generating structured insights
    5. Applying user preferences and feedback
    """
    
    def __init__(
        self, 
        feature_engineer: Optional[FeatureEngineer] = None,
        prompt_builder: Optional[InsightPromptBuilder] = None,
        llm_engine: Optional[LLMEngine] = None,
        memory_store: Optional[Dict] = None
    ):
        """
        Initialize the Insight Engine with its component modules.
        
        Args:
            feature_engineer: Feature engineering module
            prompt_builder: Prompt building module
            llm_engine: LLM integration module
            memory_store: Simple memory store for user feedback and history
        """
        self.feature_engineer = feature_engineer or FeatureEngineer()
        self.prompt_builder = prompt_builder or InsightPromptBuilder()
        self.llm_engine = llm_engine or LLMEngine()
        self.memory_store = memory_store or {}
        self.logger = logging.getLogger(__name__)
    
    def process_data(self, data: Union[pd.DataFrame, Dict, str], data_format: str = "dataframe") -> pd.DataFrame:
        """
        Process and normalize input data from various sources.
        
        Args:
            data: Input data as DataFrame, Dict, or file path
            data_format: Format of the input data ("dataframe", "dict", "json", "csv")
            
        Returns:
            Normalized DataFrame with standardized columns
        """
        try:
            if data_format == "dataframe" and isinstance(data, pd.DataFrame):
                df = data
            elif data_format == "dict" and isinstance(data, dict):
                df = pd.DataFrame.from_dict(data)
            elif data_format == "json" and isinstance(data, str):
                # Check if it's a file path or JSON string
                if data.endswith('.json'):
                    with open(data, 'r') as f:
                        json_data = json.load(f)
                else:
                    json_data = json.loads(data)
                df = pd.DataFrame.from_dict(json_data)
            elif data_format == "csv" and isinstance(data, str):
                df = pd.read_csv(data)
            else:
                raise ValueError(f"Unsupported data format: {data_format} or data type mismatch")
            
            # Normalize data to standard schema
            return normalize_data(df)
            
        except Exception as e:
            self.logger.error(f"Error processing data: {str(e)}")
            raise
    
    def extract_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Extract relevant features from normalized data.
        
        Args:
            df: Normalized DataFrame
            
        Returns:
            Dictionary of extracted features
        """
        return self.feature_engineer.extract_features(df)
    
    def generate_insight(
        self, 
        data: Union[pd.DataFrame, Dict, str],
        user_config: Optional[UserConfig] = None,
        insight_types: Optional[List[InsightType]] = None,
        data_format: str = "dataframe"
    ) -> List[Insight]:
        """
        Generate insights from wearable data.
        
        Args:
            data: Input data (DataFrame, Dict, or file path)
            user_config: User configuration for personalization
            insight_types: Types of insights to generate
            data_format: Format of the input data
            
        Returns:
            List of generated insights
        """
        try:
            # Set defaults
            user_config = user_config or UserConfig()
            insight_types = insight_types or [
                InsightType(category="recovery", priority=1),
                InsightType(category="training", priority=2),
                InsightType(category="sleep", priority=3)
            ]
            
            # Process data
            df = self.process_data(data, data_format)
            
            # Extract features
            features = self.extract_features(df)
            
            # Apply user context from memory if available
            user_id = user_config.preferences.get("user_id", "default")
            if user_id in self.memory_store:
                # Merge with historical context
                historical_context = self.memory_store[user_id].get("context", {})
                features["historical_context"] = historical_context
            
            # Generate insights for each requested type
            insights = []
            for insight_type in insight_types:
                # Build prompt for this insight type
                prompt = self.prompt_builder.build_prompt(
                    features=features,
                    insight_type=insight_type.category,
                    user_persona=user_config.persona,
                    user_goals=user_config.goals
                )
                
                # Get LLM response
                llm_response = self.llm_engine.generate_insight(prompt)
                
                # Parse and structure the response
                structured_insight = self._parse_llm_response(llm_response, insight_type, features)
                insights.append(structured_insight)
                
                # Update memory store with new insight
                self._update_memory(user_id, structured_insight)
            
            # Sort insights by priority
            insights.sort(key=lambda x: x.insight_type.priority)
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Error generating insight: {str(e)}")
            raise
    
    def _parse_llm_response(
        self, 
        llm_response: str, 
        insight_type: InsightType,
        features: Dict[str, Any]
    ) -> Insight:
        """
        Parse and structure the LLM response into an Insight object.
        
        Args:
            llm_response: Raw response from the LLM
            insight_type: Type of insight requested
            features: Features used to generate the insight
            
        Returns:
            Structured Insight object
        """
        try:
            # This is a simplified parser - in production, use more robust parsing
            lines = llm_response.strip().split('\n')
            
            # Extract key components (simplified)
            summary = lines[0] if lines else "No insight available"
            detail = '\n'.join(lines[1:3]) if len(lines) > 1 else ""
            
            # Extract recommendations (lines that start with "- ")
            recommendations = [line[2:] for line in lines if line.startswith("- ")]
            
            # Create structured insight
            return Insight(
                insight_type=insight_type,
                summary=summary,
                detail=detail,
                metrics={k: v for k, v in features.items() if k in ["hrv", "sleep_quality", "activity_level"]},
                confidence=0.85,  # Placeholder - would be calculated in production
                recommendations=recommendations,
                sources=["wearable_data", "user_profile", "historical_trends"]
            )
            
        except Exception as e:
            self.logger.error(f"Error parsing LLM response: {str(e)}")
            # Return a fallback insight
            return Insight(
                insight_type=insight_type,
                summary="Unable to generate detailed insight",
                detail="There was an error processing the LLM response.",
                recommendations=["Consult raw data metrics", "Try again later"]
            )
    
    def _update_memory(self, user_id: str, insight: Insight) -> None:
        """
        Update the memory store with new insights and context.
        
        Args:
            user_id: User identifier
            insight: Generated insight
        """
        if user_id not in self.memory_store:
            self.memory_store[user_id] = {"insights": [], "context": {}}
            
        # Add insight to history (limit to last 10)
        self.memory_store[user_id]["insights"].append(insight.dict())
        if len(self.memory_store[user_id]["insights"]) > 10:
            self.memory_store[user_id]["insights"].pop(0)
            
        # Update context with latest metrics
        self.memory_store[user_id]["context"].update(insight.metrics)
        
    def get_user_history(self, user_id: str) -> List[Dict]:
        """
        Retrieve insight history for a specific user.
        
        Args:
            user_id: User identifier
            
        Returns:
            List of historical insights
        """
        if user_id in self.memory_store:
            return self.memory_store[user_id].get("insights", [])
        return []
