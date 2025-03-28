"""
Unit tests for the Insight Engine module.

This module contains tests for the InsightEngine class and its methods.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.insight_engine import InsightEngine, UserConfig, InsightType, Insight
from src.data_utils import generate_synthetic_data

class TestInsightEngine(unittest.TestCase):
    """Test cases for the InsightEngine class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock components
        self.mock_feature_engineer = MagicMock()
        self.mock_prompt_builder = MagicMock()
        self.mock_llm_engine = MagicMock()
        
        # Create test data
        self.test_data = generate_synthetic_data(days=7, user_type="athlete")
        
        # Configure mocks
        self.mock_feature_engineer.extract_features.return_value = {
            "hrv": 65.0,
            "sleep_quality": 85.0,
            "activity_level": 70.0,
            "subjective_recovery": 80.0
        }
        
        self.mock_prompt_builder.build_prompt.return_value = "Test prompt"
        
        self.mock_llm_engine.generate_insight.return_value = """Your recovery is trending positively.
Your HRV has increased by 15% over the past week, indicating improved recovery capacity.
- Consider increasing training intensity by 10%
- Maintain current sleep schedule
- Monitor HRV response to increased load"""
        
        # Create insight engine with mocks
        self.insight_engine = InsightEngine(
            feature_engineer=self.mock_feature_engineer,
            prompt_builder=self.mock_prompt_builder,
            llm_engine=self.mock_llm_engine,
            memory_store={}
        )
    
    def test_process_data_dataframe(self):
        """Test processing data from a DataFrame."""
        # Call the method
        result = self.insight_engine.process_data(self.test_data, data_format="dataframe")
        
        # Assert result is a DataFrame with expected columns
        self.assertIsInstance(result, pd.DataFrame)
        expected_columns = ['timestamp', 'hrv', 'sleep_quality', 'activity_level', 'subjective_recovery', 'notes']
        for col in expected_columns:
            self.assertIn(col, result.columns)
    
    def test_process_data_dict(self):
        """Test processing data from a dictionary."""
        # Convert test data to dict
        data_dict = self.test_data.to_dict(orient="records")
        
        # Call the method
        result = self.insight_engine.process_data(data_dict, data_format="dict")
        
        # Assert result is a DataFrame with expected columns
        self.assertIsInstance(result, pd.DataFrame)
        expected_columns = ['timestamp', 'hrv', 'sleep_quality', 'activity_level', 'subjective_recovery', 'notes']
        for col in expected_columns:
            self.assertIn(col, result.columns)
    
    def test_extract_features(self):
        """Test feature extraction."""
        # Call the method
        result = self.insight_engine.extract_features(self.test_data)
        
        # Assert feature engineer was called
        self.mock_feature_engineer.extract_features.assert_called_once_with(self.test_data)
        
        # Assert result matches mock return value
        self.assertEqual(result, self.mock_feature_engineer.extract_features.return_value)
    
    def test_generate_insight(self):
        """Test insight generation."""
        # Set up test parameters
        user_config = UserConfig(
            persona="athlete",
            goals=["Improve endurance", "Better recovery"],
            preferences={"user_id": "test_user"}
        )
        
        insight_types = [InsightType(category="recovery", priority=1)]
        
        # Call the method
        results = self.insight_engine.generate_insight(
            data=self.test_data,
            user_config=user_config,
            insight_types=insight_types,
            data_format="dataframe"
        )
        
        # Assert results
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 1)
        self.assertIsInstance(results[0], Insight)
        self.assertEqual(results[0].insight_type.category, "recovery")
        self.assertEqual(results[0].summary, "Your recovery is trending positively.")
        
        # Assert mocks were called correctly
        self.mock_feature_engineer.extract_features.assert_called_once()
        self.mock_prompt_builder.build_prompt.assert_called_once()
        self.mock_llm_engine.generate_insight.assert_called_once_with("Test prompt")
    
    def test_parse_llm_response(self):
        """Test parsing LLM response."""
        # Set up test parameters
        llm_response = """Your recovery is trending positively.
Your HRV has increased by 15% over the past week, indicating improved recovery capacity.
- Consider increasing training intensity by 10%
- Maintain current sleep schedule
- Monitor HRV response to increased load"""
        
        insight_type = InsightType(category="recovery", priority=1)
        
        features = {
            "hrv": 65.0,
            "sleep_quality": 85.0,
            "activity_level": 70.0,
            "subjective_recovery": 80.0
        }
        
        # Call the method
        result = self.insight_engine._parse_llm_response(llm_response, insight_type, features)
        
        # Assert result
        self.assertIsInstance(result, Insight)
        self.assertEqual(result.insight_type.category, "recovery")
        self.assertEqual(result.summary, "Your recovery is trending positively.")
        self.assertIn("HRV has increased", result.detail)
        self.assertEqual(len(result.recommendations), 3)
        self.assertIn("Consider increasing training intensity by 10%", result.recommendations)
    
    def test_update_memory(self):
        """Test memory update."""
        # Set up test parameters
        user_id = "test_user"
        insight = Insight(
            insight_type=InsightType(category="recovery", priority=1),
            summary="Test summary",
            detail="Test detail",
            metrics={"hrv": 65.0},
            recommendations=["Test recommendation"]
        )
        
        # Call the method
        self.insight_engine._update_memory(user_id, insight)
        
        # Assert memory was updated
        self.assertIn(user_id, self.insight_engine.memory_store)
        self.assertIn("insights", self.insight_engine.memory_store[user_id])
        self.assertEqual(len(self.insight_engine.memory_store[user_id]["insights"]), 1)
        self.assertIn("context", self.insight_engine.memory_store[user_id])
        self.assertIn("hrv", self.insight_engine.memory_store[user_id]["context"])
    
    def test_get_user_history(self):
        """Test retrieving user history."""
        # Set up test data
        user_id = "test_user"
        insight = Insight(
            insight_type=InsightType(category="recovery", priority=1),
            summary="Test summary",
            detail="Test detail",
            metrics={"hrv": 65.0},
            recommendations=["Test recommendation"]
        )
        
        # Update memory
        self.insight_engine._update_memory(user_id, insight)
        
        # Call the method
        result = self.insight_engine.get_user_history(user_id)
        
        # Assert result
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["summary"], "Test summary")
        
        # Test non-existent user
        result = self.insight_engine.get_user_history("non_existent_user")
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
