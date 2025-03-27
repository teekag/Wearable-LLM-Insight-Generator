"""
Insight Prompt Builder Module for Wearable LLM Insight Generator

This module prepares structured prompts for LLMs based on physiological inputs
and supports different tones (coach-like, medical, motivational).
"""

import json
from typing import Dict, List, Optional, Union, Any
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class InsightPromptBuilder:
    """Class to build prompts for LLM-based insights from wearable data."""
    
    def __init__(self, templates_path: Optional[str] = None):
        """
        Initialize the InsightPromptBuilder.
        
        Args:
            templates_path: Optional path to JSON file with prompt templates
        """
        self.templates = {
            "coach": {
                "system": "You are an expert fitness and wellness coach analyzing biometric data. Provide actionable insights based on the data provided.",
                "user": "Here's my recent biometric data:\n{data_summary}\n\nBased on this data and my goal of {user_goal}, what insights and recommendations can you provide?"
            },
            "medical": {
                "system": "You are a medical professional specializing in physiological data analysis. Provide clinical insights based on the data provided.",
                "user": "Here's my recent physiological data:\n{data_summary}\n\nConsidering my health objective of {user_goal}, what medical insights can you provide? Note any concerning patterns and potential improvements."
            },
            "motivational": {
                "system": "You are a motivational wellness guide focused on encouraging positive health behaviors. Provide supportive insights based on the data provided.",
                "user": "Here's my recent wellness data:\n{data_summary}\n\nMy current goal is {user_goal}. What positive trends do you see? How can I stay motivated and improve further?"
            },
            "scientist": {
                "system": "You are a data scientist specializing in biometric analysis. Provide detailed analytical insights based on the data provided.",
                "user": "Here's my recent biometric data:\n{data_summary}\n\nConsidering my objective of {user_goal}, what statistical patterns and correlations do you observe? What hypotheses can you form about my physiological responses?"
            },
            "sleep_specialist": {
                "system": "You are a sleep specialist analyzing sleep and recovery data. Provide detailed insights on sleep quality and recovery patterns.",
                "user": "Here's my recent sleep and recovery data:\n{data_summary}\n\nConsidering my goal of {user_goal}, what insights can you provide about my sleep patterns and recovery status? What adjustments would you recommend?"
            }
        }
        
        # Load custom templates if provided
        if templates_path:
            try:
                with open(templates_path, 'r') as f:
                    custom_templates = json.load(f)
                    self.templates.update(custom_templates)
                logger.info(f"Loaded custom templates from {templates_path}")
            except Exception as e:
                logger.error(f"Error loading templates from {templates_path}: {str(e)}")
    
    def format_data_summary(self, features: Dict[str, Any], 
                           include_categories: Optional[List[str]] = None,
                           exclude_features: Optional[List[str]] = None) -> str:
        """
        Format feature data into a human-readable summary.
        
        Args:
            features: Dictionary of features
            include_categories: Optional list of feature categories to include (e.g., 'hrv', 'sleep')
            exclude_features: Optional list of specific features to exclude
            
        Returns:
            Formatted string summary of the data
        """
        if exclude_features is None:
            exclude_features = []
            
        # Group features by category
        categorized = {}
        
        for key, value in features.items():
            # Skip excluded features
            if key in exclude_features:
                continue
                
            # Determine category from feature name prefix
            parts = key.split('_', 1)
            category = parts[0] if len(parts) > 1 else 'other'
            
            # Skip if not in included categories
            if include_categories and category not in include_categories:
                continue
                
            if category not in categorized:
                categorized[category] = {}
                
            # Clean up feature name for display
            display_name = key
            if category + '_' in key:
                display_name = key.replace(category + '_', '')
                
            # Format value based on type
            if isinstance(value, float):
                formatted_value = f"{value:.2f}"
            elif isinstance(value, datetime):
                formatted_value = value.strftime("%Y-%m-%d %H:%M")
            else:
                formatted_value = str(value)
                
            categorized[category][display_name] = formatted_value
        
        # Build summary text
        summary = []
        
        for category, features in categorized.items():
            summary.append(f"## {category.upper()} METRICS:")
            for name, value in features.items():
                summary.append(f"- {name}: {value}")
            summary.append("")
        
        return "\n".join(summary)
    
    def build_prompt(self, features: Dict[str, Any], 
                    tone: str = "coach",
                    user_goal: str = "improving overall wellness",
                    include_categories: Optional[List[str]] = None,
                    exclude_features: Optional[List[str]] = None,
                    additional_context: Optional[str] = None,
                    time_range: Optional[str] = "the past week") -> Dict[str, str]:
        """
        Build a complete prompt for LLM insight generation.
        
        Args:
            features: Dictionary of features
            tone: Tone of the prompt (coach, medical, motivational, etc.)
            user_goal: User's stated goal
            include_categories: Optional list of feature categories to include
            exclude_features: Optional list of specific features to exclude
            additional_context: Optional additional context to include
            time_range: Time range of the data
            
        Returns:
            Dictionary with system and user prompts
        """
        # Determine if we're dealing with DiaTrend data
        is_diatrend = any(key.startswith(('glucose_', 'avg_glucose', 'insulin_')) for key in features.keys())
        
        if is_diatrend:
            return self.build_diatrend_prompt(features, tone, user_goal)
        
        # Validate tone
        if tone not in self.templates:
            logger.warning(f"Tone '{tone}' not found in templates, defaulting to 'coach'")
            tone = "coach"
        
        # Format data summary
        data_summary = self.format_data_summary(
            features, 
            include_categories=include_categories,
            exclude_features=exclude_features
        )
        
        # Get template
        template = self.templates[tone]
        
        # Build system prompt
        system_prompt = template["system"]
        
        # Build user prompt with data summary and goal
        user_prompt = template["user"].format(
            data_summary=data_summary,
            user_goal=user_goal
        )
        
        # Add time range if provided
        if time_range:
            user_prompt = user_prompt.replace("recent biometric data", f"biometric data from {time_range}")
            user_prompt = user_prompt.replace("recent physiological data", f"physiological data from {time_range}")
            user_prompt = user_prompt.replace("recent wellness data", f"wellness data from {time_range}")
        
        # Add additional context if provided
        if additional_context:
            user_prompt += f"\n\nAdditional context: {additional_context}"
        
        return {
            "system": system_prompt,
            "user": user_prompt
        }
    
    def build_diatrend_prompt(self, features: Dict[str, Any], tone: str = 'coach', user_goal: str = None) -> Dict[str, str]:
        """
        Build a prompt specifically for DiaTrend glucose and insulin data.
        
        Args:
            features: Dictionary of DiaTrend features
            tone: Tone of the prompt (coach, medical, neutral)
            user_goal: User's diabetes management goal
            
        Returns:
            Dictionary with system and user prompts
        """
        # Default diabetes management goal if none provided
        if not user_goal:
            user_goal = "Maintain stable glucose levels and reduce glycemic variability"
        
        # Build system prompt
        system_prompt = self._build_diatrend_system_prompt(tone)
        
        # Build user prompt
        user_prompt = self._build_diatrend_user_prompt(features, user_goal)
        
        return {
            'system': system_prompt,
            'user': user_prompt
        }
    
    def _build_diatrend_system_prompt(self, tone: str) -> str:
        """
        Build a system prompt for DiaTrend data.
        
        Args:
            tone: Tone of the prompt
            
        Returns:
            System prompt string
        """
        base_prompt = """You are an AI assistant specialized in analyzing glucose and insulin data for people with diabetes. 
Your role is to provide helpful insights based on the data provided. 
Focus on patterns, trends, and actionable recommendations that could help improve glucose management.
"""
        
        if tone == 'coach':
            base_prompt += """
Adopt a supportive coaching tone. Be encouraging and motivational.
Focus on progress, not perfection. Highlight what's working well and suggest small, achievable improvements.
Avoid medical jargon when possible and explain concepts in simple terms.
Frame recommendations positively and emphasize the benefits of making changes.
"""
        elif tone == 'medical':
            base_prompt += """
Adopt a professional medical tone. Be precise and evidence-based.
Reference established diabetes management guidelines when appropriate.
Use proper medical terminology but explain complex concepts when necessary.
Focus on clinical outcomes and risk reduction.
"""
        elif tone == 'neutral':
            base_prompt += """
Adopt a neutral, informative tone. Be clear and concise.
Present the data objectively without emotional language.
Balance technical information with accessibility.
Focus on education and understanding of the data.
"""
        
        base_prompt += """
Important guidelines:
1. Never claim to provide medical advice or diagnosis.
2. Always encourage users to consult healthcare providers for medical decisions.
3. Focus on general patterns rather than specific medical recommendations.
4. Acknowledge the limitations of the data analysis.
5. Be sensitive to the challenges of diabetes management.
"""
        
        return base_prompt
    
    def _build_diatrend_user_prompt(self, features: Dict[str, Any], user_goal: str) -> str:
        """
        Build a user prompt for DiaTrend data.
        
        Args:
            features: Dictionary of DiaTrend features
            user_goal: User's diabetes management goal
            
        Returns:
            User prompt string
        """
        # Extract key metrics
        avg_glucose = features.get('avg_glucose', 'unknown')
        min_glucose = features.get('min_glucose', 'unknown')
        max_glucose = features.get('max_glucose', 'unknown')
        glucose_range = features.get('glucose_range', 'unknown')
        glucose_std = features.get('glucose_std', 'unknown')
        
        hypo_events = features.get('hypo_events', 0)
        hyper_events = features.get('hyper_events', 0)
        time_in_range = features.get('time_in_range_percent', 'unknown')
        
        total_insulin = features.get('total_daily_insulin', 'unknown')
        insulin_doses = features.get('insulin_doses_count', 'unknown')
        
        # Glucose volatility
        volatility = features.get('glucose_volatility_2hr_mean', 'unknown')
        
        # Meal response
        pre_meal_slope = features.get('pre_meal_glucose_slope_mean', 'unknown')
        post_meal_slope = features.get('post_meal_glucose_slope_mean', 'unknown')
        
        # Comment sentiment
        sentiment = features.get('comment_sentiment_score', 'unknown')
        
        # Tags
        tags = []
        for key, value in features.items():
            if key.startswith('tag_') and value == 1:
                tags.append(key.replace('tag_', ''))
        
        # Build the prompt
        prompt = f"""Please analyze the following diabetes management data and provide insights:

USER PROFILE:
- Goal: {user_goal}
- Tags: {', '.join(tags) if tags else 'None'}

GLUCOSE METRICS:
- Average glucose: {avg_glucose} mg/dL
- Range: {min_glucose} - {max_glucose} mg/dL (span of {glucose_range} mg/dL)
- Standard deviation: {glucose_std} mg/dL
- Time in range (70-180 mg/dL): {time_in_range}%
- Hypoglycemic events (<70 mg/dL): {hypo_events}
- Hyperglycemic events (>180 mg/dL): {hyper_events}
- 2-hour glucose volatility: {volatility}

INSULIN DATA:
- Total daily insulin: {total_insulin} units
- Number of insulin doses: {insulin_doses}

MEAL RESPONSE:
- Pre-meal glucose trend: {pre_meal_slope} mg/dL/min
- Post-meal glucose trend: {post_meal_slope} mg/dL/min

"""
        
        # Add comment sentiment if available
        if sentiment != 'unknown':
            sentiment_description = "positive" if sentiment > 0.3 else "negative" if sentiment < -0.3 else "neutral"
            prompt += f"COMMENT SENTIMENT: {sentiment_description} ({sentiment})\n\n"
        
        # Request specific insights
        prompt += """Based on this data, please provide:
1. A summary of the overall glucose management for this period
2. Identification of any concerning patterns or positive trends
3. 2-3 specific, actionable recommendations to improve glucose management
4. Any correlations between the tags/activities and glucose patterns

Please format your response in a clear, structured way that would be helpful for someone managing their diabetes.
"""
        
        return prompt
    
    def build_comparative_prompt(self, current_features: Dict[str, Any],
                               previous_features: Dict[str, Any],
                               tone: str = "coach",
                               user_goal: str = "improving overall wellness",
                               include_categories: Optional[List[str]] = None,
                               exclude_features: Optional[List[str]] = None) -> Dict[str, str]:
        """
        Build a prompt comparing current data with previous period.
        
        Args:
            current_features: Dictionary of current features
            previous_features: Dictionary of previous period features
            tone: Tone of the prompt
            user_goal: User's stated goal
            include_categories: Optional list of feature categories to include
            exclude_features: Optional list of specific features to exclude
            
        Returns:
            Dictionary with system and user prompts
        """
        # Format current and previous data summaries
        current_summary = self.format_data_summary(
            current_features,
            include_categories=include_categories,
            exclude_features=exclude_features
        )
        
        previous_summary = self.format_data_summary(
            previous_features,
            include_categories=include_categories,
            exclude_features=exclude_features
        )
        
        # Get template
        if tone not in self.templates:
            logger.warning(f"Tone '{tone}' not found in templates, defaulting to 'coach'")
            tone = "coach"
            
        template = self.templates[tone]
        
        # Build system prompt with comparative focus
        system_prompt = template["system"] + " Focus on comparing current data with previous period data to identify trends and changes."
        
        # Build user prompt
        user_prompt = f"""Here's my biometric data from the current period:
{current_summary}

And here's my data from the previous period:
{previous_summary}

My goal is {user_goal}. What trends or changes do you notice between these periods? What insights and recommendations can you provide based on these changes?"""
        
        return {
            "system": system_prompt,
            "user": user_prompt
        }
    
    def build_focused_prompt(self, features: Dict[str, Any],
                           focus_area: str,
                           tone: str = "coach",
                           user_goal: Optional[str] = None) -> Dict[str, str]:
        """
        Build a prompt focused on a specific area (sleep, recovery, activity, etc.).
        
        Args:
            features: Dictionary of features
            focus_area: Area to focus on (sleep, recovery, activity, etc.)
            tone: Tone of the prompt
            user_goal: Optional user's stated goal specific to the focus area
            
        Returns:
            Dictionary with system and user prompts
        """
        # Map focus areas to relevant categories and specialized tone
        focus_map = {
            "sleep": {
                "categories": ["sleep", "hrv"],
                "specialized_tone": "sleep_specialist",
                "default_goal": "improving sleep quality"
            },
            "recovery": {
                "categories": ["hrv", "sleep", "training"],
                "specialized_tone": "coach",
                "default_goal": "optimizing recovery between workouts"
            },
            "activity": {
                "categories": ["activity", "training"],
                "specialized_tone": "coach",
                "default_goal": "optimizing training and activity patterns"
            },
            "stress": {
                "categories": ["hrv", "stress"],
                "specialized_tone": "medical",
                "default_goal": "managing stress levels"
            }
        }
        
        # Get focus configuration
        focus_config = focus_map.get(focus_area.lower(), {
            "categories": None,  # Include all categories
            "specialized_tone": tone,
            "default_goal": "improving overall wellness"
        })
        
        # Use specialized tone if available and no specific tone requested
        if tone == "coach" and "specialized_tone" in focus_config:
            tone = focus_config["specialized_tone"]
            
        # Use focus-specific goal if no user goal provided
        if user_goal is None:
            user_goal = focus_config["default_goal"]
            
        # Build prompt with focus area categories
        return self.build_prompt(
            features,
            tone=tone,
            user_goal=user_goal,
            include_categories=focus_config["categories"]
        )
    
    def add_few_shot_examples(self, prompt: Dict[str, str], 
                             examples: List[Dict[str, str]]) -> Dict[str, str]:
        """
        Add few-shot examples to a prompt.
        
        Args:
            prompt: Dictionary with system and user prompts
            examples: List of dictionaries with example inputs and outputs
            
        Returns:
            Updated prompt dictionary
        """
        # Create few-shot examples text
        few_shot_text = "\n\nHere are some examples of how to analyze similar data:\n\n"
        
        for i, example in enumerate(examples):
            few_shot_text += f"Example {i+1}:\n"
            few_shot_text += f"Input: {example.get('input', '')}\n"
            few_shot_text += f"Output: {example.get('output', '')}\n\n"
            
        # Add examples to system prompt
        updated_system = prompt["system"] + few_shot_text
        
        return {
            "system": updated_system,
            "user": prompt["user"]
        }
    
    def save_prompt_template(self, name: str, system_prompt: str, user_prompt: str, 
                           output_path: str) -> None:
        """
        Save a new prompt template to a JSON file.
        
        Args:
            name: Name of the template
            system_prompt: System prompt text
            user_prompt: User prompt text
            output_path: Path to save the template
        """
        # Create new template
        new_template = {
            name: {
                "system": system_prompt,
                "user": user_prompt
            }
        }
        
        try:
            # Load existing templates if file exists
            existing_templates = {}
            try:
                with open(output_path, 'r') as f:
                    existing_templates = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                pass
                
            # Update with new template
            existing_templates.update(new_template)
            
            # Save updated templates
            with open(output_path, 'w') as f:
                json.dump(existing_templates, f, indent=2)
                
            logger.info(f"Saved template '{name}' to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving template to {output_path}: {str(e)}")


# Example usage
if __name__ == "__main__":
    # Sample features data
    sample_features = {
        "hrv_rmssd_mean": 65.3,
        "hrv_rmssd_min": 45.2,
        "hrv_rmssd_max": 85.7,
        "activity_total_steps": 8500,
        "activity_active_minutes": 45,
        "sleep_total_sleep_hours": 7.2,
        "sleep_deep_sleep_percentage": 22.5,
        "sleep_rem_sleep_percentage": 18.3,
        "recovery_score": 75.8
    }
    
    # Initialize prompt builder
    prompt_builder = InsightPromptBuilder()
    
    # Build a coach-tone prompt
    coach_prompt = prompt_builder.build_prompt(
        sample_features,
        tone="coach",
        user_goal="improving my recovery between workouts"
    )
    
    print("=== COACH PROMPT ===")
    print("System:", coach_prompt["system"])
    print("\nUser:", coach_prompt["user"])
    
    # Build a focused prompt for sleep
    sleep_prompt = prompt_builder.build_focused_prompt(
        sample_features,
        focus_area="sleep",
        user_goal="addressing my early morning wakeups"
    )
    
    print("\n\n=== SLEEP FOCUSED PROMPT ===")
    print("System:", sleep_prompt["system"])
    print("\nUser:", sleep_prompt["user"])
