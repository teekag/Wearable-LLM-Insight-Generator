"""
Agent Simulator Module for Wearable LLM Insight Generator

This module provides conversational agent logic for interactive coaching
based on wearable data and user goals.
"""

import json
import os
from typing import Dict, List, Optional, Union, Any, Tuple
import logging
from datetime import datetime, timedelta
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AgentSimulator:
    """Class to simulate an interactive coaching agent based on wearable data."""
    
    def __init__(self, llm_engine, feature_data: Optional[Dict[str, Any]] = None):
        """
        Initialize the Agent Simulator.
        
        Args:
            llm_engine: LLM engine instance for generating responses
            feature_data: Optional dictionary of pre-loaded feature data
        """
        self.llm_engine = llm_engine
        self.feature_data = feature_data or {}
        self.conversation_history = []
        self.user_goals = {}
        self.agent_persona = {
            "name": "Insight Coach",
            "role": "Wearable Data Coach",
            "tone": "coach",
            "expertise": ["sleep", "recovery", "training", "stress management"]
        }
    
    def load_feature_data(self, feature_data: Dict[str, Any]) -> None:
        """
        Load feature data for the agent to use.
        
        Args:
            feature_data: Dictionary of feature data
        """
        self.feature_data = feature_data
        logger.info("Loaded feature data for agent")
    
    def set_user_goals(self, goals: Dict[str, Any]) -> None:
        """
        Set user goals for the agent to reference.
        
        Args:
            goals: Dictionary of user goals
        """
        self.user_goals = goals
        logger.info("Set user goals for agent")
    
    def set_agent_persona(self, persona: Dict[str, Any]) -> None:
        """
        Set the agent's persona.
        
        Args:
            persona: Dictionary with agent persona details
        """
        self.agent_persona.update(persona)
        logger.info(f"Updated agent persona: {self.agent_persona['name']} ({self.agent_persona['role']})")
    
    def _build_system_prompt(self) -> str:
        """
        Build the system prompt for the agent.
        
        Returns:
            System prompt string
        """
        # Basic system prompt
        system_prompt = f"You are {self.agent_persona['name']}, a {self.agent_persona['role']} specializing in "
        system_prompt += ", ".join(self.agent_persona['expertise']) + ".\n\n"
        
        # Add user goals if available
        if self.user_goals:
            system_prompt += "The user has the following goals:\n"
            for area, goal in self.user_goals.items():
                system_prompt += f"- {area}: {goal}\n"
            system_prompt += "\n"
        
        # Add conversation guidelines
        system_prompt += "Guidelines for your responses:\n"
        system_prompt += "1. Be concise and actionable in your advice\n"
        system_prompt += "2. Reference the user's biometric data when relevant\n"
        system_prompt += "3. Tailor your recommendations to the user's specific goals\n"
        system_prompt += "4. Maintain a supportive and motivational tone\n"
        system_prompt += "5. When uncertain, acknowledge limitations and avoid making definitive claims\n"
        
        return system_prompt
    
    def _build_user_message(self, user_input: str) -> str:
        """
        Build the user message with context.
        
        Args:
            user_input: Raw user input
            
        Returns:
            Formatted user message
        """
        # Start with the user's input
        message = user_input
        
        # Add feature data context if relevant to the query
        relevant_data = self._get_relevant_data(user_input)
        if relevant_data:
            message += "\n\nFor reference, here's my recent data:\n"
            message += relevant_data
        
        return message
    
    def _get_relevant_data(self, user_input: str) -> str:
        """
        Extract data relevant to the user's query.
        
        Args:
            user_input: User's input text
            
        Returns:
            Formatted string of relevant data
        """
        if not self.feature_data:
            return ""
            
        # Simple keyword matching to determine relevant data categories
        keywords = {
            "sleep": ["sleep", "rest", "bed", "tired", "fatigue", "nap"],
            "activity": ["workout", "exercise", "run", "training", "activity", "steps"],
            "recovery": ["recovery", "strain", "stress", "readiness", "hrv"],
            "hrv": ["hrv", "heart rate variability", "nervous system"]
        }
        
        # Find matching categories
        relevant_categories = []
        for category, terms in keywords.items():
            if any(term in user_input.lower() for term in terms):
                relevant_categories.append(category)
                
        # If no specific categories matched, return a summary
        if not relevant_categories:
            return self._format_data_summary(["sleep", "recovery", "activity"], 2)
            
        # Return data for relevant categories
        return self._format_data_summary(relevant_categories, 5)
    
    def _format_data_summary(self, categories: List[str], max_per_category: int = 3) -> str:
        """
        Format a summary of data for specific categories.
        
        Args:
            categories: List of data categories to include
            max_per_category: Maximum number of metrics per category
            
        Returns:
            Formatted data summary
        """
        summary = []
        
        for category in categories:
            category_metrics = []
            count = 0
            
            # Find metrics for this category
            for key, value in self.feature_data.items():
                if key.startswith(f"{category}_") and count < max_per_category:
                    # Format the metric name and value
                    metric_name = key.replace(f"{category}_", "").replace("_", " ").title()
                    
                    if isinstance(value, float):
                        formatted_value = f"{value:.1f}"
                    else:
                        formatted_value = str(value)
                        
                    category_metrics.append(f"- {metric_name}: {formatted_value}")
                    count += 1
            
            # Add category if we found metrics
            if category_metrics:
                summary.append(f"## {category.upper()}")
                summary.extend(category_metrics)
        
        return "\n".join(summary)
    
    def process_message(self, user_input: str) -> Tuple[str, Dict[str, Any]]:
        """
        Process a user message and generate a response.
        
        Args:
            user_input: User's message
            
        Returns:
            Tuple of (agent response, response metadata)
        """
        # Build the system prompt
        system_prompt = self._build_system_prompt()
        
        # Build the user message with context
        user_message = self._build_user_message(user_input)
        
        # Prepare conversation history for context
        conversation_context = ""
        if self.conversation_history:
            conversation_context = "Previous conversation:\n"
            for entry in self.conversation_history[-3:]:  # Last 3 exchanges
                conversation_context += f"User: {entry['user']}\n"
                conversation_context += f"Assistant: {entry['assistant']}\n\n"
            
            # Add to system prompt
            system_prompt += f"\n\n{conversation_context}"
        
        # Create prompt for LLM
        prompt = {
            "system": system_prompt,
            "user": user_message
        }
        
        # Generate response
        response, metadata = self.llm_engine.generate_insight(
            prompt,
            tone=self.agent_persona.get("tone", "coach")
        )
        
        # Update conversation history
        self.conversation_history.append({
            "user": user_input,
            "assistant": response,
            "timestamp": datetime.now().isoformat()
        })
        
        return response, metadata
    
    def save_conversation(self, output_path: str) -> None:
        """
        Save the conversation history to a file.
        
        Args:
            output_path: Path to save the conversation
        """
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Prepare conversation data
        conversation_data = {
            "agent_persona": self.agent_persona,
            "user_goals": self.user_goals,
            "conversation": self.conversation_history,
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "exchange_count": len(self.conversation_history)
            }
        }
        
        # Save to file
        try:
            with open(output_path, 'w') as f:
                json.dump(conversation_data, f, indent=2)
            logger.info(f"Saved conversation to {output_path}")
        except Exception as e:
            logger.error(f"Error saving conversation to {output_path}: {str(e)}")
    
    def load_conversation(self, input_path: str) -> bool:
        """
        Load a conversation history from a file.
        
        Args:
            input_path: Path to the conversation file
            
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            with open(input_path, 'r') as f:
                conversation_data = json.load(f)
                
            # Load conversation components
            if "agent_persona" in conversation_data:
                self.agent_persona = conversation_data["agent_persona"]
                
            if "user_goals" in conversation_data:
                self.user_goals = conversation_data["user_goals"]
                
            if "conversation" in conversation_data:
                self.conversation_history = conversation_data["conversation"]
                
            logger.info(f"Loaded conversation from {input_path} with {len(self.conversation_history)} exchanges")
            return True
            
        except Exception as e:
            logger.error(f"Error loading conversation from {input_path}: {str(e)}")
            return False
    
    def simulate_conversation(self, 
                            initial_prompt: str,
                            num_exchanges: int = 3,
                            output_path: Optional[str] = None) -> List[Dict[str, str]]:
        """
        Simulate a conversation with predefined user inputs.
        
        Args:
            initial_prompt: Initial user message
            num_exchanges: Number of exchanges to simulate
            output_path: Optional path to save the conversation
            
        Returns:
            List of conversation exchanges
        """
        # Start with initial prompt
        logger.info(f"Starting simulated conversation with: '{initial_prompt}'")
        response, _ = self.process_message(initial_prompt)
        
        # Define follow-up questions based on initial response
        follow_ups = [
            "Can you give me more specific recommendations?",
            "How does this compare to my previous data?",
            "What should I focus on improving first?",
            "What's the science behind this recommendation?",
            "How long until I should see improvements?"
        ]
        
        # Simulate additional exchanges
        for i in range(num_exchanges - 1):
            if i < len(follow_ups):
                follow_up = follow_ups[i]
            else:
                follow_up = "Thank you for the information."
                
            logger.info(f"Simulated user follow-up: '{follow_up}'")
            response, _ = self.process_message(follow_up)
        
        # Save conversation if output path provided
        if output_path:
            self.save_conversation(output_path)
        
        return self.conversation_history


# Example usage
if __name__ == "__main__":
    # This requires the LLMEngine from llm_engine.py
    from llm_engine import LLMEngine
    
    # Sample feature data
    sample_features = {
        "sleep_total_hours": 6.8,
        "sleep_deep_percentage": 22.5,
        "sleep_rem_percentage": 18.3,
        "activity_total_steps": 8500,
        "activity_active_minutes": 45,
        "recovery_score": 75.8,
        "hrv_rmssd_mean": 65.3
    }
    
    # Initialize LLM engine
    llm_engine = LLMEngine()
    
    # Initialize agent simulator
    agent = AgentSimulator(llm_engine, sample_features)
    
    # Set user goals
    agent.set_user_goals({
        "sleep": "Improve sleep quality and consistency",
        "activity": "Train for a half marathon in 3 months"
    })
    
    # Set agent persona
    agent.set_agent_persona({
        "name": "FitCoach",
        "tone": "motivational"
    })
    
    # Check if API key is set
    if os.environ.get("OPENAI_API_KEY"):
        # Process a user message
        response, metadata = agent.process_message(
            "I've been feeling tired after my workouts. What should I do?"
        )
        
        print("=== AGENT RESPONSE ===")
        print(response)
        
        # Simulate a conversation
        conversation = agent.simulate_conversation(
            "How should I adjust my training based on my recovery score?",
            num_exchanges=2,
            output_path="../outputs/sample_conversation.json"
        )
        
        print("\n=== SIMULATED CONVERSATION ===")
        for entry in conversation:
            print(f"User: {entry['user']}")
            print(f"Assistant: {entry['assistant']}")
            print()
    else:
        print("OpenAI API key not set. Please set the OPENAI_API_KEY environment variable.")
        print("Example agent persona:", agent.agent_persona)
        print("Example user goals:", agent.user_goals)
