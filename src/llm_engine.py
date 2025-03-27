"""
LLM Engine Module for Wearable LLM Insight Generator

This module integrates with various LLM APIs (OpenAI, Mistral, Gemini) to generate
insights from wearable data using structured prompts.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Union, Any, Tuple
import time
import requests
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LLMEngine:
    """Class to interact with various LLM APIs for insight generation."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the LLM Engine.
        
        Args:
            config_path: Optional path to JSON file with API configurations
        """
        # Default configuration
        self.config = {
            "openai": {
                "api_key": os.environ.get("OPENAI_API_KEY", ""),
                "model": "gpt-4-turbo",
                "temperature": 0.7,
                "max_tokens": 1000
            },
            "mistral": {
                "api_key": os.environ.get("MISTRAL_API_KEY", ""),
                "model": "mistral-large-latest",
                "temperature": 0.7,
                "max_tokens": 1000
            },
            "gemini": {
                "api_key": os.environ.get("GEMINI_API_KEY", ""),
                "model": "gemini-pro",
                "temperature": 0.7,
                "max_tokens": 1000
            },
            "default_provider": "openai"
        }
        
        # Load custom configuration if provided
        if config_path:
            try:
                with open(config_path, 'r') as f:
                    custom_config = json.load(f)
                    # Update config with custom values
                    for provider, settings in custom_config.items():
                        if provider in self.config:
                            self.config[provider].update(settings)
                        else:
                            self.config[provider] = settings
                logger.info(f"Loaded custom LLM configuration from {config_path}")
            except Exception as e:
                logger.error(f"Error loading LLM configuration from {config_path}: {str(e)}")
    
    def generate_insight(self, 
                        prompt: Dict[str, str],
                        provider: Optional[str] = None,
                        model: Optional[str] = None,
                        temperature: Optional[float] = None,
                        max_tokens: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Generate insights using the specified LLM provider.
        
        Args:
            prompt: Dictionary with system and user prompts
            provider: LLM provider to use (openai, mistral, gemini)
            model: Specific model to use (overrides config)
            temperature: Temperature setting (overrides config)
            max_tokens: Maximum tokens to generate (overrides config)
            
        Returns:
            Tuple of (generated text, metadata)
        """
        # Use default provider if not specified
        if not provider:
            provider = self.config.get("default_provider", "openai")
            
        # Check if provider is supported
        if provider not in self.config:
            logger.error(f"Provider '{provider}' not supported")
            return "Error: Unsupported LLM provider", {"error": "Unsupported provider"}
            
        # Get provider config
        provider_config = self.config[provider]
        
        # Override config with function arguments if provided
        if model:
            provider_config = {**provider_config, "model": model}
        if temperature is not None:
            provider_config = {**provider_config, "temperature": temperature}
        if max_tokens:
            provider_config = {**provider_config, "max_tokens": max_tokens}
            
        # Check for API key
        api_key = provider_config.get("api_key", "")
        if not api_key:
            logger.error(f"No API key found for provider '{provider}'")
            return "Error: Missing API key", {"error": "Missing API key"}
            
        # Generate based on provider
        if provider == "openai":
            return self._generate_openai(prompt, provider_config)
        elif provider == "mistral":
            return self._generate_mistral(prompt, provider_config)
        elif provider == "gemini":
            return self._generate_gemini(prompt, provider_config)
        else:
            logger.error(f"Provider '{provider}' implementation not found")
            return "Error: Provider implementation not found", {"error": "Implementation not found"}
    
    def _generate_openai(self, prompt: Dict[str, str], config: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """
        Generate insights using OpenAI API.
        
        Args:
            prompt: Dictionary with system and user prompts
            config: OpenAI configuration
            
        Returns:
            Tuple of (generated text, metadata)
        """
        try:
            import openai
            
            # Configure client
            client = openai.OpenAI(api_key=config["api_key"])
            
            # Prepare messages
            messages = [
                {"role": "system", "content": prompt["system"]},
                {"role": "user", "content": prompt["user"]}
            ]
            
            # Make API call
            start_time = time.time()
            response = client.chat.completions.create(
                model=config["model"],
                messages=messages,
                temperature=config["temperature"],
                max_tokens=config["max_tokens"]
            )
            end_time = time.time()
            
            # Extract response
            generated_text = response.choices[0].message.content
            
            # Prepare metadata
            metadata = {
                "provider": "openai",
                "model": config["model"],
                "temperature": config["temperature"],
                "max_tokens": config["max_tokens"],
                "latency_seconds": end_time - start_time,
                "timestamp": datetime.now().isoformat(),
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
            
            logger.info(f"Generated OpenAI response with {metadata['total_tokens']} tokens")
            return generated_text, metadata
            
        except ImportError:
            logger.error("OpenAI package not installed. Install with 'pip install openai'")
            return "Error: OpenAI package not installed", {"error": "Package not installed"}
        except Exception as e:
            logger.error(f"Error generating OpenAI response: {str(e)}")
            return f"Error: {str(e)}", {"error": str(e)}
    
    def _generate_mistral(self, prompt: Dict[str, str], config: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """
        Generate insights using Mistral AI API.
        
        Args:
            prompt: Dictionary with system and user prompts
            config: Mistral configuration
            
        Returns:
            Tuple of (generated text, metadata)
        """
        try:
            import mistralai.client
            from mistralai.client import MistralClient
            from mistralai.models.chat_completion import ChatMessage
            
            # Configure client
            client = MistralClient(api_key=config["api_key"])
            
            # Prepare messages
            messages = [
                ChatMessage(role="system", content=prompt["system"]),
                ChatMessage(role="user", content=prompt["user"])
            ]
            
            # Make API call
            start_time = time.time()
            response = client.chat(
                model=config["model"],
                messages=messages,
                temperature=config["temperature"],
                max_tokens=config["max_tokens"]
            )
            end_time = time.time()
            
            # Extract response
            generated_text = response.choices[0].message.content
            
            # Prepare metadata
            metadata = {
                "provider": "mistral",
                "model": config["model"],
                "temperature": config["temperature"],
                "max_tokens": config["max_tokens"],
                "latency_seconds": end_time - start_time,
                "timestamp": datetime.now().isoformat(),
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
            
            logger.info(f"Generated Mistral response with {metadata['total_tokens']} tokens")
            return generated_text, metadata
            
        except ImportError:
            logger.error("Mistral AI package not installed. Install with 'pip install mistralai'")
            return "Error: Mistral AI package not installed", {"error": "Package not installed"}
        except Exception as e:
            logger.error(f"Error generating Mistral response: {str(e)}")
            return f"Error: {str(e)}", {"error": str(e)}
    
    def _generate_gemini(self, prompt: Dict[str, str], config: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """
        Generate insights using Google Gemini API.
        
        Args:
            prompt: Dictionary with system and user prompts
            config: Gemini configuration
            
        Returns:
            Tuple of (generated text, metadata)
        """
        try:
            import google.generativeai as genai
            
            # Configure client
            genai.configure(api_key=config["api_key"])
            
            # Combine system and user prompts (Gemini doesn't have system prompt)
            combined_prompt = f"{prompt['system']}\n\n{prompt['user']}"
            
            # Make API call
            start_time = time.time()
            model = genai.GenerativeModel(
                model_name=config["model"],
                generation_config={
                    "temperature": config["temperature"],
                    "max_output_tokens": config["max_tokens"],
                }
            )
            response = model.generate_content(combined_prompt)
            end_time = time.time()
            
            # Extract response
            generated_text = response.text
            
            # Prepare metadata (Gemini doesn't provide token counts)
            metadata = {
                "provider": "gemini",
                "model": config["model"],
                "temperature": config["temperature"],
                "max_tokens": config["max_tokens"],
                "latency_seconds": end_time - start_time,
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"Generated Gemini response")
            return generated_text, metadata
            
        except ImportError:
            logger.error("Google Generative AI package not installed. Install with 'pip install google-generativeai'")
            return "Error: Google Generative AI package not installed", {"error": "Package not installed"}
        except Exception as e:
            logger.error(f"Error generating Gemini response: {str(e)}")
            return f"Error: {str(e)}", {"error": str(e)}
    
    def save_response(self, 
                     prompt: Dict[str, str], 
                     response: str, 
                     metadata: Dict[str, Any],
                     output_dir: str,
                     filename: Optional[str] = None) -> str:
        """
        Save prompt, response, and metadata to a JSON file.
        
        Args:
            prompt: Dictionary with system and user prompts
            response: Generated response text
            metadata: Response metadata
            output_dir: Directory to save the output
            filename: Optional filename (defaults to timestamp)
            
        Returns:
            Path to the saved file
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate filename if not provided
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"insight_{timestamp}.json"
            
        # Ensure filename has .json extension
        if not filename.endswith('.json'):
            filename += '.json'
            
        # Prepare output data
        output_data = {
            "prompt": prompt,
            "response": response,
            "metadata": metadata
        }
        
        # Save to file
        output_path = os.path.join(output_dir, filename)
        try:
            with open(output_path, 'w') as f:
                json.dump(output_data, f, indent=2)
            logger.info(f"Saved response to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error saving response to {output_path}: {str(e)}")
            return ""
    
    def batch_generate(self, 
                      prompts: List[Dict[str, str]],
                      provider: Optional[str] = None,
                      output_dir: Optional[str] = None) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Generate insights for multiple prompts.
        
        Args:
            prompts: List of prompt dictionaries
            provider: LLM provider to use
            output_dir: Optional directory to save responses
            
        Returns:
            List of (response, metadata) tuples
        """
        results = []
        
        for i, prompt in enumerate(prompts):
            logger.info(f"Generating response for prompt {i+1}/{len(prompts)}")
            
            # Generate response
            response, metadata = self.generate_insight(prompt, provider=provider)
            results.append((response, metadata))
            
            # Save response if output directory provided
            if output_dir:
                self.save_response(
                    prompt, 
                    response, 
                    metadata, 
                    output_dir, 
                    filename=f"batch_{i+1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                )
            
            # Add small delay to avoid rate limiting
            if i < len(prompts) - 1:
                time.sleep(1)
        
        return results


# Example usage
if __name__ == "__main__":
    # Sample prompt
    sample_prompt = {
        "system": "You are an expert fitness coach analyzing biometric data.",
        "user": "Here's my recent HRV data:\n- Average RMSSD: 65.3\n- Sleep: 7.2 hours\n\nWhat insights can you provide?"
    }
    
    # Initialize LLM engine
    llm_engine = LLMEngine()
    
    # Check if API key is set
    if os.environ.get("OPENAI_API_KEY"):
        # Generate insight
        response, metadata = llm_engine.generate_insight(sample_prompt)
        
        print("=== GENERATED INSIGHT ===")
        print(response)
        print("\n=== METADATA ===")
        print(json.dumps(metadata, indent=2))
        
        # Save response
        llm_engine.save_response(sample_prompt, response, metadata, "../outputs")
    else:
        print("OpenAI API key not set. Please set the OPENAI_API_KEY environment variable.")
        print("Example prompt that would be sent:")
        print("System:", sample_prompt["system"])
        print("User:", sample_prompt["user"])
