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
            "local": {
                "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # Small model that works on most hardware
                "temperature": 0.7,
                "max_tokens": 1000,
                "device": "auto",  # "cpu", "cuda", "mps", or "auto" to detect automatically
                "quantization": "4bit",  # "4bit", "8bit", or None for full precision
                "cache_dir": "./models"  # Directory to cache downloaded models
            },
            "default_provider": "local"  # Changed default to local
        }
        
        # Initialize model cache
        self.models_cache = {}
        
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
            provider: LLM provider to use (openai, mistral, gemini, local)
            model: Specific model to use (overrides config)
            temperature: Temperature setting (overrides config)
            max_tokens: Maximum tokens to generate (overrides config)
            
        Returns:
            Tuple of (generated text, metadata)
        """
        # Use default provider if not specified
        if not provider:
            provider = self.config.get("default_provider", "local")
            
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
            
        # Check for API key for cloud providers
        if provider in ["openai", "mistral", "gemini"]:
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
        elif provider == "local":
            return self._generate_local(prompt, provider_config)
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
            
            # Get model
            model = genai.GenerativeModel(config["model"])
            
            # Combine prompts (Gemini has different API)
            combined_prompt = f"{prompt['system']}\n\n{prompt['user']}"
            
            # Make API call
            start_time = time.time()
            response = model.generate_content(
                combined_prompt,
                generation_config={
                    "temperature": config["temperature"],
                    "max_output_tokens": config["max_tokens"]
                }
            )
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
            
            logger.info(f"Generated Gemini response in {metadata['latency_seconds']:.2f} seconds")
            return generated_text, metadata
            
        except ImportError:
            logger.error("Google Generative AI package not installed. Install with 'pip install google-generativeai'")
            return "Error: Google Generative AI package not installed", {"error": "Package not installed"}
        except Exception as e:
            logger.error(f"Error generating Gemini response: {str(e)}")
            return f"Error: {str(e)}", {"error": str(e)}

    def _generate_local(self, prompt: Dict[str, str], config: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """
        Generate insights using local Hugging Face Transformers models.
        
        Args:
            prompt: Dictionary with system and user prompts
            config: Local model configuration
            
        Returns:
            Tuple of (generated text, metadata)
        """
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            import torch
            
            model_id = config["model"]
            cache_key = f"{model_id}_{config.get('quantization', 'none')}"
            
            # Determine device
            if config.get("device", "auto") == "auto":
                if torch.cuda.is_available():
                    device = "cuda"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    device = "mps"
                else:
                    device = "cpu"
            else:
                device = config.get("device", "cpu")
                
            logger.info(f"Using device: {device} for local model inference")
            
            # Load model and tokenizer from cache or initialize
            if cache_key not in self.models_cache:
                logger.info(f"Loading model {model_id} (this may take a while for the first run)")
                
                # Set up quantization if requested
                quantization = config.get("quantization")
                if quantization == "4bit":
                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_quant_type="nf4"
                    )
                elif quantization == "8bit":
                    bnb_config = BitsAndBytesConfig(
                        load_in_8bit=True
                    )
                else:
                    bnb_config = None
                
                # Load tokenizer and model
                tokenizer = AutoTokenizer.from_pretrained(
                    model_id, 
                    cache_dir=config.get("cache_dir"),
                    trust_remote_code=True
                )
                
                if bnb_config:
                    model = AutoModelForCausalLM.from_pretrained(
                        model_id,
                        quantization_config=bnb_config,
                        device_map=device,
                        cache_dir=config.get("cache_dir"),
                        trust_remote_code=True
                    )
                else:
                    model = AutoModelForCausalLM.from_pretrained(
                        model_id,
                        device_map=device,
                        cache_dir=config.get("cache_dir"),
                        trust_remote_code=True
                    )
                
                # Cache the model and tokenizer
                self.models_cache[cache_key] = (model, tokenizer)
            else:
                model, tokenizer = self.models_cache[cache_key]
            
            # Format prompt based on model type
            if "llama" in model_id.lower():
                # Llama 2/3 style prompt format
                formatted_prompt = f"<s>[INST] <<SYS>>\n{prompt['system']}\n<</SYS>>\n\n{prompt['user']} [/INST]"
            elif "mistral" in model_id.lower():
                # Mistral style prompt format
                formatted_prompt = f"<s>[INST] {prompt['system']}\n\n{prompt['user']} [/INST]"
            elif "gemma" in model_id.lower():
                # Gemma style prompt format
                formatted_prompt = f"<start_of_turn>user\n{prompt['system']}\n{prompt['user']}<end_of_turn>\n<start_of_turn>model\n"
            else:
                # Generic chat format
                formatted_prompt = f"System: {prompt['system']}\nUser: {prompt['user']}\nAssistant:"
            
            # Tokenize input
            inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
            input_tokens = inputs.input_ids.shape[1]
            
            # Generate response
            start_time = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    max_new_tokens=config["max_tokens"],
                    temperature=config["temperature"],
                    do_sample=config["temperature"] > 0,
                    pad_token_id=tokenizer.eos_token_id
                )
            end_time = time.time()
            
            # Decode and clean up response
            full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract just the model's response (remove the prompt)
            if "Assistant:" in full_output:
                generated_text = full_output.split("Assistant:", 1)[1].strip()
            elif "[/INST]" in full_output:
                generated_text = full_output.split("[/INST]", 1)[1].strip()
            elif "<end_of_turn>" in full_output:
                generated_text = full_output.split("<start_of_turn>model\n", 1)[1].split("<end_of_turn>", 1)[0].strip()
            else:
                # Just return everything after the prompt as a fallback
                generated_text = full_output[len(formatted_prompt):].strip()
            
            # Calculate token counts
            output_tokens = outputs.shape[1] - input_tokens
            
            # Prepare metadata
            metadata = {
                "provider": "local",
                "model": model_id,
                "temperature": config["temperature"],
                "max_tokens": config["max_tokens"],
                "device": device,
                "quantization": config.get("quantization"),
                "latency_seconds": end_time - start_time,
                "timestamp": datetime.now().isoformat(),
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens
            }
            
            logger.info(f"Generated local model response in {metadata['latency_seconds']:.2f} seconds")
            return generated_text, metadata
            
        except ImportError:
            logger.error("Required packages not installed. Install with 'pip install transformers torch accelerate bitsandbytes'")
            return "Error: Required packages not installed", {"error": "Package not installed"}
        except Exception as e:
            logger.error(f"Error generating local model response: {str(e)}")
            return f"Error: {str(e)}", {"error": str(e)}


# Example usage
if __name__ == "__main__":
    # Sample prompt
    sample_prompt = {
        "system": "You are an AI assistant that analyzes health data and provides insights.",
        "user": "My resting heart rate has increased from 60 to 68 over the past week. What might this indicate?"
    }
    
    # Initialize engine
    engine = LLMEngine()
    
    # Generate insight using local model (default)
    response, metadata = engine.generate_insight(sample_prompt)
    print(f"Local model response: {response}")
    print(f"Metadata: {metadata}")
    
    # You can also try different models
    # For example, a smaller model that runs well on CPU:
    response, metadata = engine.generate_insight(
        sample_prompt, 
        provider="local",
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    )
