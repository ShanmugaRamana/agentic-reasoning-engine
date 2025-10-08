# src/reasoners/llm/llm_factory.py
from dotenv import load_dotenv
from src.logger import logger
from .openrouter_client import OpenRouterClient, RateLimitError
from .ollama_client import OllamaClient

class LLMFactory:
    def __init__(self):
        load_dotenv() 
        self.primary_client_type = 'openrouter'
        self.openrouter_client = None
        self.ollama_client = None

        try:
            self.openrouter_client = OpenRouterClient()
            logger.info("LLMFactory: Using OpenRouter as the primary client.")
        except ValueError as e:
            logger.warning(f"LLMFactory: {e}. Falling back to Ollama.")
            self.primary_client_type = 'ollama'
            self.ollama_client = OllamaClient()

    def get_client(self):
        """Returns the current active client."""
        if self.primary_client_type == 'openrouter':
            return self.openrouter_client
        else:
            if not self.ollama_client:
                self.ollama_client = OllamaClient()
            return self.ollama_client

    def generate_response(self, prompt: str) -> str:
        """
        Generates a response using the primary client, with a fallback to Ollama
        in case of a rate limit error.
        """
        # --- NEW: Log which client is being used for the call ---
        logger.info(f"Attempting to generate response using '{self.primary_client_type}' client.")

        if self.primary_client_type == 'openrouter':
            try:
                return self.openrouter_client.generate_response(prompt)
            except RateLimitError:
                logger.warning("OpenRouter rate limit hit. Switching to Ollama for this session.")
                self.primary_client_type = 'ollama'
            except Exception as e:
                logger.error(f"OpenRouter failed: {e}. Switching to Ollama for this session.")
                self.primary_client_type = 'ollama'

        if self.primary_client_type == 'ollama':
            if not self.ollama_client:
                self.ollama_client = OllamaClient()
            return self.ollama_client.generate_response(prompt)