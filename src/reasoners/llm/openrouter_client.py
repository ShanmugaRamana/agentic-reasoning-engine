# src/reasoners/llm/openrouter_client.py
import os
import requests
from src.logger import logger

# Custom exception for clarity
class RateLimitError(Exception):
    pass

class OpenRouterClient:
    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key or self.api_key == "your_api_key_here":
            raise ValueError("OPENROUTER_API_KEY not found or not set in .env file.")

        self.base_url = "https://openrouter.ai/api/v1"
        self.model = "meta-llama/llama-3-8b-instruct"
        logger.info(f"OpenRouter client initialized for model '{self.model}'.")

    def generate_response(self, prompt: str) -> str:
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
            }
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )
            
            # Check for rate limit error (HTTP 429)
            if response.status_code == 429:
                raise RateLimitError("OpenRouter API rate limit reached.")
            
            # Raise an exception for other HTTP errors
            response.raise_for_status()
            
            # Parse the response
            data = response.json()
            return data["choices"][0]["message"]["content"]
            
        except RateLimitError:
            # Re-raise our custom rate limit error
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"An error occurred with OpenRouter client: {e}")
            raise
        except (KeyError, IndexError) as e:
            logger.error(f"Unexpected response format from OpenRouter: {e}")
            raise