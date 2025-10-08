# src/reasoners/llm/openrouter_client.py
import os
from openai import OpenAI, RateLimitError as OpenAIRateLimitError
from src.logger import logger

# Custom exception for clarity
class RateLimitError(Exception):
    pass

class OpenRouterClient:
    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key or self.api_key == "your_api_key_here":
            raise ValueError("OPENROUTER_API_KEY not found or not set in .env file.")

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
        )
        self.model = "meta-llama/llama-3-8b-instruct"
        logger.info(f"OpenRouter client initialized for model '{self.model}'.")

    def generate_response(self, prompt: str) -> str:
        try:
            # Note: OpenRouter uses the 'user' role for prompts, not a raw string.
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
            )
            return completion.choices[0].message.content
        except OpenAIRateLimitError:
            # Catch the specific rate limit error and re-raise our custom one
            raise RateLimitError("OpenRouter API rate limit reached.")
        except Exception as e:
            logger.error(f"An error occurred with OpenRouter client: {e}")
            raise