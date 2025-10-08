# src/reasoners/llm/ollama_client.py

import ollama
import time
from src.logger import logger

class OllamaClient:
    """
    A robust client for interacting with a local Llama 3 model via Ollama.
    """
    def __init__(self, model: str = 'llama3', host: str = None, retries: int = 3, delay: int = 5):
        self.model = model
        self.retries = retries
        self.delay = delay
        try:
            self.client = ollama.Client(host=host)
            logger.info(f"Ollama client initialized for model '{self.model}' at host '{host or 'default'}'.")
            self._verify_connection()
        except Exception as e:
            logger.error(f"Failed to initialize Ollama client: {e}")
            self.client = None

    def _verify_connection(self):
        """Verifies that the model is available and the client can connect."""
        try:
            logger.info("Verifying connection to Ollama and model availability...")
            self.client.show(self.model)
            logger.info(f"Successfully connected to Ollama and model '{self.model}' is available.")
        except ollama.ResponseError as e:
            logger.error(f"Model '{self.model}' not found. Please pull it with `ollama pull {self.model}`.")
            logger.error(f"Ollama server response: {e.error}")
            self.client = None
        except Exception as e:
            logger.error(f"Could not connect to Ollama server. Is it running? Error: {e}")
            self.client = None

    def generate_response_stream(self, prompt: str):
        """
        Generates a streaming response from the LLM.
        """
        if not self.client:
            logger.error("Ollama client is not available. Cannot generate response.")
            return

        messages = [{'role': 'user', 'content': prompt}]
        
        for attempt in range(self.retries):
            try:
                logger.info(f"Sending prompt to '{self.model}' (Attempt {attempt + 1}/{self.retries})...")
                stream = self.client.chat(model=self.model, messages=messages, stream=True)
                for chunk in stream:
                    yield chunk['message']['content']
                return
            except Exception as e:
                logger.warning(f"Error during streaming generation (Attempt {attempt + 1}): {e}")
                if attempt < self.retries - 1:
                    logger.info(f"Retrying in {self.delay} seconds...")
                    time.sleep(self.delay)
                else:
                    logger.error("Max retries reached. Failed to generate response.")
                    yield f"Error: Failed to get a response from the model after {self.retries} attempts."
    
    # --- THIS IS THE MISSING METHOD ---
    def generate_response(self, prompt: str) -> str:
        """
        Generates a single, complete response from the LLM (non-streaming).

        Args:
            prompt (str): The user prompt to send to the model.

        Returns:
            str: The full response content.
        """
        if not self.client:
            logger.error("Ollama client is not available. Cannot generate response.")
            return "Error: Client not available."

        messages = [{'role': 'user', 'content': prompt}]
        
        for attempt in range(self.retries):
            try:
                logger.info(f"Sending prompt to '{self.model}' (Attempt {attempt + 1}/{self.retries})...")
                response = self.client.chat(model=self.model, messages=messages)
                return response['message']['content']
            except Exception as e:
                logger.warning(f"Error during generation (Attempt {attempt + 1}): {e}")
                if attempt < self.retries - 1:
                    logger.info(f"Retrying in {self.delay} seconds...")
                    time.sleep(self.delay)
                else:
                    logger.error("Max retries reached. Failed to generate response.")
                    return "Error: Max retries reached."