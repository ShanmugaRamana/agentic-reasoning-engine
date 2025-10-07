# src/llm/ollama_client.py

import ollama
import time
from src.logger import logger

class OllamaClient:
    
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
        
        if not self.client:
            logger.error("Ollama client is not available. Cannot generate response.")
            return

        messages = [{'role': 'user', 'content': prompt}]
        
        for attempt in range(self.retries):
            try:
                logger.info(f"Sending prompt to '{self.model}' (Attempt {attempt + 1}/{self.retries})...")
                stream = self.client.chat(
                    model=self.model,
                    messages=messages,
                    stream=True
                )
                for chunk in stream:
                    yield chunk['message']['content']
                return # Success, exit the loop
            except Exception as e:
                logger.warning(f"Error during streaming generation (Attempt {attempt + 1}): {e}")
                if attempt < self.retries - 1:
                    logger.info(f"Retrying in {self.delay} seconds...")
                    time.sleep(self.delay)
                else:
                    logger.error("Max retries reached. Failed to generate response.")
                    yield f"Error: Failed to get a response from the model after {self.retries} attempts."


if __name__ == '__main__':
    
    
    logger.info("--- Running OllamaClient Example ---")
    
    llm_client = OllamaClient(model='llama3')

    if llm_client.client:
        example_prompt = "Explain the importance of project structure in a Python application in 3 concise points."
        
        print(f"\\n> PROMPT: {example_prompt}\\n")
        print(f"> RESPONSE from {llm_client.model}:")
        
        full_response = ""
        for chunk in llm_client.generate_response_stream(prompt=example_prompt):
            print(chunk, end='', flush=True)
            full_response += chunk
        print("\\n\\n--- Example Finished ---")
    else:
        logger.error("Could not run example because Ollama client failed to initialize.")