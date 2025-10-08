# src/reasoners/heuristic_reasoner.py
from src.logger import logger
from .llm.llm_factory import LLMFactory
from .llm.llm_router import LLMRouter
from .llm.response_parser import parse_llm_response
import pandas as pd

class HeuristicReasoner:
    def __init__(self):
        logger.info("Initializing Heuristic Reasoner...")
        try:
            self.llm_factory = LLMFactory()
            self.llm_router = LLMRouter()
            logger.info("Heuristic Reasoner initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize Heuristic Reasoner: {e}")
            self.llm_factory = None
            self.llm_router = None

    def solve(self, row: pd.Series, topic: str) -> dict | None:
        if not self.llm_factory or not self.llm_router:
            logger.error("Heuristic Reasoner is not properly initialized. Cannot solve.")
            return None
        
        prompt = self.llm_router.get_prompt(topic=topic, row=row)
        if not prompt:
            return None
            
        raw_response = self.llm_factory.generate_response(prompt)
        
        return parse_llm_response(raw_response)