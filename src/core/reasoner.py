# src/core/reasoner.py
from src.logger import logger
from src.reasoners.symbolic_reasoner import SymbolicReasoner
import pandas as pd

class Reasoner:
    """
    A wrapper for the main reasoning components of the system.
    """
    def __init__(self):
        try:
            logger.info("Loading reasoning components...")
            self.symbolic_reasoner = SymbolicReasoner()
            logger.info("✅ Reasoning components loaded successfully.")
        except Exception as e:
            logger.error(f"An error occurred while loading the reasoners: {e}")
            raise

    def solve_symbolically(self, row: pd.Series, topic: str) -> dict | None:
        return self.symbolic_reasoner.solve(row, topic)