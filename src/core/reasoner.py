# src/core/reasoner.py
from src.logger import logger
from src.reasoners.symbolic_reasoner import SymbolicReasoner
from src.reasoners.heuristic_reasoner import HeuristicReasoner
import pandas as pd

class Reasoner:
    """
    A wrapper for the main reasoning components of the system.
    """
    def __init__(self):
        try:
            logger.info("Loading reasoning components...")
            self.symbolic_reasoner = SymbolicReasoner()
            self.heuristic_reasoner = HeuristicReasoner() # <-- NEW
            logger.info("✅ Reasoning components loaded successfully.")
        except Exception as e:
            logger.error(f"An error occurred while loading the reasoners: {e}")
            raise

    def solve_symbolically(self, row: pd.Series, topic: str) -> dict | None:
        """
        Attempts to find a solution using the symbolic reasoning path.
        """
        return self.symbolic_reasoner.solve(row, topic)

    def solve_heuristically(self, row: pd.Series, topic: str) -> dict | None:
        """
        Attempts to find a solution using the heuristic (LLM) reasoning path.
        """
        return self.heuristic_reasoner.solve(row, topic)