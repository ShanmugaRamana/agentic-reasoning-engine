# src/reasoners/solvers/base_solver.py
from abc import ABC, abstractmethod
import pandas as pd

class BaseSolver(ABC):
    """Abstract base class for all specialized solvers."""
    @abstractmethod
    def solve(self, row: pd.Series) -> dict | None:
        """
        Attempts to solve a problem given the entire data row.
        Returns a dictionary {'answer': int, 'confidence': float} or None.
        """
        pass