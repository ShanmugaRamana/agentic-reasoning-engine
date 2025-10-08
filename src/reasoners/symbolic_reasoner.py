# src/reasoners/symbolic_reasoner.py
from src.logger import logger
from .solvers.topic_router import TopicRouter
import pandas as pd

class SymbolicReasoner:
    # ... (__init__ is the same) ...
    def __init__(self):
        logger.info("Initializing Symbolic Reasoner...")
        self.router = TopicRouter()

    def solve(self, row: pd.Series, topic: str) -> dict | None:
        return self.router.route(row, topic)