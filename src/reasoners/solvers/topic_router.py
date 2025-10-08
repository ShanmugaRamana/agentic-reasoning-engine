# src/reasoners/solvers/topic_router.py
from src.logger import logger
import pandas as pd
from .spatial_solver import SpatialSolver
from .optimization_solver import OptimizationSolver
from .mechanism_solver import MechanismSolver
from .riddle_solver import RiddleSolver
from .sequence_solver import SequenceSolver
from .logical_solver import LogicalSolver
from .lateral_solver import LateralSolver

class TopicRouter:
    """
    Routes a problem to a specialized solver based on its topic.
    """
    def __init__(self):
        self.solvers = {
            'spatial reasoning': SpatialSolver(),
            'optimization of actions and planning': OptimizationSolver(),
            'operation of mechanisms': MechanismSolver(),
            'classic riddles': RiddleSolver(),
            'sequence solving': SequenceSolver(),
            'lateral thinking': LateralSolver(),
            'logical traps': LogicalSolver() # <-- NEW ROUTING RULE

        }
        logger.info(f"TopicRouter initialized with {len(self.solvers)} specialized solvers.")

    def route(self, row: pd.Series, topic: str) -> dict | None:
        """
        Finds the appropriate solver for the topic and calls its solve() method.

        Args:
            row (pd.Series): The entire data row for the problem.
            topic (str): The problem's classified topic.

        Returns:
            dict | None: The answer from the solver, or None if no solver is found or fails.
        """
        solver = self.solvers.get(topic.lower())
        
        if solver:
            logger.info(f"Routing problem to '{solver.__class__.__name__}'.")
            return solver.solve(row)
        else:
            logger.warning(f"No specialized solver found for topic: '{topic}'.")
            return None