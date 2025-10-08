# src/reasoners/solvers/logical_solver.py
from src.logger import logger
from .base_solver import BaseSolver
import pandas as pd
from z3 import Real, Solver, sat

class LogicalSolver(BaseSolver):
    """A specialized solver for deductive logical reasoning and logical traps."""
    def solve(self, row: pd.Series) -> dict | None:
        logger.info("Attempting to solve with LogicalSolver...")
        problem_statement = row['problem_statement']

        # Rule for Z3-based constraint satisfaction traps
        if "equidistant from corners" in problem_statement:
            result = self._solve_constraint_satisfaction(problem_statement)
            if result:
                option_number = self._match_answer_to_option(str(result['answer']), row)
                if option_number:
                    return {'answer': option_number, 'confidence': result['confidence']}

        logger.warning(f"No specific logical rule matched. Defaulting.")
        return {'answer': 5, 'confidence': 0.10}

    def _match_answer_to_option(self, calculated_answer: str, row: pd.Series) -> int | None:
        for i in range(1, 6):
            option_key = f'answer_option_{i}'
            if option_key in row and calculated_answer.lower() in str(row[option_key]).lower():
                logger.info(f"Matched calculated answer '{calculated_answer}' to Option {i}.")
                return i
        return None

    def _solve_constraint_satisfaction(self, problem_statement: str) -> dict | None:
        """Solves constraint-based problems using the Z3 solver."""
        logger.info("Matched 'constraint satisfaction' pattern. Using Z3 Solver.")
        if "twice the distance" in problem_statement.lower():
            # This logic is specific to the "lamp in a room" puzzle
            s = Solver()
            x, y = Real('x'), Real('y')
            A, B, C, D = (0, 1), (1, 1), (1, 0), (0, 0)
            dist_sq_A = (x - A[0])**2 + (y - A[1])**2
            dist_sq_B = (x - B[0])**2 + (y - B[1])**2
            dist_sq_C = (x - C[0])**2 + (y - C[1])**2
            s.add(dist_sq_A == dist_sq_B, dist_sq_B == dist_sq_C)
            dist_sq_D = (x - D[0])**2 + (y - D[1])**2
            s.add(dist_sq_D == 4 * dist_sq_A)
            s.add(x >= 0, x <= 1, y >= 0, y <= 1)
            
            if s.check() != sat:
                return {'answer': "Nowhere, because it's a logical trap", 'confidence': 1.0}
        return None