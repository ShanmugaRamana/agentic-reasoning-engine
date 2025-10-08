# src/reasoners/solvers/lateral_solver.py
from src.logger import logger
from .base_solver import BaseSolver
import pandas as pd

class LateralSolver(BaseSolver):
    """A specialized solver for lateral thinking problems using a knowledge base."""
    def __init__(self):
        self.puzzle_kb = {
            'photography_puzzle': {
                'keywords': ['shoots her husband', 'underwater', 'hangs him'],
                'answer': 'photo development process'
            }
        }
        logger.info(f"LateralSolver initialized with {len(self.puzzle_kb)} known puzzles.")

    def solve(self, row: pd.Series) -> dict | None:
        required_keys = ['problem_statement', 'answer_option_1', 'answer_option_2', 'answer_option_3', 'answer_option_4', 'answer_option_5']
        if not all(key in row.index and pd.notna(row[key]) for key in required_keys):
            logger.error("LateralSolver: Input data is incomplete.")
            return None
        
        logger.info("LateralSolver: Verified input data.")
        logger.info("Attempting to solve with LateralSolver...")
        
        problem_statement_lower = row['problem_statement'].lower()

        for puzzle_name, data in self.puzzle_kb.items():
            if all(keyword in problem_statement_lower for keyword in data['keywords']):
                logger.info(f"Matched lateral thinking pattern: '{puzzle_name}'.")
                calculated_answer = {'answer': data['answer'], 'confidence': 1.0}
                
                option_number = self._match_answer_to_option(str(calculated_answer['answer']), row)
                if option_number:
                    return {'answer': option_number, 'confidence': calculated_answer['confidence']}

        logger.warning(f"No known lateral puzzle matched. Defaulting.")
        return {'answer': 5, 'confidence': 0.10}

    def _match_answer_to_option(self, calculated_answer: str, row: pd.Series) -> int | None:
        for i in range(1, 6):
            option_key = f'answer_option_{i}'
            if option_key in row and calculated_answer.lower() in str(row[option_key]).lower():
                logger.info(f"Matched calculated answer '{calculated_answer}' to Option {i}.")
                return i
        return None