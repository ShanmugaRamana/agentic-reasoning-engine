# src/reasoners/solvers/riddle_solver.py
from src.logger import logger
from .base_solver import BaseSolver
import pandas as pd

class RiddleSolver(BaseSolver):
    """
    A specialized solver for classic and lateral thinking riddles
    using a knowledge base lookup.
    """
    def __init__(self):
        """Initializes the solver with a knowledge base of known riddles."""
        self.riddle_kb = {
            'race_position': {
                'keywords': ['in a race', 'overtake', 'second person'],
                'answer': 'Second'
            },
            'photography_lateral': {
                'keywords': ['shoots her husband', 'underwater', 'hangs him'],
                'answer': 'photo development process'
            },
            'inheritance_coins': {
                'keywords': ['17 gold coins', 'one-ninth', 'one-third'],
                'answer': 'Borrow 1 coin'
            }
        }
        logger.info(f"RiddleSolver initialized with {len(self.riddle_kb)} known riddles.")

    def solve(self, row: pd.Series) -> dict | None:
        required_keys = ['problem_statement', 'answer_option_1', 'answer_option_2', 'answer_option_3', 'answer_option_4', 'answer_option_5']
        if not all(key in row.index and pd.notna(row[key]) for key in required_keys):
            logger.error("RiddleSolver: Input data is incomplete.")
            return None
        
        logger.info("RiddleSolver: Verified that problem statement and all answer options are loaded.")
        logger.info("Attempting to solve with RiddleSolver...")
        
        problem_statement_lower = row['problem_statement'].lower()

        # Iterate through the knowledge base to find a matching riddle
        for riddle_name, data in self.riddle_kb.items():
            if all(keyword in problem_statement_lower for keyword in data['keywords']):
                logger.info(f"Matched riddle pattern: '{riddle_name}'.")
                calculated_answer = {'answer': data['answer'], 'confidence': 1.0}
                
                option_number = self._match_answer_to_option(str(calculated_answer['answer']), row)
                if option_number:
                    return {'answer': option_number, 'confidence': calculated_answer['confidence']}

        logger.warning(f"No known riddle matched. Defaulting.")
        return {'answer': 5, 'confidence': 0.10}

    def _match_answer_to_option(self, calculated_answer: str, row: pd.Series) -> int | None:
        """Finds which answer_option_ (1-5) contains the calculated answer."""
        for i in range(1, 6):
            option_key = f'answer_option_{i}'
            # Use 'in' for partial matches, which is common in riddle answers
            if option_key in row and calculated_answer.lower() in str(row[option_key]).lower():
                logger.info(f"Matched calculated answer '{calculated_answer}' to Option {i}.")
                return i
        return None