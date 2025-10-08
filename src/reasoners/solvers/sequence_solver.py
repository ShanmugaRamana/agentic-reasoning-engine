# src/reasoners/solvers/sequence_solver.py
from src.logger import logger
from .base_solver import BaseSolver
import pandas as pd
import re
import numpy as np

class SequenceSolver(BaseSolver):
    """
    A specialized solver for numerical sequence problems using polynomial fitting.
    """
    def solve(self, row: pd.Series) -> dict | None:
        required_keys = ['problem_statement', 'answer_option_1', 'answer_option_2', 'answer_option_3', 'answer_option_4', 'answer_option_5']
        if not all(key in row.index and pd.notna(row[key]) for key in required_keys):
            logger.error("SequenceSolver: Input data is incomplete.")
            return None
        
        logger.info("SequenceSolver: Verified that problem statement and all answer options are loaded.")
        logger.info("Attempting to solve with SequenceSolver...")
        
        calculated_answer = self._solve_polynomial_sequence(row['problem_statement'])

        if calculated_answer:
            # The sequence "37 and 50" requires special handling for matching
            # We check if both numbers are in the option
            if ' and ' in calculated_answer['answer']:
                nums = calculated_answer['answer'].split(' and ')
                option_number = self._match_multiple_answers(nums, row)
            else:
                option_number = self._match_answer_to_option(str(calculated_answer['answer']), row)
            
            if option_number:
                return {'answer': option_number, 'confidence': calculated_answer['confidence']}

        logger.warning(f"No specific sequence rule matched. Defaulting.")
        return {'answer': 5, 'confidence': 0.10}

    def _match_answer_to_option(self, calculated_answer: str, row: pd.Series) -> int | None:
        for i in range(1, 6):
            option_key = f'answer_option_{i}'
            if option_key in row and calculated_answer.lower() in str(row[option_key]).lower():
                logger.info(f"Matched calculated answer '{calculated_answer}' to Option {i}.")
                return i
        return None
        
    def _match_multiple_answers(self, calculated_answers: list, row: pd.Series) -> int | None:
        """Special matching for answers with multiple numbers."""
        for i in range(1, 6):
            option_key = f'answer_option_{i}'
            option_text = str(row.get(option_key, '')).lower()
            if all(num in option_text for num in calculated_answers):
                logger.info(f"Matched multiple calculated answers '{calculated_answers}' to Option {i}.")
                return i
        return None

    def _solve_polynomial_sequence(self, problem_statement: str) -> dict | None:
        """
        Solves sequences by finding the polynomial that generates them.
        """
        if "sequence of numbers" not in problem_statement and "sequence:" not in problem_statement:
            return None

        logger.info("Matched 'polynomial sequence' pattern.")
        
        try:
            # Extract all numbers, including potential negative ones
            sequence_str = re.findall(r'-?\d+', problem_statement)
            sequence = [int(s) for s in sequence_str]
            
            if len(sequence) < 3: # Need at least 3 points to detect a pattern
                return None

            # Determine the order of the polynomial by finding constant finite differences
            diffs = np.array(sequence)
            degree = 0
            for i in range(1, len(sequence)):
                diffs = np.diff(diffs)
                degree = i
                if len(set(diffs)) == 1: # Found a constant difference
                    break
            
            # Fit the polynomial
            x = np.arange(1, len(sequence) + 1)
            coeffs = np.polyfit(x, sequence, deg=degree)
            poly = np.poly1d(coeffs)
            
            # Predict the next term(s)
            next_x = len(sequence) + 1
            next_val = int(round(poly(next_x)))

            # Handle cases that ask for two numbers
            if "next two numbers" in problem_statement:
                next_x2 = len(sequence) + 2
                next_val2 = int(round(poly(next_x2)))
                answer = f"{next_val} and {next_val2}"
            else:
                answer = str(next_val)

            return {'answer': answer, 'confidence': 1.0}
            
        except Exception as e:
            logger.error(f"Error during polynomial sequence solving: {e}")
            return None