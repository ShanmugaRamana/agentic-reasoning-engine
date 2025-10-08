# src/reasoners/llm/llm_router.py
from src.logger import logger
from .prompt_templates import PROMPT_TEMPLATES
import pandas as pd

class LLMRouter:
    """
    Selects and formats the appropriate prompt template for a given problem.
    """
    def get_prompt(self, topic: str, row: pd.Series) -> str:
        """
        Retrieves the best prompt template for the topic and formats it.

        Args:
            topic (str): The classified topic of the problem.
            row (pd.Series): The data row containing the problem statement and options.

        Returns:
            str: The fully formatted prompt ready to be sent to the LLM.
        """
        # Select the template for the given topic, or fall back to the base template
        template = PROMPT_TEMPLATES.get(topic.lower(), PROMPT_TEMPLATES["base"])
        logger.info(f"Selected '{topic.lower()}' prompt template.")

        try:
            # Prepare a dictionary with all the data needed by the template
            prompt_data = {
                "problem_statement": row["problem_statement"],
                "answer_option_1": row["answer_option_1"],
                "answer_option_2": row["answer_option_2"],
                "answer_option_3": row["answer_option_3"],
                "answer_option_4": row["answer_option_4"],
                "answer_option_5": row["answer_option_5"]
            }
            return template.format(**prompt_data)
        except KeyError as e:
            logger.error(f"Failed to format prompt. Data key missing: {e}")
            return "" # Return empty string on formatting failure