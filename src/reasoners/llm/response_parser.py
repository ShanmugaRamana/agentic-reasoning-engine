# src/reasoners/llm/response_parser.py
import re
from src.logger import logger

LLM_CONFIDENCE_SCORE = 0.85 

def parse_llm_response(response_text: str) -> dict | None:
    """
    Parses the raw text response from the LLM to extract the answer,
    the reasoning (solution), and a confidence score.
    """
    try:
        # --- Extract the solution/reasoning from the scratchpad ---
        scratchpad_match = re.search(r'<scratchpad>(.*?)</scratchpad>', response_text, re.DOTALL)
        if scratchpad_match:
            solution = scratchpad_match.group(1).strip()
        else:
            solution = "[No reasoning provided by LLM]"
            logger.warning(f"Could not find <scratchpad> tags in LLM response.")

        # --- Extract the answer number using the robust multi-layered strategy ---
        answer_num = None
        
        # Strategy 1: Look for <answer> tag
        answer_marker = '<answer>'
        start_index = response_text.find(answer_marker)
        if start_index != -1:
            answer_content = response_text[start_index + len(answer_marker):]
            digit_match = re.search(r'\d', answer_content)
            if digit_match:
                num = int(digit_match.group(0))
                if 1 <= num <= 5:
                    answer_num = num

        # Fallback strategies if <answer> tag is missing or empty
        if answer_num is None:
            # Strategy 2: Look for explicit phrases
            match = re.search(r'(?:option|answer is|correct answer is:?)\s*(\d)', response_text, re.IGNORECASE)
            if match:
                num = int(match.group(1))
                if 1 <= num <= 5:
                    answer_num = num
        
        if answer_num is None:
            # Strategy 3: Look for the last digit
            all_digits = re.findall(r'\d', response_text)
            if all_digits:
                num = int(all_digits[-1])
                if 1 <= num <= 5:
                    answer_num = num

        if answer_num:
            logger.info(f"LLM response parsed successfully. Found option: {answer_num}")
            return {'answer': answer_num, 'solution': solution, 'confidence': LLM_CONFIDENCE_SCORE}

        logger.error(f"Failed to parse a valid option number from LLM response: '{response_text}'")
        return None

    except Exception as e:
        logger.error(f"An unexpected error occurred during response parsing: {e}", exc_info=True)
        return None