# src/data_pipeline/schemas.py

from pydantic import BaseModel, Field, ValidationError
from typing import Optional

class TrainDataRow(BaseModel):
    """Pydantic model for a single row in train.csv."""
    topic: str
    problem_statement: str
    solution: str
    answer_option_1: str
    answer_option_2: str
    answer_option_3: str
    answer_option_4: str
    answer_option_5: Optional[str] = None # Optional field
    correct_option_number: int = Field(..., gt=0, lt=6) # Must be an integer between 1 and 5

class TestDataRow(BaseModel):
    """Pydantic model for a single row in test.csv."""
    topic: str
    problem_statement: str
    answer_option_1: str
    answer_option_2: str
    answer_option_3: str
    answer_option_4: str
    answer_option_5: Optional[str] = None