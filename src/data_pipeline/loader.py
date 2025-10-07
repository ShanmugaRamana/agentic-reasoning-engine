# src/data_pipeline/loader.py

import pandas as pd
from src.logger import logger
from .schemas import TrainDataRow, TestDataRow, ValidationError

def load_data(train_path: str, test_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads and validates raw data from CSV files.

    Rows that fail validation are logged and skipped.
    """
    logger.info(f"Attempting to load data from {train_path} and {test_path}")

    try:
        # Load raw dataframes first
        raw_train_df = pd.read_csv(train_path)
        raw_test_df = pd.read_csv(test_path)
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}. Cannot proceed.")
        return pd.DataFrame(), pd.DataFrame()
    except pd.errors.ParserError as e:
        logger.error(f"Error parsing CSV file: {e}. Check file for corruption.")
        return pd.DataFrame(), pd.DataFrame()

    # Validate training data
    valid_train_rows = []
    for index, row in raw_train_df.iterrows():
        try:
            valid_train_rows.append(TrainDataRow(**row.to_dict()).model_dump())
        except ValidationError as e:
            logger.warning(f"Skipping invalid row {index+2} in train.csv: {e}")
    
    # Validate testing data
    valid_test_rows = []
    for index, row in raw_test_df.iterrows():
        try:
            valid_test_rows.append(TestDataRow(**row.to_dict()).model_dump())
        except ValidationError as e:
            logger.warning(f"Skipping invalid row {index+2} in test.csv: {e}")

    train_df = pd.DataFrame(valid_train_rows)
    test_df = pd.DataFrame(valid_test_rows)
    
    logger.info(f"Data loading complete. Valid train rows: {len(train_df)}, Valid test rows: {len(test_df)}")
    return train_df, test_df