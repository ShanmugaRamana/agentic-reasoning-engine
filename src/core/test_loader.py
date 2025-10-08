import os
import pandas as pd
import json
from src.logger import logger
from src import config as main_config
from src.data_pipeline.schemas import TestDataRow, ValidationError

def load_test_data():
    
    test_data_path = os.path.join(main_config.PROCESSED_DATA_DIR, 'test_processed.json')
    logger.info(f"Loading and validating test data from {test_data_path}...")
    
    try:
        with open(test_data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        logger.error(f"Test data file not found at {test_data_path}.")
        return pd.DataFrame() # Return empty dataframe on error
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from {test_data_path}. The file may be corrupt.")
        return pd.DataFrame()

    valid_records = []
    invalid_count = 0
    for i, record in enumerate(data):
        try:
            # Validate each record using the Pydantic model
            validated_record = TestDataRow(**record)
            valid_records.append(validated_record.model_dump())
        except ValidationError as e:
            # Log any data quality issues
            logger.warning(f"Skipping invalid record #{i+1} in test data. Reason: {e}")
            invalid_count += 1
            
    if invalid_count > 0:
        logger.warning(f"Found and skipped a total of {invalid_count} invalid records.")

    df = pd.DataFrame(valid_records)
    
    # Display the final count of problems loaded
    logger.info(f"Successfully loaded and validated {len(df)} test records.")
    
    return df