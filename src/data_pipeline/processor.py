# src/data_pipeline/processor.py

import pandas as pd
import json
import os
from src.logger import logger

def process_data(train_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: str):
    """Processes DataFrames and saves them as structured JSON files."""
    if train_df.empty or test_df.empty:
        logger.warning("DataFrames are empty. Skipping JSON processing.")
        return

    logger.info("Processing and converting data to JSON format...")
    
    train_records = train_df.to_dict(orient='records')
    test_records = test_df.to_dict(orient='records')
    
    train_output_path = os.path.join(output_dir, 'train_processed.json')
    test_output_path = os.path.join(output_dir, 'test_processed.json')
    
    try:
        with open(train_output_path, 'w', encoding='utf-8') as f:
            json.dump(train_records, f, ensure_ascii=False, indent=4)
        logger.info(f"Saved processed training data to {train_output_path}")
        
        with open(test_output_path, 'w', encoding='utf-8') as f:
            json.dump(test_records, f, ensure_ascii=False, indent=4)
        logger.info(f"Saved processed testing data to {test_output_path}")
    except IOError as e:
        logger.error(f"Failed to write JSON files to disk: {e}")