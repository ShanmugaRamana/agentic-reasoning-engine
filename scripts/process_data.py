import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.data_pipeline import load_data, process_data, create_embeddings
from src import config
from src.logger import logger

def run_pipeline():
    
    logger.info("Starting the data processing pipeline script...")
    
    train_json_path = os.path.join(config.PROCESSED_DATA_DIR, 'train_processed.json')
    test_json_path = os.path.join(config.PROCESSED_DATA_DIR, 'test_processed.json')
    embeddings_path = os.path.join(config.PROCESSED_DATA_DIR, 'problem_embeddings.pkl')

    if all(os.path.exists(p) for p in [train_json_path, test_json_path, embeddings_path]):
        logger.info("Processed data and embeddings already exist. Skipping pipeline run.")
        logger.info("Pipeline script finished successfully!")
        return
    
    logger.info("Processed data not found. Running the full data pipeline...")
    
    os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)
    
    train_df, test_df = load_data(
        train_path=config.TRAIN_CSV_PATH, 
        test_path=config.TEST_CSV_PATH
    )
    
    if train_df.empty or test_df.empty:
        logger.error("Pipeline stopped due to data loading errors.")
        return
        
    process_data(train_df, test_df, output_dir=config.PROCESSED_DATA_DIR)
    
    create_embeddings(
        train_df, 
        output_dir=config.PROCESSED_DATA_DIR,
        model_name=config.EMBEDDING_MODEL_NAME
    )
    
    logger.info("🎉 Pipeline script finished successfully!")

if __name__ == "__main__":
    run_pipeline()