import os
import joblib
import pandas as pd
from src.logger import logger
from . import config, builder

def run_training():
    
    logger.info("--- Starting Analogical Reasoner Training ---")
    
    search_index, embeddings = builder.build_search_index()
    
    if search_index is None:
        logger.error("Training stopped because search index could not be built.")
        return

    
    df = pd.read_json(config.PROCESSED_TRAIN_DATA_PATH)
    index_to_data_map = df[['problem_statement', 'solution']].to_dict(orient='records')
    
    artifact = {
        'search_index': search_index,
        'index_to_data_map': index_to_data_map,
        'embeddings': embeddings # Include embeddings for potential future use
    }

    os.makedirs(config.MODEL_OUTPUT_DIR, exist_ok=True)
    model_path = os.path.join(config.MODEL_OUTPUT_DIR, config.MODEL_NAME)
    
    joblib.dump(artifact, model_path)
    logger.info(f"Analogical Reasoner artifact saved successfully to {model_path}")
    logger.info("--- Training Finished ---")

if __name__ == '__main__':
    run_training()