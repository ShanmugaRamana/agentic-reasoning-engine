# src/data_pipeline/embedder.py

import pandas as pd
import pickle
import os
from sentence_transformers import SentenceTransformer
from src.logger import logger

def create_embeddings(train_df: pd.DataFrame, output_dir: str, model_name: str):
    """Generates and saves sentence embeddings for the 'problem_statement' column."""
    if train_df.empty:
        logger.warning("Training DataFrame is empty. Skipping embedding creation.")
        return
    
    try:
        logger.info(f"Initializing embedding model '{model_name}'...")
        model = SentenceTransformer(model_name)
    except Exception as e:
        logger.error(f"Failed to load SentenceTransformer model: {e}")
        return

    logger.info("Generating embeddings for problem statements...")
    problem_statements = train_df['problem_statement'].tolist()
    embeddings = model.encode(problem_statements, show_progress_bar=True)
    
    output_path = os.path.join(output_dir, 'problem_embeddings.pkl')
    
    try:
        with open(output_path, 'wb') as f:
            pickle.dump(embeddings, f)
        logger.info(f"Embeddings saved successfully to {output_path}")
    except IOError as e:
        logger.error(f"Failed to save embeddings to disk: {e}")