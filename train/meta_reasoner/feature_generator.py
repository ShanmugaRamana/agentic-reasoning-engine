# train/meta_reasoner/feature_generator.py

import joblib
import pickle
import numpy as np
from src.logger import logger
from . import config

def generate_meta_features():
    """
    Generates a 'meta-dataset' for training the meta-reasoner.
    
    It uses the analogical reasoner to create heuristic labels (strategy choice)
    for each problem in the training set.
    
    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple of (features, labels).
                                       Features are the original embeddings.
                                       Labels are the generated strategies ('ANALOGY' or 'LLM').
    """
    logger.info("Generating features and heuristic labels for meta-reasoner...")

    try:
        # 1. Load the trained analogical reasoner
        analogy_artifact = joblib.load(config.ANALOGY_MODEL_PATH)
        search_index = analogy_artifact['search_index']
        
        # 2. Load the embeddings
        with open(config.EMBEDDINGS_PATH, 'rb') as f:
            embeddings = pickle.load(f)
            
    except FileNotFoundError as e:
        logger.error(f"Required model/data not found: {e}. Cannot generate meta-features.")
        return None, None

    # Find the 2 nearest neighbors for each embedding (the first is always itself)
    distances, _ = search_index.kneighbors(embeddings, n_neighbors=2)
    
    # The distance to the *closest other* example is in the second column
    closest_distances = distances[:, 1]
    
    # Cosine similarity = 1 - cosine distance
    similarities = 1 - closest_distances
    
    # 3. Apply the heuristic to generate labels
    # If similarity is high, the best strategy is analogy. Otherwise, use the LLM.
    labels = np.where(similarities > config.SIMILARITY_THRESHOLD, 'ANALOGY', 'LLM')
    
    logger.info(f"Meta-feature generation complete. Found {np.sum(labels == 'ANALOGY')} potential ANALOGY strategies.")
    
    # The features for the meta-reasoner are the embeddings themselves
    features = embeddings
    
    return features, labels