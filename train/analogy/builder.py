import pickle
from sklearn.neighbors import NearestNeighbors
from src.logger import logger
from . import config

def build_search_index():
    
    logger.info("Loading embeddings to build the search index...")
    try:
        with open(config.EMBEDDINGS_PATH, 'rb') as f:
            embeddings = pickle.load(f)
    except FileNotFoundError:
        logger.error(f"Embeddings file not found at {config.EMBEDDINGS_PATH}. Cannot build index.")
        return None, None

    logger.info(f"Building nearest neighbors index with k={config.N_NEIGHBORS} and metric='{config.METRIC}'...")
    
    nn_model = NearestNeighbors(
        n_neighbors=config.N_NEIGHBORS,
        algorithm='auto',
        metric=config.METRIC
    )
    
    nn_model.fit(embeddings)
    
    logger.info("Search index built successfully.")
    return nn_model, embeddings