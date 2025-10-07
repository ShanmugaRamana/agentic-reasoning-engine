import sys
import os

N_NEIGHBORS = 5  
METRIC = 'cosine' 

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from src import config as main_config

PROCESSED_TRAIN_DATA_PATH = os.path.join(main_config.PROCESSED_DATA_DIR, 'train_processed.json')
EMBEDDINGS_PATH = os.path.join(main_config.PROCESSED_DATA_DIR, 'problem_embeddings.pkl')

MODEL_OUTPUT_DIR = main_config.MODELS_DIR
MODEL_NAME = "analogical_reasoner.pkl"