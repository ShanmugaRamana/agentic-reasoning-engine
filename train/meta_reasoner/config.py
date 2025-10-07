# train/meta_reasoner/config.py

import sys
import os

# --- Heuristic & Model Hyperparameters ---
# If the top analogy has a similarity > this threshold, we label it as an 'ANALOGY' strategy.
SIMILARITY_THRESHOLD = 0.90 

# --- File Paths & Names ---
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from src import config as main_config
from train.analogy.config import MODEL_NAME as ANALOGY_MODEL_NAME

# Input data paths
EMBEDDINGS_PATH = os.path.join(main_config.PROCESSED_DATA_DIR, 'problem_embeddings.pkl')
ANALOGY_MODEL_PATH = os.path.join(main_config.MODELS_DIR, ANALOGY_MODEL_NAME)

# Output model path
MODEL_OUTPUT_DIR = main_config.MODELS_DIR
MODEL_NAME = "meta_reasoner.pkl"