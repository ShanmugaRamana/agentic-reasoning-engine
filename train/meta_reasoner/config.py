# train/meta_reasoner/config.py
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from src import config as main_config
from train.classifier.config import MODEL_NAME as CLASSIFIER_MODEL_NAME
from train.analogy.config import MODEL_NAME as ANALOGY_MODEL_NAME

# --- Input Paths ---
PROCESSED_TRAIN_DATA_PATH = os.path.join(main_config.PROCESSED_DATA_DIR, 'train_processed.json')
CLASSIFIER_MODEL_PATH = os.path.join(main_config.MODELS_DIR, CLASSIFIER_MODEL_NAME)
ANALOGY_MODEL_PATH = os.path.join(main_config.MODELS_DIR, ANALOGY_MODEL_NAME)

# --- Output Model Path ---
MODEL_OUTPUT_DIR = main_config.MODELS_DIR
MODEL_NAME = "meta_reasoner.pkl"