# src/config.py
import os

# Define the absolute path to the project's root directory
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- Directory Paths ---
DATA_DIR = os.path.join(ROOT_DIR, 'dataset')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')

# --- File Paths ---
TRAIN_CSV_PATH = os.path.join(RAW_DATA_DIR, 'train.csv')
TEST_CSV_PATH = os.path.join(RAW_DATA_DIR, 'test.csv')

# --- Model Configuration ---
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'