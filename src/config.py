# src/config.py
import os

# --- Application ---
VERSION = "1.0.0"

# --- Directory Paths ---
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, 'dataset')
MODELS_DIR = os.path.join(ROOT_DIR, 'models')
OUTPUT_DIR = os.path.join(ROOT_DIR, 'outputs')
OUTPUT_JSON_DIR = os.path.join(OUTPUT_DIR, 'json')
OUTPUT_CSV_DIR = os.path.join(OUTPUT_DIR, 'csv')

# --- Input Data Paths ---
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
TRAIN_CSV_PATH = os.path.join(RAW_DATA_DIR, 'train.csv')
TEST_CSV_PATH = os.path.join(RAW_DATA_DIR, 'test.csv')

# --- Output File Paths ---
OUTPUT_JSON_PATH = os.path.join(OUTPUT_JSON_DIR, 'output.json')
OUTPUT_JSON_CONFIDENCE_PATH = os.path.join(OUTPUT_JSON_DIR, 'output_with_confidence.json')
OUTPUT_CSV_PATH = os.path.join(OUTPUT_CSV_DIR, 'output.csv')
OUTPUT_CSV_CONFIDENCE_PATH = os.path.join(OUTPUT_CSV_DIR, 'output_with_confidence.csv')

# --- Model Configuration ---
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'