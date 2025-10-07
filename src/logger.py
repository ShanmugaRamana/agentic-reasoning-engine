# src/logger.py

import logging
import sys
from logging.handlers import TimedRotatingFileHandler
import os

# --- Configuration ---
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "pipeline.log")
FORMATTER = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
)

# --- Setup ---
# Create logs directory if it doesn't exist
os.makedirs(LOG_DIR, exist_ok=True)

def get_console_handler():
    """Returns a console handler for logging."""
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(FORMATTER)
    return console_handler

def get_file_handler():
    """Returns a file handler that rotates logs daily."""
    # Rotates log file every day, keeps 7 days of backup logs
    file_handler = TimedRotatingFileHandler(LOG_FILE, when='midnight', backupCount=7)
    file_handler.setFormatter(FORMATTER)
    return file_handler

def get_logger(logger_name):
    """Configures and returns a logger instance."""
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)  # Set the minimum level of logs to capture
    
    # Add handlers only if they haven't been added before
    if not logger.handlers:
        logger.addHandler(get_console_handler())
        logger.addHandler(get_file_handler())
        
    # Propagate messages to the root logger
    logger.propagate = False
    
    return logger

# Create a default logger for easy import
logger = get_logger("AgenticReasoningPipeline")