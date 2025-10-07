# main.py

from scripts.process_data import run_pipeline
from scripts.train_components import train_classifier_if_needed

if __name__ == "__main__":
   
    run_pipeline()
    train_classifier_if_needed()