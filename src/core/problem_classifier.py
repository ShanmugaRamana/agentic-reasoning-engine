# src/core/problem_classifier.py

import joblib
import torch
import pandas as pd
from src.logger import logger
from train.classifier.model import SimpleClassifierNN

class ProblemClassifier:
    """
    A wrapper for the trained problem classifier model.
    Handles loading the model artifact and running inference.
    """
    def __init__(self, model_path: str):
        try:
            logger.info(f"Loading problem classifier artifact from {model_path}...")
            artifact = joblib.load(model_path)
            
            self.label_encoder = artifact['label_encoder']
            input_dim = artifact['input_dim']
            num_classes = artifact['num_classes']
            
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = SimpleClassifierNN(input_dim=input_dim, num_classes=num_classes).to(self.device)
            self.model.load_state_dict(artifact['model_state_dict'])
            self.model.eval()
            
            logger.info("✅ Problem classifier loaded successfully.")
        except FileNotFoundError:
            logger.error(f"Classifier model artifact not found at {model_path}.")
            raise
        except Exception as e:
            logger.error(f"An error occurred while loading the problem classifier: {e}")
            raise

    def predict(self, row: pd.Series, embedding: torch.Tensor, row_index: int) -> str:
        """
        Predicts the topic for a given row. If a topic already exists, it returns
        the existing one. Otherwise, it uses the embedding to predict a new one.

        Args:
            row (pd.Series): The row of data from the DataFrame.
            embedding (torch.Tensor): The pre-computed embedding for the row's problem statement.
            row_index (int): The index of the row for logging purposes.

        Returns:
            str: The existing or predicted topic name.
        """
        # --- UPDATED: Added explicit info-level logging for every check ---
        if 'topic' in row.index and pd.notna(row['topic']):
            logger.info(f"Row {row_index+1}: Topic found. Using existing topic: '{row['topic']}'")
            return row['topic']
        
        logger.info(f"Row {row_index+1}: Topic not found. Running prediction...")
        
        if self.model is None:
            logger.error("Classifier model is not loaded. Cannot predict.")
            return "Error: Model not loaded"
            
        with torch.no_grad():
            embedding_tensor = embedding.to(self.device).reshape(1, -1)
            output = self.model(embedding_tensor)
            _, predicted_idx = torch.max(output.data, 1)
            
        predicted_topic = self.label_encoder.inverse_transform([predicted_idx.item()])
        return predicted_topic[0]