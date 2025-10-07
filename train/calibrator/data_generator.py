# train/calibrator/data_generator.py

import torch
import joblib
import numpy as np
from src.logger import logger
from train.classifier import data_loader, model as model_def
from . import config

def generate_calibration_data():
    """
    Generates a 'meta-dataset' for training the confidence calibrator.
    
    It runs the trained problem_classifier on its validation set and records
    the confidence of each prediction and whether the prediction was correct.
    
    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple of (confidences, is_correct_labels).
    """
    logger.info("Generating data for confidence calibrator...")
    
    try:
        # 1. Load the trained classifier model
        artifact = joblib.load(config.CLASSIFIER_MODEL_PATH)
        classifier_model = model_def.SimpleClassifierNN(
            input_dim=artifact['input_dim'], 
            num_classes=artifact['num_classes']
        )
        classifier_model.load_state_dict(artifact['model_state_dict'])
        classifier_model.eval()
        
        # 2. Get the validation data
        _, val_loader, _ = data_loader.get_dataloaders()
        
    except FileNotFoundError:
        logger.error(f"Classifier model not found at {config.CLASSIFIER_MODEL_PATH}. Cannot generate calibration data.")
        return None, None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    classifier_model.to(device)
    
    all_confidences = []
    all_correct_flags = []
    
    with torch.no_grad():
        for embeddings, labels in val_loader:
            embeddings, labels = embeddings.to(device), labels.to(device)
            
            # Get model outputs (logits)
            outputs = classifier_model(embeddings)
            
            # Convert to probabilities (confidence scores)
            probabilities = torch.softmax(outputs, dim=1)
            
            # Get the confidence of the predicted class (the max probability)
            confidences, predictions = torch.max(probabilities, 1)
            
            # Check if predictions were correct (1 if correct, 0 if incorrect)
            is_correct = (predictions == labels).cpu().numpy()
            
            all_confidences.extend(confidences.cpu().numpy())
            all_correct_flags.extend(is_correct)
            
    logger.info("Calibration data generated successfully.")
    return np.array(all_confidences), np.array(all_correct_flags)