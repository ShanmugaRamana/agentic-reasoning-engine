import torch
import torch.nn as nn
import torch.optim as optim
import joblib
import os
from src.logger import logger
from . import config, data_loader, model

def run_training():
    """Executes the full training and saving pipeline."""
    logger.info("--- Starting Problem Classifier Training ---")
    
    train_loader, val_loader, label_encoder = data_loader.get_dataloaders()
    num_classes = len(label_encoder.classes_)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    classifier_model = model.SimpleClassifierNN(
        input_dim=config.INPUT_DIM, 
        num_classes=num_classes
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier_model.parameters(), lr=config.LEARNING_RATE)
    
    for epoch in range(config.EPOCHS):
        classifier_model.train()
        total_loss = 0
        for embeddings, labels in train_loader:
            embeddings, labels = embeddings.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = classifier_model(embeddings)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)
        
        classifier_model.eval()
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for embeddings, labels in val_loader:
                embeddings, labels = embeddings.to(device), labels.to(device)
                outputs = classifier_model(embeddings)
                _, predicted = torch.max(outputs.data, 1)
                total_samples += labels.size(0)
                total_correct += (predicted == labels).sum().item()
        
        accuracy = (total_correct / total_samples) * 100
        logger.info(f"Epoch {epoch+1}/{config.EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Accuracy: {accuracy:.2f}%")

    os.makedirs(config.MODEL_OUTPUT_DIR, exist_ok=True)
    model_path = os.path.join(config.MODEL_OUTPUT_DIR, config.MODEL_NAME)
    
    artifact = {
        'model_state_dict': classifier_model.state_dict(),
        'label_encoder': label_encoder,
        'input_dim': config.INPUT_DIM,
        'num_classes': num_classes
    }
    
    joblib.dump(artifact, model_path)
    logger.info(f"✅ Model artifact saved successfully to {model_path}")
    logger.info("--- Training Finished ---")

if __name__ == '__main__':
    run_training()