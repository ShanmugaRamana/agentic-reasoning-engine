import pandas as pd
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from src.logger import logger
from . import config

class ProblemTopicDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

def get_dataloaders():
    logger.info("Loading data for classifier training...")
    
    with open(config.EMBEDDINGS_PATH, 'rb') as f:
        embeddings = pickle.load(f)
    
    df = pd.read_json(config.PROCESSED_TRAIN_DATA_PATH)
    
    if len(df) != len(embeddings):
        raise ValueError("Mismatch between number of data points and embeddings.")
    
    le = LabelEncoder()
    labels = le.fit_transform(df['topic'])
    
    X_train, X_val, y_train, y_val = train_test_split(
        embeddings, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    train_dataset = ProblemTopicDataset(X_train, y_train)
    val_dataset = ProblemTopicDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    logger.info(f"Data loading complete. Train size: {len(X_train)}, Val size: {len(X_val)}")
    
    return train_loader, val_loader, le