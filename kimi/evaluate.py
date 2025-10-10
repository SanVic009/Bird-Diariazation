# evaluate.py
from config import *
from model import CNNEncoder
from dataset import BirdCLEFSupervised  # Need to create this
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import numpy as np

class BirdClassifier(nn.Module):
    def __init__(self, encoder, num_classes):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(EMBED_DIM, num_classes)
    
    def forward(self, x):
        with torch.no_grad():
            embeddings = self.encoder(x)
        return self.classifier(embeddings)

def evaluate():
    # Load trained encoder
    encoder = CNNEncoder(EMBED_DIM, N_MELS).to(DEVICE)
    encoder.load_state_dict(torch.load("ckpt/birdclef24_encoder.pt"))
    encoder.eval()
    
    # Create supervised dataset (with labels)
    train_ds = BirdCLEFSupervised("data", dur=DUR, sr=SR, split="train")
    test_ds = BirdCLEFSupervised("data", dur=DUR, sr=SR, split="test") 
    
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    num_classes = len(train_ds.classes)
    classifier = BirdClassifier(encoder, num_classes).to(DEVICE)
    opt = torch.optim.Adam(classifier.classifier.parameters(), lr=1e-3)
    
    # Train classifier
    for epoch in range(10):
        classifier.train()
        for x, y in train_dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = classifier(x)
            loss = nn.CrossEntropyLoss()(pred, y)
            opt.zero_grad(); loss.backward(); opt.step()
    
    # Evaluate
    classifier.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in test_dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = classifier(x).argmax(1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Classification Accuracy: {accuracy:.4f}")
    return accuracy

if __name__ == "__main__":
    evaluate()