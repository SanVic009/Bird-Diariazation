# supervised_train.py
from config import *
from dataset import BirdCLEFSupervised  # Need labeled version
from model import CNNEncoder
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

class SupervisedBirdModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.encoder = CNNEncoder(EMBED_DIM, N_MELS)
        self.classifier = nn.Linear(EMBED_DIM, num_classes)
    
    def forward(self, x):
        embeddings = self.encoder(x)
        return self.classifier(embeddings)

def train_supervised():
    # Create labeled dataset
    train_ds = BirdCLEFSupervised("data", dur=DUR, sr=SR, split="train")
    val_ds = BirdCLEFSupervised("data", dur=DUR, sr=SR, split="val")
    
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    model = SupervisedBirdModel(len(train_ds.classes)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(EPOCHS):
        # Training
        model.train()
        train_loss = 0
        for x, y in train_dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(DEVICE), y.to(DEVICE)
                pred = model(x).argmax(1)
                val_preds.extend(pred.cpu())
                val_labels.extend(y.cpu())
        
        accuracy = accuracy_score(val_labels, val_preds)
        print(f"Epoch {epoch+1} | Loss: {train_loss/len(train_dl):.4f} | Val Acc: {accuracy:.4f}")
    
    return model

if __name__ == "__main__":
    model = train_supervised()