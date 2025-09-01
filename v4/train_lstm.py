# train_lstm.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from tqdm import tqdm
from models import BirdCNNLSTM
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
from datetime import datetime
import logging

# Set up logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    logger.addHandler(handler)

# Global lists for plotting metrics (for debugging purposes)
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

class EarlyStopping:
    def __init__(self, patience=10, delta=1e-4):
        self.patience = patience
        self.delta = delta
        self.best_loss = float("inf")
        self.counter = 0
        self.early_stop = False

    def step(self, val_loss):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None, grad_clip=1.0):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for xb, yb in tqdm(loader, desc="Train", leave=False):
        if xb is None:
            continue
        xb, yb = xb.to(device), yb.to(device)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device, enabled=(scaler is not None)):
            outputs = model(xb)
            loss = criterion(outputs, yb)

        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        running_loss += loss.item() * xb.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == yb).sum().item()
        total += yb.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(yb.cpu().numpy())

    avg_loss = running_loss / total
    acc = correct / total
    f1 = f1_score(all_labels, all_preds, average="macro")
    return avg_loss, acc, f1

def validate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for xb, yb in tqdm(loader, desc="Val", leave=False):
            if xb is None:
                continue
            xb, yb = xb.to(device), yb.to(device)
            outputs = model(xb)
            loss = criterion(outputs, yb)

            running_loss += loss.item() * xb.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(yb.cpu().numpy())

    avg_loss = running_loss / total
    acc = correct / total
    f1 = f1_score(all_labels, all_preds, average="macro")
    return avg_loss, acc, f1

def train_lstm(train_loader, val_loader, n_classes, device="cuda", 
               patience=10, lr=1e-3, weight_decay=1e-4, 
               scheduler_type="plateau", max_epochs=50, save_dir="checkpoints_lstm", n_mels=128):

    global train_losses, val_losses, train_accuracies, val_accuracies

    # Log model type and training start
    logger.info(f"Starting CNN+LSTM training with {n_classes} classes")
    logger.info(f"Training parameters: epochs={max_epochs}, lr={lr}, device={device}, n_mels={n_mels}")

    # Generate a timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    model = BirdCNNLSTM(n_classes=n_classes, dropout=0.5, n_mels=n_mels).to(device)
    
    # Calculate class weights
    print("[INFO] Calculating class weights...")
    logger.info("Calculating class weights...")
    full_df = train_loader.dataset.dataset.df # Re-introduce full_df
    all_unique_labels = train_loader.dataset.dataset.unique_labels # Get all unique labels from the full dataset
    train_indices = train_loader.dataset.indices
    train_df = full_df.iloc[train_indices]
    train_labels = train_df['primary_label']
    
    # Compute weights for all possible classes, even if not present in train_labels
    # This will return weights for all classes in all_unique_labels
    class_weights_array = compute_class_weight('balanced', classes=np.array(all_unique_labels), y=train_labels.values)
    
    # Create a mapping from label to its index in all_unique_labels
    label_to_idx = {label: i for i, label in enumerate(all_unique_labels)}
    
    # Initialize a weight tensor of size n_classes with zeros
    class_weights = torch.zeros(n_classes, dtype=torch.float).to(device)
    
    # Populate the class_weights tensor with computed weights
    for label, weight in zip(all_unique_labels, class_weights_array):
        class_weights[label_to_idx[label]] = weight
    
    print("[INFO] Class weights calculated.")
    logger.info("Class weights calculated.")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler(enabled=(device.startswith("cuda")))

    if scheduler_type == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    elif scheduler_type == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    elif scheduler_type == "onecycle":
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(train_loader), epochs=max_epochs)
    else:
        scheduler = None

    early_stopping = EarlyStopping(patience=patience)
    os.makedirs(save_dir, exist_ok=True)

    best_val_loss = float("inf")
    for epoch in range(max_epochs):
        print(f"Epoch {epoch+1}/{max_epochs}")
        logger.info(f"Epoch {epoch+1}/{max_epochs}")

        train_loss, train_acc, train_f1 = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_acc, val_f1 = validate(model, val_loader, criterion, device)

        if scheduler_type == "plateau":
            scheduler.step(val_loss)
        elif scheduler_type in ["cosine", "onecycle"]:
            scheduler.step()

        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f}")
        logger.info(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}")
        logger.info(f"Val Loss:   {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}")

        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(save_dir, f"best_model_{timestamp}_acc{val_acc:.4f}.pth"))
            print(f"Saved best model with accuracy: {val_acc:.4f}")
            logger.info(f"Saved best model with accuracy: {val_acc:.4f}")

        torch.save(model.state_dict(), os.path.join(save_dir, f"last_model_{timestamp}_acc{val_acc:.4f}.pth"))

        early_stopping.step(val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            logger.info("Early stopping triggered.")
            break

    # Plotting
    epochs_range = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, label='Train Loss')
    plt.plot(epochs_range, val_losses, label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_accuracies, label='Train Accuracy')
    plt.plot(epochs_range, val_accuracies, label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"training_metrics_{timestamp}.png"))
    plt.close()

    return model
