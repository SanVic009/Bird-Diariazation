# train_mobilenet.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, confusion_matrix
from tqdm import tqdm
from mobilenet import MobileNetBird
import numpy as np
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

def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None, grad_clip=1.0, multi_label=False, threshold=0.5):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for xb, yb in tqdm(loader, desc="Train", leave=False):
        if xb is None:
            continue
        xb, yb = xb.to(device), yb.to(device)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device, enabled=(scaler is not None)):
            # Always get raw logits for loss calculation
            model.multi_label = False  # Get raw logits
            outputs = model(xb)
            
            if multi_label:
                loss = criterion(outputs, yb.float())  # BCEWithLogitsLoss expects float targets
            else:
                # For single-class, convert to multi-hot format for BCEWithLogitsLoss
                yb_onehot = torch.zeros(yb.size(0), outputs.size(1), device=device)
                yb_onehot.scatter_(1, yb.unsqueeze(1), 1.0)
                loss = criterion(outputs, yb_onehot)
                
            # Always get predictions using sigmoid + threshold
            probs = torch.sigmoid(outputs)
            preds = (probs > threshold).float()
            
            if multi_label:
                # For multi-label accuracy: exact match (all labels must be correct)
                target = yb.float()
                correct += (preds == target).all(dim=1).sum().item()
            else:
                # For single-class: convert target to multi-hot for comparison
                target_onehot = torch.zeros(yb.size(0), outputs.size(1), device=device)
                target_onehot.scatter_(1, yb.unsqueeze(1), 1.0)
                correct += (preds == target_onehot).all(dim=1).sum().item()

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
        total += yb.size(0)

        # Store the appropriate target format
        if multi_label:
            target_for_storage = yb.float()
        else:
            target_for_storage = torch.zeros(yb.size(0), outputs.size(1), device=device)
            target_for_storage.scatter_(1, yb.unsqueeze(1), 1.0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(target_for_storage.cpu().numpy())

    avg_loss = running_loss / total
    acc = correct / total
    
    # Since predictions are now always multi-hot arrays, use appropriate F1 calculation
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        
    return avg_loss, acc, f1

def validate(model, loader, criterion, device, multi_label=False, threshold=0.5):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for xb, yb in tqdm(loader, desc="Val", leave=False):
            if xb is None:
                continue
            xb, yb = xb.to(device), yb.to(device)
            
            # Always get raw logits for loss calculation
            model.multi_label = False  # Get raw logits
            logits = model(xb)
            
            if multi_label:
                loss = criterion(logits, yb.float())  # BCEWithLogitsLoss expects float targets
                target = yb.float()
            else:
                # For single-class, convert to multi-hot format for BCEWithLogitsLoss
                yb_onehot = torch.zeros(yb.size(0), logits.size(1), device=device)
                yb_onehot.scatter_(1, yb.unsqueeze(1), 1.0)
                loss = criterion(logits, yb_onehot)
                target = yb_onehot
                
            # Always get predictions using sigmoid + threshold
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).float()
            
            # For accuracy: exact match (all labels must be correct)
            correct += (preds == target).all(dim=1).sum().item()
                
            running_loss += loss.item() * xb.size(0)
            total += yb.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(target.cpu().numpy())

    avg_loss = running_loss / total
    acc = correct / total
    
    # Since predictions are now always multi-hot arrays, use appropriate F1 calculation
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        
    return avg_loss, acc, f1, all_labels, all_preds

def train_mobilenet(train_loader, val_loader, n_classes, device="cuda",
               patience=10, lr=1e-3, weight_decay=1e-4,
               scheduler_type="cosine", max_epochs=50, save_dir="checkpoints_mobilenet", multi_label=False):

    global train_losses, val_losses, train_accuracies, val_accuracies

    # Log model type and training start
    mode = "multi-label" if multi_label else "multi-class"
    logger.info(f"Starting MobileNet training ({mode}) with {n_classes} classes")
    logger.info(f"Training parameters: epochs={max_epochs}, lr={lr}, device={device}")

    # Generate a timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    model = MobileNetBird(n_classes=n_classes, multi_label=multi_label).to(device)

    # Use BCEWithLogitsLoss for all MobileNet training
    criterion = nn.BCEWithLogitsLoss()
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

        train_loss, train_acc, train_f1 = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler, multi_label=multi_label)
        val_loss, val_acc, val_f1, all_labels, all_preds = validate(model, val_loader, criterion, device, multi_label)

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

    # Confusion Matrix
    if not multi_label:
        logger.info("Generating confusion matrix for single-label classification.")
        # Convert multi-hot encoded labels/preds to class indices for confusion matrix
        true_labels_cm = np.argmax(all_labels, axis=1)
        pred_labels_cm = np.argmax(all_preds, axis=1)
        cm = confusion_matrix(true_labels_cm, pred_labels_cm)
        np.save(os.path.join(save_dir, f"confusion_matrix_{timestamp}.npy"), cm)
    else:
        logger.info("Skipping confusion matrix generation for multi-label classification.")
        np.save(os.path.join(save_dir, f"multilabel_gts_{timestamp}.npy"), all_labels)
        np.save(os.path.join(save_dir, f"multilabel_preds_{timestamp}.npy"), all_preds)


    return model
