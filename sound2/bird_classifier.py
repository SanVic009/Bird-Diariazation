#!/usr/bin/env python3
import os
import sys
import warnings
import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from tqdm import tqdm
import logging
import random
import datetime

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bird_classifier.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ---------------------- AUGMENTATION HELPERS ----------------------
def add_noise(x, noise_factor=0.005):
    return x + noise_factor * np.random.randn(*x.shape)

def time_shift(x, shift_max=0.2):
    shift = int(np.random.uniform(-shift_max, shift_max) * x.shape[-1])
    return np.roll(x, shift, axis=-1)

def spec_augment(spec, num_mask=2, freq_mask=10, time_mask=20):
    spec = spec.copy()
    num_mel_channels, num_time_steps = spec.shape
    for _ in range(num_mask):
        f = np.random.randint(0, freq_mask)
        f0 = np.random.randint(0, num_mel_channels - f)
        spec[f0:f0+f, :] = 0
        t = np.random.randint(0, time_mask)
        t0 = np.random.randint(0, num_time_steps - t)
        spec[:, t0:t0+t] = 0
    return spec

# ---------------------- DATASET ----------------------
class AudioDataset(Dataset):
    def __init__(self, file_paths, labels, audio_config=None, is_train=True, augment=True):
        self.file_paths = file_paths
        self.labels = labels
        self.audio_config = audio_config
        self.is_train = is_train
        self.augment = augment and is_train

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        try:
            features = np.load(self.file_paths[idx])
            label = self.labels[idx]

            # Apply augmentations only during training
            if self.augment:
                if random.random() < 0.5:
                    features = add_noise(features)
                if random.random() < 0.5:
                    features = time_shift(features)
                if random.random() < 0.5:
                    features = spec_augment(features)

            return torch.FloatTensor(features), torch.LongTensor([label])
        except Exception as e:
            logger.error(f"Error loading {self.file_paths[idx]}: {str(e)}")
            return None, None

# ---------------------- MODEL ----------------------
class BirdClassifierModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.5):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(2),
            nn.Dropout(dropout)
        )

        self.lstm = nn.LSTM(128, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers>1 else 0, bidirectional=True)

        self.attention = nn.Sequential(
            nn.Linear(hidden_size*2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size*2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        cnn_out = self.cnn(x)
        cnn_out = cnn_out.transpose(1, 2)
        lstm_out, _ = self.lstm(cnn_out)
        attn_w = torch.softmax(self.attention(lstm_out), dim=1)
        out = torch.sum(attn_w * lstm_out, dim=1)
        return self.classifier(out)

# ---------------------- CLASSIFIER ----------------------
class BirdClassifier:
    def __init__(self, data_dir, config=None, config_name='full'):
        self.data_dir = data_dir
        self.config_name = config_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        self.config = config or {}
        self.model = None
        self.label_encoder = None
        self.species_info = None

    def load_data(self):
        feature_dir = os.path.join(self.data_dir, 'train_features2', self.config_name)
        if not os.path.exists(feature_dir):
            logger.error(f"Features not found at {feature_dir}")
            sys.exit(1)

        train_df = pd.read_csv(os.path.join(self.data_dir, 'train_metadata.csv'))
        taxonomy_df = pd.read_csv(os.path.join(self.data_dir, 'eBird_Taxonomy_v2021.csv'))
        self.species_info = taxonomy_df.set_index('SPECIES_CODE')[['PRIMARY_COM_NAME','SCI_NAME','ORDER1','FAMILY']].to_dict('index')

        file_paths, labels = [], []
        for _, row in tqdm(train_df.iterrows(), total=len(train_df)):
            fpath = os.path.join(feature_dir, row['filename'].replace('.ogg','.npy'))
            if os.path.exists(fpath):
                file_paths.append(fpath)
                labels.append(row['primary_label'])

        self.label_encoder = LabelEncoder()
        labels = self.label_encoder.fit_transform(labels)
        logger.info(f"Loaded {len(file_paths)} files, {len(self.label_encoder.classes_)} species")
        return file_paths, labels

    def create_data_loaders(self, file_paths, labels):
        test_size = self.config['training'].get('test_size',0.2)
        val_size = self.config['training'].get('val_size',0.1)
        X_temp, X_test, y_temp, y_test = train_test_split(file_paths, labels, test_size=test_size, stratify=labels, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_size/(1-test_size), stratify=y_temp, random_state=42)

        train_dataset = AudioDataset(X_train, y_train, is_train=True, augment=True)
        val_dataset = AudioDataset(X_val, y_val, is_train=False)
        test_dataset = AudioDataset(X_test, y_test, is_train=False)

        if self.config['dataset'].get('balanced_sampling', False):
            class_sample_count = np.bincount(y_train)
            weights = 1. / (class_sample_count + 1e-6)
            sample_weights = np.array([weights[t] for t in y_train])
            sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        else:
            sampler = None

        batch_size = self.config['training']['batch_size']
        num_workers = self.config['training'].get('num_workers',4)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, shuffle=(sampler is None), num_workers=num_workers, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        return train_loader, val_loader, test_loader

    def create_model(self, num_classes):
        input_size = self.config['audio']['n_mfcc'] + 64 + 12
        self.model = BirdClassifierModel(
            input_size=input_size,
            hidden_size=self.config['model']['hidden_size'],
            num_layers=self.config['model']['num_layers'],
            num_classes=num_classes,
            dropout=self.config['model']['dropout']
        ).to(self.device)
        logger.info(f"Model has {sum(p.numel() for p in self.model.parameters())} parameters")
        return self.model

    def train_model(self, train_loader, val_loader):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=self.config['training']['learning_rate'], weight_decay=self.config['training'].get('weight_decay',1e-4))
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

        best_val_acc = 0
        patience_counter = 0
        train_losses, val_losses, train_accs, val_accs = [], [], [], []

        for epoch in range(self.config['training']['num_epochs']):
            self.model.train()
            train_loss, train_correct, train_total = 0,0,0
            for data, target in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
                if data is None: continue
                data, target = data.to(self.device), target.squeeze().to(self.device)
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                train_loss += loss.item()
                _, pred = torch.max(output, 1)
                train_correct += (pred==target).sum().item()
                train_total += target.size(0)

            train_loss /= len(train_loader)
            train_acc = 100*train_correct/train_total

            # Validation
            self.model.eval()
            val_loss, val_correct, val_total = 0,0,0
            with torch.no_grad():
                for data, target in val_loader:
                    if data is None: continue
                    data, target = data.to(self.device), target.squeeze().to(self.device)
                    output = self.model(data)
                    loss = criterion(output, target)
                    val_loss += loss.item()
                    _, pred = torch.max(output,1)
                    val_correct += (pred==target).sum().item()
                    val_total += target.size(0)

            val_loss /= len(val_loader)
            val_acc = 100*val_correct/val_total

            scheduler.step(val_loss)
            train_losses.append(train_loss); val_losses.append(val_loss)
            train_accs.append(train_acc); val_accs.append(val_acc)

            logger.info(f"Epoch {epoch+1}: Train {train_acc:.2f}% | Val {val_acc:.2f}%")

            if val_acc>best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                model_filename = f"best_model_{timestamp}_acc_{best_val_acc:.2f}.pth"
                torch.save(self.model.state_dict(), model_filename)
                logger.info(f"Saved new best model: {model_filename}")
            else:
                patience_counter += 1

            if patience_counter >= self.config['training']['patience']:
                logger.info("Early stopping")
                break

        self.plot_training_history(train_losses, val_losses, train_accs, val_accs)
        return train_losses, val_losses, train_accs, val_accs

    def evaluate_model(self, test_loader):
        self.model.eval()
        correct, total = 0,0
        preds, targets = [], []
        with torch.no_grad():
            for data, target in tqdm(test_loader, desc="Testing"):
                if data is None: continue
                data, target = data.to(self.device), target.squeeze().to(self.device)
                output = self.model(data)
                _, pred = torch.max(output,1)
                correct += (pred==target).sum().item()
                total += target.size(0)
                preds.extend(pred.cpu().numpy()); targets.extend(target.cpu().numpy())
        acc = 100*correct/total
        logger.info(f"Test Accuracy {acc:.2f}%")
        report = classification_report(targets, preds, target_names=self.label_encoder.classes_)
        logger.info(report)
        return acc, preds, targets

    def plot_training_history(self, train_losses, val_losses, train_accs, val_accs):
        fig, (ax1, ax2) = plt.subplots(1,2,figsize=(12,5))
        ax1.plot(train_losses,label="train"); ax1.plot(val_losses,label="val"); ax1.set_title("Loss"); ax1.legend()
        ax2.plot(train_accs,label="train"); ax2.plot(val_accs,label="val"); ax2.set_title("Accuracy"); ax2.legend()
        plt.tight_layout(); plt.savefig("training_history.png",dpi=200)

    def save_model(self, filepath):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'label_encoder': self.label_encoder,
            'config': self.config,
            'species_info': self.species_info
        }, filepath)
        logger.info(f"Saved to {filepath}")

    def load_model(self, filepath):
        ckpt = torch.load(filepath, map_location=self.device)
        self.config = ckpt['config']
        self.label_encoder = ckpt['label_encoder']
        self.species_info = ckpt['species_info']
        self.create_model(len(self.label_encoder.classes_))
        self.model.load_state_dict(ckpt['model_state_dict'])
        logger.info(f"Loaded model from {filepath}")
