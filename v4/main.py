import argparse
import torch
import random
import numpy as np
from dataset import create_dataloaders
from train_lstm import train_lstm
from train_rf import train_rf
from train_resnet import train_resnet
from train_resnet_gru import train_resnet_gru
from train_mobilenet import train_mobilenet
from train_mobilenet_gru import train_mobilenet_gru
import logging
import sys
import os
from datetime import datetime

# Create a logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Create a logs directory if it doesn't exist
if not os.path.exists('logs'):
    os.makedirs('logs')

# Create a file handler
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
handler = logging.FileHandler(f"logs/output_{timestamp}.log")
handler.setLevel(logging.INFO)

# Create a stream handler to also print to console
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.INFO)

# Create a formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)


# Add the handlers to the logger
logger.addHandler(handler)
logger.addHandler(stream_handler)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Bird Audio Classification Pipeline")

    parser.add_argument("--processed_csv", type=str, default="processed/metadata.csv",
                        help="Path to the pre-processed metadata CSV")
    parser.add_argument("--model_type", type=str, choices=["lstm", "rf", "resnet", "gru", "mobilenet", "mobilenet_gru"], default="lstm",
                        help="Which model to train: 'lstm' or 'rf'")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training. Default is 16 to avoid CUDA out of memory errors.")
    parser.add_argument("--epochs", type=int, default=75)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n_mels", type=int, default=128, help="Number of mel frequency bins")

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed()

    print(f"[INFO] Loading data from {args.processed_csv}")
    logger.info(f"Loading data from {args.processed_csv}")
    train_loader, val_loader, classes = create_dataloaders(
        processed_csv_file=args.processed_csv,
        batch_size=args.batch_size,
    )
    num_classes = len(classes)
    print(f"[INFO] Found {num_classes} bird species")
    logger.info(f"Found {num_classes} bird species")

    if args.model_type == "lstm":
        print("[INFO] Training CNN+LSTM model")
        logger.info("Training CNN+LSTM model")
        train_lstm(
            train_loader=train_loader,
            val_loader=val_loader,
            n_classes=num_classes,
            device=args.device,
            lr=args.lr,
            max_epochs=args.epochs,
            n_mels=args.n_mels
        )

    elif args.model_type == "rf":
        print("[INFO] Training CNN+RandomForest model")
        logger.info("Training CNN+RandomForest model")
        train_rf(
            train_loader=train_loader,
            val_loader=val_loader,
            device=args.device,
            n_mels=args.n_mels
        )

    elif args.model_type == "resnet":
        print("[INFO] Training ResNet model")
        logger.info("Training ResNet model")
        train_resnet(
            train_loader=train_loader,
            val_loader=val_loader,
            n_classes=num_classes,
            device=args.device,
            lr=args.lr,
            max_epochs=args.epochs
        )

    elif args.model_type == "gru":
        print("[INFO] Training ResNet+GRU model")
        logger.info("Training ResNet+GRU model")
        train_resnet_gru(
            train_loader=train_loader,
            val_loader=val_loader,
            n_classes=num_classes,
            device=args.device,
            lr=args.lr,
            max_epochs=args.epochs
        )
    
    elif args.model_type == "mobilenet":
        print("[INFO] Training MobileNet model")
        logger.info("Training MobileNet model")
        train_mobilenet(
            train_loader=train_loader,
            val_loader=val_loader,
            n_classes=num_classes,
            device=args.device,
            lr=args.lr,
            max_epochs=args.epochs
        )

    elif args.model_type == "mobilenet_gru":
        print("[INFO] Training MobileNet+GRU model")
        logger.info("Training MobileNet+GRU model")
        train_mobilenet_gru(
            train_loader=train_loader,
            val_loader=val_loader,
            n_classes=num_classes,
            device=args.device,
            lr=args.lr,
            max_epochs=args.epochs
        )

    else:
        raise ValueError(f"Unknown model type {args.model_type}")


if __name__ == "__main__":
    main()