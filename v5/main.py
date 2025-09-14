import argparse
import torch
import random
import numpy as np
import logging
import sys
import os
from datetime import datetime

# Add v4 directory to Python path to import training modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'v4')))

from dataset import create_dataloaders
from v4.train_lstm import train_lstm
from v4.train_rf import train_rf
from v4.train_resnet import train_resnet
from v4.train_resnet_gru import train_resnet_gru
from v4.train_mobilenet import train_mobilenet
from v4.train_mobilenet_gru import train_mobilenet_gru
from v4.train_efficientnet import train_efficientnet

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

    parser.add_argument("--processed_csv", type=str, default="processed_rfcx/metadata.csv",
                        help="Path to the pre-processed metadata CSV")
    parser.add_argument("--model_type", type=str, choices=["lstm", "rf", "resnet", "gru", "mobilenet", "mobilenet_gru", "efficientnet"], default="lstm",
                        help="Which model to train: 'lstm', 'rf', 'resnet', 'gru', 'mobilenet', 'mobilenet_gru', or 'efficientnet'")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training. Default is 16 to avoid CUDA out of memory errors.")
    parser.add_argument("--epochs", type=int, default=75)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n_mels", type=int, default=128, help="Number of mel frequency bins")
    parser.add_argument("--patience", type=int, default=5, help="Patience for early stopping")
    parser.add_argument("--multi_label", action=argparse.BooleanOptionalAction, default=False,
                        help="Enable multi-label classification (for synthetic mixed audio datasets)")
    return parser.parse_args()
    
def main():
    args = parse_args()
    set_seed()

    print(f"[INFO] Loading data from {args.processed_csv}")
    logger.info(f"Loading data from {args.processed_csv}")
    
    # Enable multi-label for MobileNet when requested
    multi_label_mode = args.multi_label and args.model_type == "mobilenet"
    
    train_loader, val_loader, classes, num_classes = create_dataloaders(
        metadata_csv_file=args.processed_csv,
        batch_size=args.batch_size,
        multi_label=multi_label_mode,
        use_processed=True
    )
    
    classification_type = "multi-label" if multi_label_mode else "multi-class"
    print(f"[INFO] Found {num_classes} bird species ({classification_type} mode)")
    logger.info(f"Found {num_classes} bird species ({classification_type} mode)")
    
    if multi_label_mode:
        print(f"[INFO] Multi-label mode enabled: Each audio sample can have multiple bird species")
        logger.info(f"Multi-label mode enabled: Each audio sample can have multiple bird species")

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
        classification_type = "multi-label" if multi_label_mode else "multi-class"
        print(f"[INFO] Training MobileNet model ({classification_type})")
        logger.info(f"Training MobileNet model ({classification_type})")
        train_mobilenet(
            train_loader=train_loader,
            val_loader=val_loader,
            n_classes=num_classes,
            device=args.device,
            lr=args.lr,
            max_epochs=args.epochs,
            multi_label=multi_label_mode
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

    elif args.model_type == "efficientnet":
        print("[INFO] Training EfficientNet model")
        logger.info("Training EfficientNet model")
        train_efficientnet(
            train_loader=train_loader,
            val_loader=val_loader,
            n_classes=num_classes,
            device=args.device,
            lr=args.lr,
            max_epochs=args.epochs,
            multi_label=multi_label_mode,
            patience=args.patience
        )

    else:
        raise ValueError(f"Unknown model type {args.model_type}")


if __name__ == "__main__":
    main()