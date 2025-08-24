#!/usr/bin/env python3
import os
import sys
import argparse
import logging
import time
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'training_pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def check_dataset():
    required_files = ["train_metadata.csv", "eBird_Taxonomy_v2021.csv", "sample_submission.csv"]
    required_dirs = ["train_audio"]
    missing = [f for f in required_files if not os.path.exists(f)] + [d for d in required_dirs if not os.path.exists(d)]
    if missing:
        logger.error("Missing: " + ", ".join(missing))
        return False
    return True

def run_preprocessing(config_name):
    from preprocess_data import preprocess_data
    logger.info(f"Preprocessing with {config_name}")
    preprocess_data(config_name)
    return True

def run_training(config_name):
    from bird_classifier import BirdClassifier
    from training_configs import get_config
    config = get_config(config_name)
    logger.info(f"Training config {config_name}: {config}")

    clf = BirdClassifier(".", config, config_name)
    fpaths, labels = clf.load_data()
    train_loader, val_loader, test_loader = clf.create_data_loaders(fpaths, labels)
    clf.create_model(num_classes=len(clf.label_encoder.classes_))
    clf.train_model(train_loader, val_loader)
    test_acc, _, _ = clf.evaluate_model(test_loader)
    clf.save_model("final_bird_model.pth")
    logger.info(f"Final Test Accuracy: {test_acc:.2f}%")
    return True

def run_evaluation():
    if not os.path.exists("best_bird_model.pth"):
        logger.error("No trained model found")
        return False
    from bird_classifier import BirdClassifier
    clf = BirdClassifier(".")
    clf.load_model("best_bird_model.pth")
    logger.info("Model ready for inference")
    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocess", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--config", default="full", choices=["quick","dev","full","balanced","gpu"])
    args = parser.parse_args()

    logger.info("="*80)
    logger.info("BirdCLEF-2024 Pipeline")
    logger.info("="*80)

    if not check_dataset():
        return

    success = True
    if args.full or args.preprocess:
        success &= run_preprocessing(args.config)
    if args.full or args.train:
        feature_dir = os.path.join("train_features2", args.config)
        if not os.path.exists(feature_dir):
            logger.error(f"Run preprocessing first (missing {feature_dir})")
            return
        if success:
            success &= run_training(args.config)
    if args.full or args.evaluate:
        if success:
            success &= run_evaluation()

    if success:
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
    else:
        logger.error("PIPELINE FAILED")

if __name__ == "__main__":
    main()
