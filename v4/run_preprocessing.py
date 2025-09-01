import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from preprocessing import BirdPreprocessor

def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess bird audio data.")
    parser.add_argument("--data_root", type=str, required=True, help="Path to BirdCLEF-2024 dataset root")
    parser.add_argument("--csv_file", type=str, required=True, help="Path to train_metadata.csv")
    parser.add_argument("--out_dir", type=str, default="processed", help="Directory to save processed files")
    parser.add_argument("--duration_strategy", type=str, default="adaptive", 
                        choices=["fixed", "adaptive", "segments"],
                        help="Duration processing strategy")
    parser.add_argument("--min_duration", type=float, default=3.0, help="Minimum duration for adaptive strategy")
    parser.add_argument("--max_duration", type=float, default=10.0, help="Maximum duration for adaptive/segments")
    parser.add_argument("--fixed_duration", type=float, default=5.0, help="Fixed duration for fixed strategy")
    return parser.parse_args()

def main():
    args = parse_args()
    
    preprocessor = BirdPreprocessor(
        out_dir=args.out_dir,
        duration_strategy=args.duration_strategy,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        fixed_duration=args.fixed_duration
    )
    df = pd.read_csv(args.csv_file)
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    processed_metadata = []
    total_segments = 0
    
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing"):
        fname = row["filename"]
        label = row["primary_label"]
        
        in_path = os.path.join(args.data_root, "train_audio", fname)
        
        try:
            out_paths = preprocessor.process_and_save(in_path, label)
            if not out_paths:
                print(f"[WARNING] No segments processed for {in_path}. Skipping file.")
                logging.warning(f"[WARNING] No segments processed for {in_path}. Skipping file.")
                continue
            # Handle multiple segments (list) or single file
            if isinstance(out_paths, list):
                for out_path in out_paths:
                    processed_metadata.append({
                        "filepath": out_path,
                        "primary_label": label,
                        "original_file": fname
                    })
                total_segments += len(out_paths)
            else:
                # Backward compatibility for single file
                processed_metadata.append({
                    "filepath": out_paths,
                    "primary_label": label,
                    "original_file": fname
                })
                total_segments += 1
        except Exception as e:
            print(f"[ERROR] Unhandled error processing {in_path}: {e}")
            logging.error(f"[ERROR] Unhandled error processing {in_path}: {e}")
            
    processed_df = pd.DataFrame(processed_metadata)
    processed_df.to_csv(out_dir / "metadata.csv", index=False)
    
    print(f"Preprocessing complete.")
    logging.info(f"Preprocessing complete.")
    print(f"Original files: {len(df)}")
    logging.info(f"Original files: {len(df)}")
    print(f"Total segments created: {total_segments}")
    logging.info(f"Total segments created: {total_segments}")
    print(f"Strategy: {args.duration_strategy}")
    logging.info(f"Strategy: {args.duration_strategy}")
    if args.duration_strategy == "adaptive":
        print(f"Duration range: {args.min_duration}s - {args.max_duration}s")
        logging.info(f"Duration range: {args.min_duration}s - {args.max_duration}s")
    elif args.duration_strategy == "segments":
        print(f"Segment length: {args.max_duration}s with 50% overlap")
        logging.info(f"Segment length: {args.max_duration}s with 50% overlap")
    elif args.duration_strategy == "fixed":
        print(f"Fixed duration: {args.fixed_duration}s")
        logging.info(f"Fixed duration: {args.fixed_duration}s")
    print(f"New metadata saved to {out_dir / 'metadata.csv'}")
    logging.info(f"New metadata saved to {out_dir / 'metadata.csv'}")

if __name__ == "__main__":
    main()
