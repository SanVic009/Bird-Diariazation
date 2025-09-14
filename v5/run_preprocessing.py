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
    parser = argparse.ArgumentParser(description="Preprocess bird audio data for RFCX dataset.")
    parser.add_argument("--rfcx_root", type=str, required=True, help="Path to RFCX dataset root")
    parser.add_argument("--csv_file", type=str, default="train_tp.csv", help="Name of the csv file in the rfcx_root, e.g., train_tp.csv")
    parser.add_argument("--out_dir", type=str, default="processed_rfcx", help="Directory to save processed files")
    parser.add_argument("--duration_strategy", type=str, default="adaptive", 
                        choices=["fixed", "adaptive", "segments"],
                        help="Duration processing strategy")
    parser.add_argument("--min_duration", type=float, default=3.0, help="Minimum duration for adaptive strategy")
    parser.add_argument("--max_duration", type=float, default=10.0, help="Maximum duration for adaptive/segments")
    parser.add_argument("--fixed_duration", type=float, default=5.0, help="Fixed duration for fixed strategy")
    parser.add_argument("--mixup_prob", type=float, default=0.3, help="Probability of applying MixUp augmentation (0.0-1.0)")
    parser.add_argument("--mixup_alpha", type=float, default=0.2, help="Beta distribution parameter for MixUp (lower = more extreme mixing)")
    
    # Enhanced augmentation parameters
    parser.add_argument("--noise_snr_min", type=float, default=15, help="Minimum SNR for Gaussian noise (dB)")
    parser.add_argument("--noise_snr_max", type=float, default=30, help="Maximum SNR for Gaussian noise (dB)")
    parser.add_argument("--freq_mask_num", type=int, default=2, help="Number of frequency masks in SpecAugment")
    parser.add_argument("--time_mask_num", type=int, default=2, help="Number of time masks in SpecAugment")
    parser.add_argument("--freq_mask_max", type=int, default=20, help="Maximum frequency mask size")
    parser.add_argument("--time_mask_max", type=int, default=30, help="Maximum time mask size")
    return parser.parse_args()

def main():
    args = parse_args()
    
    preprocessor = BirdPreprocessor(
        out_dir=args.out_dir,
        duration_strategy=args.duration_strategy,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        fixed_duration=args.fixed_duration,
        mixup_prob=args.mixup_prob,
        mixup_alpha=args.mixup_alpha,
        noise_snr_range=(args.noise_snr_min, args.noise_snr_max),
        freq_mask_num=args.freq_mask_num,
        time_mask_num=args.time_mask_num,
        freq_mask_max=args.freq_mask_max,
        time_mask_max=args.time_mask_max
    )
    
    csv_path = os.path.join(args.rfcx_root, args.csv_file)
    df = pd.read_csv(csv_path)
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    processed_metadata = []
    total_segments = 0
    
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing"):
        recording_id = row["recording_id"]
        species_id = str(row["species_id"])
        
        in_path = os.path.join(args.rfcx_root, "train", f"{recording_id}.flac")
        
        try:
            out_paths = preprocessor.process_and_save(in_path, species_id)
            if not out_paths:
                print(f"[WARNING] No segments processed for {in_path}. Skipping file.")
                logging.warning(f"[WARNING] No segments processed for {in_path}. Skipping file.")
                continue
            # Handle multiple segments (list) or single file
            if isinstance(out_paths, list):
                for out_path in out_paths:
                    processed_metadata.append({
                        "filepath": out_path,
                        "species_id": species_id,
                        "recording_id": recording_id
                    })
                total_segments += len(out_paths)
            else:
                # Backward compatibility for single file
                processed_metadata.append({
                    "filepath": out_paths,
                    "species_id": species_id,
                    "recording_id": recording_id
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
