import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend
import matplotlib.pyplot as plt
import os

try:
    csv_path='processed/metadata.csv'
    num_samples=5
    out_image='spectrogram_check.png'

    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Please run the preprocessing script first.")
        logging.error(f"Error: {csv_path} not found. Please run the preprocessing script first.")
    else:
        df = pd.read_csv(csv_path)
        
        if len(df) < num_samples:
            print(f"Warning: Not enough samples in {csv_path} to pick {num_samples}. Picking {len(df)} instead.")
            logging.warning(f"Warning: Not enough samples in {csv_path} to pick {num_samples}. Picking {len(df)} instead.")
            num_samples = len(df)
        
        if num_samples > 0:
            sample_df = df.sample(num_samples)
            
            print("Verifying samples...")
            logging.info("Verifying samples...")
            
            fig, axes = plt.subplots(num_samples, 1, figsize=(10, 3 * num_samples))
            if num_samples == 1:
                axes = [axes] # make it iterable

            for i, (idx, row) in enumerate(sample_df.iterrows()):
                filepath = row['filepath']
                label = row['primary_label']
                
                filename_species = os.path.basename(filepath).split('_')[0]
                print(f"--- Sample {i+1} ---")
                logging.info(f"--- Sample {i+1} ---")
                print(f"  Filepath: {filepath}")
                logging.info(f"  Filepath: {filepath}")
                print(f"  Label from CSV: {label}")
                logging.info(f"  Label from CSV: {label}")
                print(f"  Species from filename: {filename_species}")
                logging.info(f"  Species from filename: {filename_species}")
                
                if label == filename_species:
                    print("  ✅ Label matches filename.")
                    logging.info("  ✅ Label matches filename.")
                else:
                    print(f"  ❌ MISMATCH: Label '{label}' does not match filename species '{filename_species}'.")
                    logging.warning(f"  ❌ MISMATCH: Label '{label}' does not match filename species '{filename_species}'.")
                    
                try:
                    spectrogram = np.load(filepath)
                    if spectrogram.size == 0:
                        print("  ❌ ERROR: Spectrogram file is empty.")
                        logging.error("  ❌ ERROR: Spectrogram file is empty.")
                        axes[i].set_title(f"Label: {label} (EMPTY FILE)")
                        continue

                    im = axes[i].imshow(spectrogram, aspect='auto', origin='lower', cmap='viridis')
                    axes[i].set_title(f"Label: {label} | Shape: {spectrogram.shape}")
                    fig.colorbar(im, ax=axes[i])
                except Exception as e:
                    print(f"  ❌ ERROR loading or plotting {filepath}: {e}")
                    logging.error(f"  ❌ ERROR loading or plotting {filepath}: {e}")
                    axes[i].set_title(f"Error loading {os.path.basename(filepath)}")

            plt.tight_layout()
            plt.savefig(out_image)
            print(f"\nSaved plot of {num_samples} spectrograms to {out_image}")
            logging.info(f"\nSaved plot of {num_samples} spectrograms to {out_image}")
        else:
            print("Error: metadata.csv is empty.")
            logging.error("Error: metadata.csv is empty.")

except Exception as e:
    print(f"An unexpected error occurred: {e}")
    logging.error(f"An unexpected error occurred: {e}")
