import pandas as pd
import librosa
import soundfile as sf
import os

# --- Config ---
meta_path = "birdclef-2024/train_metadata.csv"
audio_root = "birdclef-2024/train_audio"       # root folder where original audio lives
out_audio_dir = "synthetic_audio"
out_meta_path = "synthetic_metadata.csv"
N = 30000                            # number of synthetic mixes to create
target_sr = 32000
# --------------

os.makedirs(out_audio_dir, exist_ok=True)

# Load metadata
df = pd.read_csv(meta_path)

# Utility: load waveform with librosa
def load_wav(path, target_sr=32000):
    wav, sr = librosa.load(path, sr=target_sr, mono=True)
    return wav, sr

# Initialize metadata store
if os.path.exists(out_meta_path):
    synth_df = pd.read_csv(out_meta_path)
else:
    synth_df = pd.DataFrame(columns=["primary_label", "filename", "type", "source_files"])

for i in range(N):
    # Pick two different species
    while True:
        r1, r2 = df.sample(2).to_dict("records")
        if r1["primary_label"] != r2["primary_label"]:
            break

    f1 = os.path.join(audio_root, r1["filename"])
    f2 = os.path.join(audio_root, r2["filename"])

    try:
        wav1, sr1 = load_wav(f1, target_sr)
        wav2, sr2 = load_wav(f2, target_sr)
    except Exception as e:
        print(f"[WARNING] Skipped {f1}, {f2} due to load error: {e}")
        continue

    min_len = min(len(wav1), len(wav2))
    wav_mix = wav1[:min_len] + wav2[:min_len]
    wav_mix = wav_mix / (abs(wav_mix).max() + 1e-9)  # normalize safely

    # Save mixed audio
    mix_name = f"{r1['primary_label']}_{r2['primary_label']}_{i:05d}.wav"
    out_path = os.path.join(out_audio_dir, mix_name)
    sf.write(out_path, wav_mix, target_sr)

    # Create metadata entry
    new_entry = {
        "primary_label": f"{r1['primary_label']},{r2['primary_label']}",
        "filename": mix_name,
        "type": "synthetic_mix",
        "source_files": f"{r1['filename']},{r2['filename']}"
    }

    synth_df = pd.concat([synth_df, pd.DataFrame([new_entry])], ignore_index=True)

    if (i + 1) % 100 == 0:
        print(f"Generated {i+1}/{N} synthetic files")

# Save metadata
synth_df.to_csv(out_meta_path, index=False)
print(f"Done! Created {N} synthetic mixes. Metadata saved to {out_meta_path}")
