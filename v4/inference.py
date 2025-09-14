import torch
import numpy as np
from mobilenet import MobileNetBird
from preprocessing import BirdPreprocessor

# set params (must match training)
N_CLASSES = 22
MODEL_PATH = "checkpoints_mobilenet/last_model_20250907-142350_acc0.7800.pth"

device = "cuda" if torch.cuda.is_available() else "cpu"

# load model
model = MobileNetBird(n_classes=N_CLASSES, multi_label=False).to(device) # Set multi_label to False for multi-class
state_dict = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state_dict)
model.eval()

# initialize preprocessor with same params but no augmentation
preprocessor = BirdPreprocessor(
    sample_rate=32000,
    n_mels=128,
    n_fft=1024,
    hop_length=512,
    fmin=20,
    fmax=16000,
    duration_strategy="adaptive",
    augment_prob=0.0   # turn off augmentation for inference
)

# your class names (must match training order)
class_names = ['barswa', 'bcnher', 'bkwsti', 'blrwar1', 'comgre', 'comkin1', 'commoo3', 'comros', 'comsan', 'eaywag1', 'eucdov', 'eurcoo', 'graher1', 'grewar3', 'grnsan', 'grywag', 'hoopoe', 'houspa', 'lirplo', 'litgre1', 'woosan', 'zitcis1']

def preprocess_for_inference(filepath: str):
    wav = preprocessor.load_audio(filepath)
    segments = preprocessor.process_duration(wav)

    tensors = []
    for seg in segments:
        log_mel = preprocessor.to_log_mel(seg)
        # normalization (same as training — here simple mean/std)
        log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-9)
        tensor = torch.tensor(log_mel).unsqueeze(0).unsqueeze(0).float()
        tensors.append(tensor)
    return tensors

def predict(filepath, threshold=0.5):
    segments = preprocess_for_inference(filepath)
    results = []
    with torch.no_grad():
        for tensor in segments:
            tensor = tensor.to(device)
            logits = model(tensor)
            probs = torch.sigmoid(logits).cpu().numpy()[0]

            labels = [(class_names[i], float(p)) for i, p in enumerate(probs) if p > threshold]
            results.extend(labels)
    # Remove duplicates and sort by confidence
    results = sorted(list(set(results)), key=lambda x: x[1], reverse=True)
    return results

def predict_from_npy(filepath, threshold=0.5):
    log_mel = np.load(filepath)
    # normalization
    log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-9)
    tensor = torch.tensor(log_mel).unsqueeze(0).unsqueeze(0).float()
    
    results = []
    with torch.no_grad():
        tensor = tensor.to(device)
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0] # Use softmax for multi-class

        labels = [(class_names[i], float(p)) for i, p in enumerate(probs) if p > threshold]
        results.extend(labels)
    
    # Sort by confidence
    results = sorted(results, key=lambda x: x[1], reverse=True)
    return results

if __name__ == "__main__":
    import pandas as pd
    import os

    METADATA_PATH = "/home/sanvict/Documents/GitHub/Bird-Diariazation/sound/train_metadata.csv"
    AUDIO_DIR = "/home/sanvict/Documents/GitHub/Bird-Diariazation/sound/train_audio"
    N_SAMPLES = 10
    THRESHOLD = 0.1

    # Read metadata
    df = pd.read_csv(METADATA_PATH)

    # Get random samples
    sample_df = df.sample(n=N_SAMPLES, random_state=42)

    for idx, row in sample_df.iterrows():
        filepath = os.path.join(AUDIO_DIR, row['filename'])
        
        print("-" * 50)
        print(f"Processing file: {filepath}")
        
        if filepath.endswith('.npy'):
            predictions = predict_from_npy(filepath, threshold=THRESHOLD)
        else:
            predictions = predict(filepath, threshold=THRESHOLD)
        
        if predictions:
            print(f"Top 3 predictions for {filepath}:")
            for label, prob in predictions[:3]:
                print(f"  - {label}: {prob:.4f}")
        else:
            print(f"No species detected above the threshold of {THRESHOLD} in {filepath}.")
