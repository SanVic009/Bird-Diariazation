# How to Use Your Saved Bird Diarization Models (.pt files)

You have several trained diarization models saved as `.pt` files. Here's how to use them:

## 🎯 Available Models

Your trained models are located in:
- `models/best_diarization_model.pt` - Best model from training (recommended)
- `models/final_diarization_encoder.pt` - Final model after all epochs
- `models/diarization_encoder_epoch_X.pt` - Checkpoints from specific epochs
- Root directory: `best_diarization_model.pt`, `final_diarization_encoder.pt`, etc.

## 🚀 Quick Start (Easiest Way)

### Option 1: Test on cached data
```bash
cd cbt/
python quick_diarization.py
```

### Option 2: Use specific model
```bash
python quick_diarization.py --model ../best_diarization_model.pt
```

### Option 3: Test existing script
```bash
python test_trained_model.py
```

## 🔧 Detailed Usage

### Use the comprehensive script:
```bash
python use_saved_model.py
```

This script provides:
- ✅ Automatic model detection and loading
- ✅ Visualization of results with plots
- ✅ t-SNE embedding visualization  
- ✅ Speaker timeline analysis
- ✅ Statistical summaries

## 📋 What Each Script Does

| Script | Purpose | Best For |
|--------|---------|----------|
| `quick_diarization.py` | Fast testing | Quick results, command line use |
| `use_saved_model.py` | Full analysis | Detailed analysis, visualizations |
| `test_trained_model.py` | Model validation | Checking if models work |

## 🎵 Processing Your Own Audio

To diarize a new audio file, you'll need to add audio processing. Here's the basic approach:

```python
from use_saved_model import DiarizationModelLoader

# Load your model
loader = DiarizationModelLoader("models/best_diarization_model.pt")

# Diarize audio file (when implemented)
results = loader.diarize_audio_file("your_audio.wav")

# Visualize results
loader.visualize_results(results)
```

## 📊 Understanding Results

The diarization output includes:
- **Speaker Labels**: Array showing which bird/speaker is active in each segment
- **Number of Speakers**: Total unique birds detected  
- **Embeddings**: Vector representations of each audio segment
- **Timestamps**: When each segment occurs (for audio files)

Example output:
```
🎉 Results:
   🔢 Number of different birds detected: 3
   📊 Total segments analyzed: 50

📋 Speaker breakdown:
   🐦 Bird 0: 18 segments (36.0%)
   🐦 Bird 1: 20 segments (40.0%) 
   🐦 Bird 2: 12 segments (24.0%)
```

## 🔍 Troubleshooting

### Model Not Found Error:
```bash
# Check what models you have:
ls -la *.pt
ls -la models/*.pt

# Use specific model:
python quick_diarization.py --model path/to/your/model.pt
```

### No Cached Data:
```bash
# Check if you have cached mel spectrograms:
ls cache_mels/
```

### CUDA/GPU Issues:
The scripts automatically detect and use GPU if available, but fall back to CPU.

## 📁 File Organization

After running diarization, results are saved to:
```
results/
├── quick_embeddings.npy      # Audio embeddings
├── quick_labels.npy          # Speaker assignments  
├── diarization_visualization.png  # Plots and analysis
└── test_embeddings.npy       # From test scripts
```

## 🎨 Visualization Features

The `use_saved_model.py` script creates:
1. **Speaker Timeline** - When each bird sings
2. **Speaker Distribution** - How much each bird sings  
3. **t-SNE Embedding Plot** - Visual clustering of similar sounds
4. **Summary Statistics** - Detailed breakdown

## ⚡ Performance Tips

- Use `best_diarization_model.pt` for best accuracy
- GPU processing is much faster than CPU
- Start with small samples (30-50 segments) for testing
- Increase `max_speakers` if you expect many different birds

## 🐛 Common Issues

1. **"No trained model found"** → Check model file paths
2. **"No cached data"** → Make sure `cache_mels/` directory exists
3. **CUDA errors** → Models work on both GPU and CPU
4. **Import errors** → Make sure you're in the `cbt/` directory

## 🎯 Next Steps

1. Run `python quick_diarization.py` to test your models
2. Try `python use_saved_model.py` for detailed analysis
3. Experiment with different `max_speakers` values
4. Add your own audio file processing if needed

Your models are ready to use! 🚀