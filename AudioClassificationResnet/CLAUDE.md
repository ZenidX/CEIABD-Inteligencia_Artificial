# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Audio classification project using deep learning. Classifies the ESC-50 (Environmental Sound Classification) dataset using a pre-trained ResNet-50 model with transfer learning. The core insight is converting audio to Mel-spectrograms (images), enabling image classification models to classify sounds.

## Setup

```bash
pip install torch torchvision torchaudio librosa matplotlib pandas tqdm scikit-learn
```

## Run

Open and execute `practica_resnet50_esc50.ipynb` in Jupyter Notebook/Lab or VS Code.

**Note**: On Windows, `NUM_WORKERS` must be set to 0 in the DataLoader to avoid freezing.

## Architecture

**Pipeline**: Audio (.wav) → Mel-spectrogram → 3-channel image (224x224) → ResNet-50 → 50 classes

1. **Audio Processing**: `librosa.load()` at 22050 Hz, fixed 5-second duration with zero-padding
2. **Mel-spectrogram**: 128 Mel bands, n_fft=2048, hop_length=512, converted to dB scale
3. **Image Conversion**: Normalized to [0,1], replicated to 3 channels (RGB), resized to 224x224
4. **Transfer Learning**: ResNet-50 backbone frozen, only custom classifier head is trained
5. **Custom Head**: Dropout(0.5) → Linear(2048, 512) → ReLU → Dropout(0.3) → Linear(512, 50)

## Key Files

- `practica_resnet50_esc50.ipynb`: Main training notebook with full pipeline
- `audio EDA.ipynb`: Exploratory analysis showing audio-to-spectrogram conversion
- `utils.py`: Contains `Audio2Spectrogram(audio_file, dict_cfg)` helper function

## Key Functions

- `audio_a_mel_espectrograma(ruta_audio, sr=22050, n_mels=128, duracion_fija=5.0)`: Converts audio file to normalized Mel-spectrogram in dB
- `ESC50Dataset`: PyTorch Dataset supporting both pre-computed (.npy) and on-the-fly spectrogram generation via `usar_precomputados` flag
- `predecir_audio()`: Inference function returning top-5 predictions with probabilities

## Spectrogram Modes

The notebook supports two modes controlled by `USAR_ESPECTROGRAMAS_PRECOMPUTADOS`:
- `True`: Loads pre-computed spectrograms from `data/ESC-50-master/audio_image/` (.npy files) - faster training
- `False`: Computes spectrograms on-the-fly from audio - more flexible but slower

## Training Configuration

- **Optimizer**: Adam (lr=0.01, weight_decay=1e-2)
- **Scheduler**: ReduceLROnPlateau (factor=0.5, patience=3)
- **Loss**: CrossEntropyLoss
- **Batch size**: 16
- **Epochs**: 15
- **Data split**: Folds 1-4 train (1600 samples), Fold 5 validation (400 samples)

## Dataset

ESC-50 located at `data/ESC-50-master/`:
- `audio/`: 2000 WAV files (5 seconds, 44.1 kHz, mono)
- `meta/esc50.csv`: Metadata with filename, fold, target, category columns
- 50 classes, 40 clips per class, pre-arranged into 5 folds for cross-validation

## Generated Files

- `mejor_modelo_esc50.pth`: Best model weights (saved when validation accuracy improves)
- `modelo_esc50_completo.pth`: Full checkpoint with weights, categories, accuracy, and training history
- `curvas_aprendizaje.png`, `matriz_confusion.png`: Training visualizations
