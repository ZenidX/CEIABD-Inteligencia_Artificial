# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a machine learning and computer vision educational repository (CEIABD-IA) containing projects focused on audio classification, signal processing, object detection, and reinforcement learning. Primary development uses Jupyter notebooks and Python scripts.

## Projects

### YOLO
Video object detection application using modern YOLO models (v8-v11) with the supervision library for annotations and tracking.

```bash
cd YOLO
pip install -r requirements.txt

# Webcam (real-time)
python video_detector.py

# Video file
python video_detector.py --source video.mp4 --output resultado.mp4
```

Interactive tutorial available in `yolo_detection_tutorial.ipynb`.

See `YOLO/CLAUDE.md` for CLI arguments and architecture details.

### AudioClassificationResnet
Deep learning project that classifies environmental sounds (50 classes) using transfer learning. Converts audio to Mel-spectrograms and feeds them through ResNet-50.

**Key insight**: Audio files are transformed into RGB images (spectrograms) to leverage image classification models.

Two training notebooks:
- `practica_resnet50_esc50.ipynb` - Main training with frozen backbone
- `practica_resnet50_esc50_congelado.ipynb` - Fine-tuning variant

See `AudioClassificationResnet/CLAUDE.md` for detailed architecture and training configuration.

### VowelsClassificationMachineLearning
Signal processing project for extracting and analyzing vowel formants (F1, F2, F3) from speech recordings using FFT-based analysis.

See `VowelsClassificationMachineLearning/CLAUDE.md` for detailed function documentation and usage examples.

### Reinforcement Learning
Complete RL curriculum organized into 4 levels, from tabular Q-Learning to advanced continuous-control algorithms (PPO, SAC, TD3) with Stable-Baselines3.

**Levels:**
- `01_teoria/` — Theory notebooks (fundamentals, DQN, SB3)
- `02_fundamentos/` — Q-Learning and DQN implementations from scratch
- `03_proyectos_dqn/` — DQN projects: Nibbler (Snake), Flappy Bird, Racing
- `04_proyectos_avanzados/` — SB3 projects: LunarLander, Highway, MiniGrid, PyBullet

```bash
cd "Reinforcement Learning"
pip install gymnasium stable-baselines3 torch numpy matplotlib pygame
pip install gymnasium[box2d] flappy-bird-gymnasium highway-env minigrid pybullet  # per-project extras

# Level 2: tabular / basic DQN
python 02_fundamentos/ejemplos/ejemplo_qlearning_taxi.py
python 02_fundamentos/ejemplos/ejemplo_cartpole_dqn.py

# Level 3: DQN projects (select variant with --variant)
python 03_proyectos_dqn/nibbler/nibbler_game.py --train --variant shaped
python 03_proyectos_dqn/flappybird/flappybird_dqn.py --algorithm PPO --simple
python 03_proyectos_dqn/racing/racing_game.py --train --variant shared

# Level 4: advanced SB3 projects
python 04_proyectos_avanzados/lunarlander/lunarlander_sb3.py --continuous
python 04_proyectos_avanzados/highway/highway_conduccion.py --transfer
python 04_proyectos_avanzados/minigrid/minigrid_navegacion.py --variant cnn
python 04_proyectos_avanzados/pybullet/pybullet_robotica.py --compare-matrix
```

**Training variants per project:**

| Project | Variant flag | Concept |
|---------|-------------|---------|
| Nibbler | `--variant standard` | Baseline DQN |
| Nibbler | `--variant shaped` | Reward shaping (penalizes loops) |
| Nibbler | `--variant curiosity` | Intrinsic curiosity (ICM) |
| Flappy Bird | `--algorithm DQN --simple` | Off-policy, vector obs |
| Flappy Bird | `--algorithm PPO --simple` | On-policy, vector obs |
| Flappy Bird | `--compare-algorithms` | DQN vs PPO comparison |
| Racing | `--variant independent` | 4 separate networks |
| Racing | `--variant shared` | 1 shared network (CTDE) |
| Racing | `--variant master_student` | Transfer between agents |
| Racing | `--variant competitive` | Rank-based reward |
| LunarLander | `--algorithm PPO/DQN/A2C` | Discrete action space |
| LunarLander | `--continuous` | Continuous space (SAC/TD3) |
| Highway | `--env highway` | Single environment |
| Highway | `--transfer` | Transfer learning |
| Highway | `--curriculum` | 5-level curriculum |
| MiniGrid | `--variant flat` | MLP + flattened obs |
| MiniGrid | `--variant cnn` | CNN for 7×7×3 obs |
| MiniGrid | `--curriculum` | Progressive environments |
| PyBullet | `--algorithm PPO/SAC/TD3` | Single algo + robot |
| PyBullet | `--compare-matrix` | 3 algos × 4 robots grid |

See `Reinforcement Learning/CLAUDE.md` for full variant reference and hyperparameter recommendations.

## Development Setup

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows

# Audio Classification dependencies
pip install torch torchvision torchaudio librosa matplotlib pandas tqdm scikit-learn

# YOLO dependencies (or use YOLO/requirements.txt)
pip install ultralytics supervision opencv-python

# Vowel Analysis dependencies
pip install numpy scipy matplotlib ipython

# Reinforcement Learning dependencies
pip install gymnasium stable-baselines3 torch numpy matplotlib pygame
pip install gymnasium[box2d] flappy-bird-gymnasium highway-env minigrid pybullet  # per-project extras

# Jupyter
pip install jupyter
```

For GPU support (CUDA 12.4) - optional but recommended for training:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

## Running Notebooks

```bash
jupyter notebook
```

Then open the desired `.ipynb` file.

## Critical Windows Configuration

**DataLoader on Windows**: Set `NUM_WORKERS = 0` in any PyTorch DataLoader to avoid freezing. This is already configured in the notebooks but must be maintained if creating new code.

## Data Locations

- **ESC-50 Dataset**: `AudioClassificationResnet/data/ESC-50-master/`
  - `audio/` - 2000 WAV files (5s, 44.1kHz)
  - `meta/esc50.csv` - Metadata with filename, fold, target, category

- **Vowel Annotations**: `VowelsClassificationMachineLearning/Vowels/vowels/`
  - `alex.wav` - Test audio (48kHz, stereo)
  - `alex.json` - Vowel timing annotations

## Architecture Patterns

### YOLO Video Detection Pipeline
```
Video frame → YOLO inference → supervision Detections → ByteTrack → Annotators → Output frame
```

### Audio Classification Pipeline
```
WAV (5s, 44.1kHz) → librosa resample (22050Hz) → Mel-spectrogram (128 bands)
→ Normalize [0,1] → RGB 3-channel (224x224) → ResNet-50 (frozen) → Custom head → 50 classes
```

### Formant Extraction Pipeline
```
WAV segment → Pre-emphasis (0.97) → Hamming window → FFT
→ Filter (200-4000Hz) → Smoothing (50Hz) → Peak detection → [F1, F2, F3]
```

## Spectrogram Pre-computation Toggle

In `practica_resnet50_esc50.ipynb`, toggle `USAR_ESPECTROGRAMAS_PRECOMPUTADOS`:
- `True`: Load from `.npy` files in `data/ESC-50-master/audio_image/` (faster)
- `False`: Compute on-the-fly from audio (flexible, slower)

## Model Artifacts

Training generates these files in `AudioClassificationResnet/`:
- `mejor_modelo_esc50.pth` - Best weights only
- `modelo_esc50_completo.pth` - Full checkpoint (weights, categories, history)
- `curvas_aprendizaje.png`, `matriz_confusion.png` - Visualizations

## Key Functions

### Audio Classification (`AudioClassificationResnet/`)
- `audio_a_mel_espectrograma()` - Converts audio to normalized Mel-spectrogram
- `ESC50Dataset` - PyTorch Dataset with pre-computed/on-the-fly modes
- `predecir_audio()` - Returns top-5 predictions

### Vowel Analysis (`VowelsClassificationMachineLearning/Vowels/`)
- `cutvowel(file, start, end)` - Extracts audio segment
- `wav2vec(audio, Fs)` - Returns [F1, F2, F3] formant frequencies
- `distancebv(v1, v2)` - Euclidean distance for vowel comparison
- `plot_spectrum()` - Visualize frequency spectrum with formants

Use `wav2vec_improved.py` over `wav2vec.py` for better error handling and adaptive parameters.
