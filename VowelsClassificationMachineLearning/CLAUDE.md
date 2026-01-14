# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains machine learning and audio processing projects:

1. **Iris Dataset ML Tutorial** - A comprehensive Jupyter notebook exploring machine learning concepts using the classic Iris dataset
2. **Vowel Formant Analysis** - Audio signal processing tools to extract and analyze vowel formants (F1, F2, F3) from speech recordings

## Project Structure

```
IA/
├── intro ML - Iris Dataset.ipynb   # ML tutorial notebook (large, 378KB)
├── Untitled-1.ipynb                 # Empty notebook
├── New folder/                      # Audio processing project
│   ├── wav2vec.py                   # Original formant extraction (basic)
│   ├── wav2vec_improved.py          # Enhanced formant extraction with validation
│   ├── audio-exploration-vowels.ipynb  # Interactive audio analysis notebook
│   └── vowels/
│       ├── alex.wav                 # Audio file for analysis (48kHz)
│       └── alex.json                # Vowel timing annotations
```

## Development Environment

**Language**: Python 3.14.0

**Required Libraries**:
- `numpy` - Array operations and numerical computing
- `scipy` - Audio I/O (`scipy.io.wavfile`)
- `matplotlib` - Plotting and visualization
- `IPython` - Audio playback in notebooks (`IPython.display.Audio`)

**Installation**:
```bash
pip install numpy scipy matplotlib ipython
```

## Running the Code

### Jupyter Notebooks

Start Jupyter and open notebooks:
```bash
jupyter notebook
```

Note: The "intro ML - Iris Dataset.ipynb" is large (378KB). For command-line inspection of specific cells:
```bash
# Count total cells
cat "intro ML - Iris Dataset.ipynb" | jq '.cells | length'

# View specific cells (e.g., first 20)
cat "intro ML - Iris Dataset.ipynb" | jq '.cells[:20]'

# View a specific cell's output (e.g., cell 11)
cat "intro ML - Iris Dataset.ipynb" | jq '.cells[11].outputs'
```

### Audio Processing Modules

The `wav2vec.py` and `wav2vec_improved.py` modules can be imported directly:

```python
from New\ folder.wav2vec_improved import cutvowel, wav2vec, distancebv, displaycase

# Extract audio segment
Fs, audiocut = cutvowel("New folder/vowels/alex.wav", start=2.285, end=2.329)

# Extract formants
formants = wav2vec(audiocut, Fs)  # Returns [F1, F2, F3] in Hz
```

Run standalone (shows usage information):
```bash
cd "New folder"
python wav2vec_improved.py
```

## Audio Processing Architecture

### Core Functions

**`cutvowel(file_address, start, end)`**
- Extracts audio segment from WAV file
- Handles both mono and stereo (converts to mono)
- Returns: `(Fs, audiocut)` where Fs is sample rate in Hz
- Location: `wav2vec_improved.py:7-44`

**`wav2vec(cut, Fs)`**
- Extracts three formant frequencies (F1, F2, F3) from audio
- Applies pre-emphasis filter (0.97 coefficient) to enhance high frequencies
- Uses Hamming window to reduce spectral leakage
- Analyzes 200-4000 Hz range (typical vocal formant region)
- Applies adaptive smoothing (~50Hz window) and peak detection
- Returns: Array of formant frequencies in Hz, sorted ascending
- Location: `wav2vec_improved.py:47-129`

**`distancebv(vect1, vect2)`**
- Calculates Euclidean distance between formant vectors
- Ignores NaN values in calculation
- Used for vowel classification/comparison
- Location: `wav2vec_improved.py:132-152`

**`normalize_formants(formants, method='log')`**
- Normalizes formants to reduce speaker variability
- Methods: 'log' (logarithmic), 'bark' (perceptual), 'mel' (mel scale)
- Location: `wav2vec_improved.py:155-178`

### Visualization Functions

**`displaycase(testcase, dictmatrix, testmatrix, labels=None, title=...)`**
- Creates 3-subplot visualization comparing formant spaces
- Plots: F1-F2 space, F2-F3 space, formant value comparison
- Supports categorical labels and color coding
- Location: `wav2vec_improved.py:181-280`

**`plot_spectrum(cut, Fs, formants=None, title=...)`**
- Visualizes frequency spectrum with formants marked
- Shows magnitude in dB scale
- Displays 0-4000 Hz range
- Location: `wav2vec_improved.py:283-332`

### Data Format

**Vowel Annotation JSON** (`alex.json`):
```json
[
  {"vocal": "E", "start": "2.285", "end": "2.329"},
  {"vocal": "O", "start": "2.391", "end": "2.442"},
  ...
]
```
- `vocal`: Vowel label (A, E, I, O, U)
- `start`/`end`: Timestamps in seconds (string format, convert to float)

### Algorithm Details

The formant extraction pipeline:
1. **Pre-emphasis**: Boosts high frequencies (coefficient 0.97)
2. **Windowing**: Applies Hamming window to minimize edge effects
3. **FFT**: Transforms to frequency domain
4. **Filtering**:
   - Removes DC component and frequencies below 200 Hz (fundamental pitch)
   - Limits analysis to 4000 Hz (above typical formants)
5. **Smoothing**: Adaptive moving average (~50 Hz width)
6. **Peak Detection**: Finds 3 highest peaks with minimum 150 Hz separation
7. **Conversion**: Bin indices to Hz frequencies

### Differences Between Versions

**`wav2vec.py`** (original):
- Basic implementation with hardcoded filter parameters
- Fixed frequency range (0-300 bins)
- No error handling
- Simple Euclidean distance
- Minimal validation

**`wav2vec_improved.py`** (enhanced):
- Comprehensive error handling and input validation
- Adaptive frequency analysis (up to 4000 Hz)
- Pre-emphasis and Hamming windowing
- Configurable normalization methods
- Handles NaN values in calculations
- Rich visualization capabilities
- Detailed documentation

## Common Workflows

### Analyzing a Single Vowel

```python
import json
from New\ folder.wav2vec_improved import cutvowel, wav2vec, plot_spectrum

# Load annotations
with open("New folder/vowels/alex.json") as f:
    data = json.load(f)

# Select vowel (e.g., index 5)
vowel = data[5]
Fs, cut = cutvowel("New folder/vowels/alex.wav",
                   float(vowel["start"]),
                   float(vowel["end"]))

# Extract formants
formants = wav2vec(cut, Fs)
print(f"Vowel {vowel['vocal']}: F1={formants[0]:.0f} Hz, F2={formants[1]:.0f} Hz, F3={formants[2]:.0f} Hz")

# Visualize
plot_spectrum(cut, Fs, formants, title=f"Vowel {vowel['vocal']}")
```

### Batch Processing All Vowels

```python
import json
import numpy as np
from New\ folder.wav2vec_improved import cutvowel, wav2vec

# Load annotations
with open("New folder/vowels/alex.json") as f:
    data = json.load(f)

# Process all vowels
results = []
for item in data:
    try:
        Fs, cut = cutvowel("New folder/vowels/alex.wav",
                          float(item["start"]),
                          float(item["end"]))
        formants = wav2vec(cut, Fs)
        results.append({
            'vowel': item['vocal'],
            'F1': formants[0],
            'F2': formants[1],
            'F3': formants[2]
        })
    except Exception as e:
        print(f"Error processing {item}: {e}")

# Analyze by vowel type
formant_matrix = np.array([[r['F1'], r['F2'], r['F3']] for r in results])
labels = [r['vowel'] for r in results]
```

### Comparing Vowels

```python
from New\ folder.wav2vec_improved import distancebv, displaycase
import numpy as np

# Assuming formant_matrix and labels from above
# Compare test vowel against dictionary
test_idx = 10
test_formants = formant_matrix[test_idx]
dict_formants = np.delete(formant_matrix, test_idx, axis=0)
dict_labels = np.delete(labels, test_idx)

# Find nearest neighbor
distances = [distancebv(test_formants, ref) for ref in dict_formants]
nearest_idx = np.argmin(distances)
print(f"Nearest vowel: {dict_labels[nearest_idx]} (distance: {distances[nearest_idx]:.1f} Hz)")

# Visualize comparison
displaycase(test_formants, dict_formants, formant_matrix, labels=dict_labels)
```

## Audio File Specifications

- **Format**: WAV (uncompressed)
- **Sample Rate**: 48000 Hz (alex.wav)
- **Channels**: Stereo (automatically converted to mono by `cutvowel`)
- **Typical Vowel Duration**: 0.04-0.3 seconds based on annotations

## Formant Analysis Notes

- **Typical Formant Ranges** (adult speakers):
  - F1: 200-1000 Hz (tongue height: high vowels have low F1)
  - F2: 600-3000 Hz (tongue position: front vowels have high F2)
  - F3: 1500-4000 Hz (less variable, affected by lip rounding)

- **Expected Values for Spanish Vowels**:
  - /a/: F1 ~700-900 Hz, F2 ~1200-1400 Hz
  - /e/: F1 ~400-600 Hz, F2 ~1800-2200 Hz
  - /i/: F1 ~250-350 Hz, F2 ~2200-2800 Hz
  - /o/: F1 ~400-600 Hz, F2 ~800-1000 Hz
  - /u/: F1 ~250-350 Hz, F2 ~600-800 Hz

## Troubleshooting

### Module Import Issues

If importing from `New folder` fails:
```python
import sys
sys.path.append('New folder')
from wav2vec_improved import cutvowel, wav2vec
```

Or use relative imports if working within the folder:
```bash
cd "New folder"
python
>>> from wav2vec_improved import cutvowel, wav2vec
```

### Audio Loading Errors

- Ensure `scipy` is installed: `pip install scipy`
- Check file paths use correct separators (Windows: `\\` or raw strings `r"path"`)
- Verify audio files are uncompressed WAV format (MP3/AAC not supported)

### Formant Detection Issues

If formants are unrealistic or NaN:
- Check audio segment duration (very short segments < 0.02s may fail)
- Verify audio quality (background noise affects peak detection)
- Try different segments from the JSON annotations
- Use `plot_spectrum()` to visually inspect the frequency content
