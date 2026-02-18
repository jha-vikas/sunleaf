# Sunleaf

An Android app that uses on-device ML models to:
1. **Assess sunlight conditions** for plant placement (GPS + compass + camera)
2. **Identify plant species** from photos (47 common houseplants)
3. **Diagnose plant diseases** from leaf photos (38 disease classes)

All ML inference runs entirely on-device. No internet required after installation.

**Technical details:** See [docs/APPROACH.md](docs/APPROACH.md) for architecture, distillation pipeline, and preprocessing contracts.

---

## Project Structure

```
sunleaf/
├── training/                          # Python scripts for model prep
│   ├── disease_detection/             # Download & convert pre-trained disease model
│   ├── species_identification/       # Distill ViT → MobileNetV2 for species ID
│   ├── sunlight/                     # Sun position algorithm + light classifier
│   └── common/                       # Shared utilities
├── models/                            # Output: TFLite model files (Git LFS)
│   └── labels/                        # Class label text files
├── data/                              # Curated knowledge bases
│   ├── disease_remedies.json          # 38 diseases → symptoms + home remedies
│   └── plant_care.json                # 47 species → light/water/humidity needs
├── docs/                              # Technical documentation
│   └── APPROACH.md                   # Architecture, distillation, preprocessing
├── tests/                             # Pytest test suite
├── notebooks/
│   └── colab_training.ipynb          # Google Colab alternative
└── android_app/                       # Kotlin Android app (Phase 2)
```

---

## Quick Start

### Prerequisites
- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- ~3GB disk space for dependencies

### Setup (any machine)

```bash
git clone https://github.com/jha-vikas/sunleaf.git
cd sunleaf

# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh
# Add uv to PATH (skip if uv is already available)
source $HOME/.local/bin/env 2>/dev/null || true

# Create venv and install dependencies
uv sync

# Run tests (sunlight algorithm + knowledge base validation)
uv run pytest tests/ -v
```

---

## Step-by-Step: Model Preparation

### Step 1: Disease Detection Model (any machine, no GPU needed, ~5 min)

Downloads the pre-trained MobileNetV2 from HuggingFace and converts to TFLite.

```bash
uv run python training/disease_detection/download_and_convert.py
```

**Output:**
- `models/plant_disease_v1.tflite` (~10 MB, 38 disease classes)
- `models/labels/disease_labels.txt`

### Step 2: Species Identification Model (Mac or Colab, GPU recommended, ~1-2 hrs)

Distills a large ViT model into a compact MobileNetV2 for mobile deployment.

```bash
uv run python training/species_identification/distill.py
```

Then export to TFLite:

```bash
uv run python training/species_identification/export_tflite.py
```

**Output:**
- `models/plant_species.tflite` (~12 MB, 47 houseplant species)
- `models/labels/species_labels.txt`

### Step 3: Validate Everything

```bash
uv run pytest tests/ -v
```

---

## Training on Mac M3 Max

Your Mac M3 Max with 96GB RAM is the recommended training machine. PyTorch automatically uses the Metal Performance Shaders (MPS) backend for GPU acceleration.

### One-time setup on Mac

```bash
# 1. Clone repo
git clone https://github.com/jha-vikas/sunleaf.git
cd sunleaf

# 2. Install uv (if needed) and add to PATH
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env 2>/dev/null || true

# 3. Install all dependencies
uv sync
```

### Run training on Mac

```bash
# Step 1: Convert disease model (CPU only, ~5 min)
uv run python training/disease_detection/download_and_convert.py

# Step 2: Run species distillation (~1-2 hours on M3 Max)
# PyTorch will auto-detect MPS (Apple Metal GPU)
uv run python training/species_identification/distill.py

# Step 3: Export species model to TFLite (~2 min)
uv run python training/species_identification/export_tflite.py

# Step 4: Validate
uv run pytest tests/ -v

# Step 5: Push trained models back to GitHub
git add models/*.tflite models/labels/*.txt
git commit -m "Add trained TFLite models"
git push
```

### Expected training output on Mac

```
Using Apple Metal (MPS)
Found 47 classes, ~24000 images total
Train: 20400 images, Val: 3600 images
Teacher loaded: 85.8M params
Student model: MobileNetV2 with 2.3M params

Epoch 1/20 | Loss: 2.1234 | Val Acc: 0.4521 | Time: 320s
Epoch 2/20 | Loss: 1.5678 | Val Acc: 0.6234 | Time: 310s
...
Epoch 20/20 | Loss: 0.34 | Val Acc: 0.93 | Time: ~140s

Training complete! Best validation accuracy: ~0.93
```

### Custom training options

```bash
# Fewer epochs for quick testing
uv run python training/species_identification/distill.py --epochs 5 --batch-size 16

# Use pre-downloaded dataset
uv run python training/species_identification/distill.py --dataset-path /path/to/dataset
```

---

## Training on Google Colab (Alternative)

If you don't have access to your Mac, use the included Colab notebook:

1. Open `notebooks/colab_training.ipynb` in Google Colab
2. Select **Runtime > Change runtime type > T4 GPU**
3. Run all cells in order
4. Download the generated `.tflite` files when done

---

## Models Summary

| Model | Architecture | Size | Classes | Accuracy | Training |
|-------|-------------|------|---------|----------|----------|
| Disease Detection | MobileNetV2 | ~10 MB | 38 diseases | ~95% | Pre-trained (download only) |
| Species ID | MobileNetV2 | ~12 MB | 47 houseplants | ~93% | Distillation (~1-2 hrs on M3) |
| Sunlight | Pure algorithm | 0 MB | N/A | N/A | None |

**Total on-device footprint: ~23 MB** (smaller than most photos)

---

## Sunlight Estimation (No ML Required)

The sunlight module uses sensor fusion instead of a trained model:

1. **Sun Position** (NOAA algorithm): GPS coordinates + UTC time → sun altitude & azimuth
2. **Camera Brightness** (EXIF metadata): ISO + shutter speed + aperture → estimated lux
3. **Compass Heading**: Phone magnetometer → which direction camera points
4. **Fusion Logic**: Combines all three to classify as Direct / Bright Indirect / Medium / Low Light

Light categories and their lux thresholds:
- **Direct Sunlight**: >10,800 lux AND camera facing within 45° of sun
- **Bright Indirect**: 2,700 - 10,800 lux
- **Medium Light**: 500 - 2,700 lux
- **Low Light / Shade**: <500 lux

Test the algorithm:

```bash
uv run python training/sunlight/sun_position.py
uv run python training/sunlight/light_classifier.py
```

---

## Development Workflow

```
MSI Laptop (Windows)              Mac M3 Max
├── Write all code                ├── git pull
├── Run disease conversion        ├── Run species distillation
├── Run tests                     ├── Export TFLite models
├── Build Android app             ├── Run tests
├── Deploy APK to phone           └── git push models
└── git push code
         ↕ GitHub ↕
```

---

## What's Next: Android App (Phase 2)

The Android app will be built with:
- **Kotlin** + **Jetpack Compose** (UI)
- **CameraX** (camera access)
- **LiteRT** (TFLite inference)
- **SensorManager** (compass, accelerometer)
- **FusedLocationProviderClient** (GPS)

The `.tflite` files from this project go into `android_app/app/src/main/assets/`.

---

## License

MIT
