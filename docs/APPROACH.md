# Sunleaf: Technical Approach Document

## 1. System Overview

Sunleaf produces two TFLite models and one algorithmic module for an on-device Android plant care app. The total inference payload is ~23 MB with zero network dependency at runtime.

```
                        TRAINING (Mac M3 Max / Colab)
  ┌──────────────────────────────────────────────────────────────┐
  │                                                              │
  │  ┌─────────────────┐   ┌──────────────────────────────────┐  │
  │  │ Disease Module   │   │ Species Module                   │  │
  │  │                  │   │                                  │  │
  │  │ HuggingFace      │   │ Kaggle Dataset (24K images)      │  │
  │  │ pretrained       │   │         │                        │  │
  │  │ MobileNetV2      │   │         ▼                        │  │
  │  │    │             │   │ ViT Teacher ──► MobileNetV2      │  │
  │  │    ▼             │   │ (85.8M)    KD   Student (2.3M)   │  │
  │  │ Key remapping    │   │                  │               │  │
  │  │    │             │   │                  ▼               │  │
  │  │    ▼             │   │           best_model.pth         │  │
  │  │ litert_torch     │   │                  │               │  │
  │  │ .convert()       │   │                  ▼               │  │
  │  │    │             │   │         litert_torch.convert()   │  │
  │  │    ▼             │   │                  │               │  │
  │  │ disease.tflite   │   │                  ▼               │  │
  │  │ (~10 MB, 38 cls) │   │        species.tflite            │  │
  │  │                  │   │        (~12 MB, 47 cls)          │  │
  │  └─────────────────┘   └──────────────────────────────────┘  │
  │                                                              │
  │  ┌─────────────────┐   ┌──────────────────────────────────┐  │
  │  │ Sunlight Module  │   │ Knowledge Bases                  │  │
  │  │ (pure algorithm) │   │ disease_remedies.json (38 keys)  │  │
  │  │ NOAA + EXIF +    │   │ plant_care.json      (47 keys)  │  │
  │  │ compass fusion   │   │                                  │  │
  │  └─────────────────┘   └──────────────────────────────────┘  │
  └──────────────────────────────────────────────────────────────┘
                                │
                                ▼
                    ANDROID APP (Phase 2)
          .tflite assets + JSON bundles + Kotlin NOAA port
```

---

## 2. Module A: Disease Detection

**File**: `training/disease_detection/download_and_convert.py`

### 2.1 Source Model

- **Repository**: `Daksh159/plant-disease-mobilenetv2` on HuggingFace
- **Architecture**: MobileNetV2 (torchvision) with a modified classifier head
- **Training data**: PlantVillage Augmented dataset (~87K images, 38 classes)
- **Reported validation accuracy**: 95%
- **Weights file**: `mobilenetv2_plant.pth` (9.34 MB)

### 2.2 Architecture Detail

The upstream author trained with:

```python
model = models.mobilenet_v2(pretrained=True)
for p in model.features.parameters():
    p.requires_grad = False
model.classifier[1] = nn.Sequential(
    nn.Dropout(0.2),
    nn.Linear(model.classifier[1].in_features, 38)
)
```

This produces a classifier structure where the Linear layer is at index `classifier.1.1` in the state_dict (because `classifier[1]` was replaced with a `Sequential` containing `[Dropout, Linear]`). Our loader remaps these keys to a flat `[Dropout, Linear]` layout:

```python
# classifier.1.1.weight  -->  classifier.1.weight
# classifier.1.1.bias    -->  classifier.1.bias
```

This lets us load into a standard MobileNetV2 with:

```python
classifier = nn.Sequential(
    nn.Dropout(p=0.2),        # classifier.0 (no params)
    nn.Linear(1280, 38),      # classifier.1
)
```

### 2.3 MobileNetV2 Backbone (shared by both models)

MobileNetV2 uses inverted residual blocks with linear bottlenecks:

```
Input (3x224x224)
    │
    ▼
Conv2d 3→32, stride 2       # 112x112
    │
    ▼
InvertedResidual blocks (×17)
  ┌─────────────────────────────┐
  │ 1x1 Conv (expand)           │  Expansion factor t ∈ {1,6}
  │ 3x3 Depthwise Conv          │  Depthwise separable convolution
  │ 1x1 Conv (project, linear)  │  No ReLU after projection
  │ + Residual connection        │  (when stride=1, in_ch==out_ch)
  └─────────────────────────────┘
    │
    ▼
Conv2d 320→1280, 1x1        # 7x7
    │
    ▼
AdaptiveAvgPool → 1280-d vector
    │
    ▼
Classifier head
```

Key properties for mobile:
- **Depthwise separable convolutions** reduce multiply-adds by ~8-9x vs standard convolutions.
- **Linear bottleneck** (no ReLU after the 1x1 projection) prevents information loss in the low-dimensional manifold.
- **~2.3M parameters** for the backbone; classifier head adds `1280 * num_classes` params.
- **last_channel = 1280**: the feature dimension entering the classifier.

### 2.4 TFLite Conversion

```python
import litert_torch
edge_model = litert_torch.convert(model, (sample_input,))
edge_model.export("models/plant_disease_v1.tflite")
```

`litert_torch.convert()` traces the model via `torch.export.export()`, converts the graph to StableHLO IR, then compiles to FlatBuffer-based TFLite format. The sample input `(1, 3, 224, 224)` defines the static input shape baked into the TFLite model.

The ONNX fallback path exists for environments where `litert-torch`'s dependencies (notably `ai-edge-tensorflow`) fail to install.

### 2.5 Label Ordering

The 38 disease labels are the standard PlantVillage classes in alphabetical order. This matches `torchvision.datasets.ImageFolder` convention, which assigns class indices by sorting directory names. The labels cover 14 crop species with both healthy and diseased states (e.g., `Tomato___healthy`, `Tomato___Late_blight`).

---

## 3. Module B: Species Identification (Knowledge Distillation)

### 3.1 Why Distillation

The best available pre-trained model for houseplant identification is a **ViT-Base/16** (`dima806/house-plant-image-detection`, 85.8M params, 343 MB safetensors). This is too large for mobile inference (~350ms+ per frame on midrange devices). Knowledge distillation compresses it into a MobileNetV2 student (2.3M params, ~12 MB TFLite) while retaining ~87% of the teacher's accuracy.

### 3.2 Teacher Model

**File**: Loaded in `distill.py` via `transformers.ViTForImageClassification`

- **Architecture**: ViT-Base/16 (Vision Transformer)
  - Patch size: 16x16
  - Hidden dimension: 768
  - 12 attention heads, 12 layers
  - Intermediate (MLP) size: 3072
- **Input**: 224x224 RGB, normalized with ImageNet stats
- **Output**: 47-class logits (verified from HF `config.json`)
- **Classes**: 47 common houseplants (African Violet through ZZ Plant)

The ViT splits the input image into 196 patches (14x14 grid of 16x16 patches), prepends a [CLS] token, and runs self-attention across all 197 tokens. The [CLS] token's final hidden state is projected to 47 logits.

The teacher is **frozen** during distillation (all `requires_grad = False`). Only forward passes are needed.

### 3.3 Student Model

**File**: Built in `distill.py:build_student()`

```python
student = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
student.classifier = nn.Sequential(
    nn.Dropout(p=0.2),
    nn.Linear(1280, 47),
)
```

The backbone starts from **ImageNet-pretrained weights** (not random init). This gives the student a strong feature extractor from the start; distillation then fine-tunes the entire network (all layers trainable) to match the teacher's behavior on plant images.

### 3.4 Dataset Pipeline

**File**: `dataset.py`

- **Source**: Kaggle `kacpergregorowicz/house-plant-species` via `kagglehub`
- **~24,000 images** across 47 species
- **Split**: 85/15 train/val with fixed seed (42) for reproducibility
- **Class names**: Derived from directory names via `ImageFolder` (alphabetical)

**Training augmentations** (`common/utils.py:get_training_transform`):

```
RandomResizedCrop(224, scale=(0.8, 1.0))   # Simulate zoom/crop variation
RandomHorizontalFlip()                      # Left-right symmetry
RandomVerticalFlip()                        # Some plants have vertical symmetry
RandomRotation(15°)                         # Slight angle variation
ColorJitter(0.2, 0.2, 0.2)                 # Lighting/color variation
ToTensor()                                  # [0,255] uint8 → [0,1] float32
Normalize(ImageNet mean, std)               # Channel-wise standardization
```

**Validation transform**: Only `Resize(224) → ToTensor → Normalize` (no augmentation).

Both teacher and student share the same ImageNet normalization, which is critical since the ViT teacher also expects ImageNet-normalized inputs (`pixel_values`).

The `TransformSubset` wrapper applies transforms lazily at `__getitem__` time. It's defined at module level (not inside the function) to allow Python's multiprocessing to pickle it for `num_workers > 0`.

### 3.5 Distillation Loss

**File**: `distill.py:DistillationLoss`

The loss is a weighted sum of two components:

```
L = α * L_soft + (1 - α) * L_hard
```

where `α = 0.7` (soft-label dominated).

**Soft loss (KL Divergence)**:

```python
soft_student = log_softmax(student_logits / T, dim=1)
soft_teacher = softmax(teacher_logits / T, dim=1)
L_soft = KL_div(soft_student, soft_teacher) * T²
```

Temperature `T = 3.0` softens the probability distributions, making the teacher's "dark knowledge" (relative probabilities of wrong classes) more visible. A Monstera image might get teacher probs of [0.85 Monstera, 0.08 Pothos, 0.04 Elephant Ear, ...]. At T=1 the student would barely learn from the 0.08 and 0.04; at T=3 these become [0.45, 0.20, 0.14, ...] giving the student more gradient signal about inter-class similarity.

The `T²` scaling factor compensates for the magnitude reduction caused by temperature scaling on the gradients (standard Hinton et al. 2015 prescription).

**Hard loss (Cross-Entropy)**:

```python
L_hard = CrossEntropyLoss(student_logits, ground_truth_labels)
```

Standard supervised loss on the ground-truth labels. This prevents the student from only learning the teacher's biases and anchors it to the actual label distribution.

**Why α = 0.7**: The teacher's soft labels carry richer information (inter-class relationships) than one-hot ground truth. However, the teacher isn't perfect, so 30% hard-label weight keeps the student grounded. This ratio is a common default for distillation.

### 3.6 Training Loop

**Hyperparameters**:

| Parameter | Value | Rationale |
|---|---|---|
| Epochs | 20 | Sufficient for convergence with cosine schedule |
| Batch size | 32 | Fits in M3 Max memory with both teacher and student |
| Optimizer | AdamW | Weight decay regularization (1e-4) |
| Learning rate | 1e-4 | Conservative for fine-tuning pretrained backbone |
| Scheduler | CosineAnnealingLR | Smooth decay to near-zero by epoch 20 |
| Temperature | 3.0 | Standard for classification distillation |
| Alpha | 0.7 | Soft-label dominated |

**Per-batch flow**:

```
1. Load batch of (images, labels) → move to MPS device
2. Teacher forward pass (no_grad): images → teacher_logits [B, 47]
3. Student forward pass: images → student_logits [B, 47]
4. If teacher has different class count, slice both to min(teacher, student)
5. Compute combined distillation loss
6. Backprop through student only (teacher is frozen)
7. AdamW step
```

**Checkpointing**: Best model (by validation accuracy) is saved to `checkpoints/best_model.pth` with:
- `model_state_dict`: Student weights
- `optimizer_state_dict`: For potential training resumption
- `val_accuracy`: For tracking
- `num_classes` and `class_names`: For export script

### 3.7 Class Count Mismatch Handling

Both the teacher (HuggingFace ViT) and the dataset (Kaggle ImageFolder) have **exactly 47 classes** (verified by cross-referencing the ViT's `config.json id2label` with the Kaggle dataset directory listing). The mismatch handling code (slicing logits to `min_classes`) is a defensive safeguard for cases where the dataset has a different version.

### 3.8 Export to TFLite

**File**: `export_tflite.py`

```python
checkpoint = torch.load("checkpoints/best_model.pth")
model = build_mobilenetv2(checkpoint["num_classes"])
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

edge_model = litert_torch.convert(model, (torch.randn(1, 3, 224, 224),))
edge_model.export("models/plant_species.tflite")
```

The export script rebuilds the exact same MobileNetV2 architecture, loads the trained weights, and converts. The `.eval()` call is critical because it switches Dropout to pass-through mode and BatchNorm to use running statistics.

---

## 4. Module C: Sunlight Assessment (Algorithmic)

This module uses **no ML model**. It's a sensor-fusion algorithm that classifies light conditions using three inputs available on any Android phone.

### 4.1 Sun Position (NOAA Algorithm)

**File**: `sun_position.py`

Implements the NOAA Solar Calculator, which computes the sun's altitude (elevation) and azimuth (compass bearing) from:
- Observer latitude/longitude (from GPS)
- UTC datetime

**Algorithm pipeline**:

```
(lat, lon, UTC time)
    │
    ▼
Julian Day Number
    │
    ▼
Julian Century (from J2000.0 epoch)
    │
    ▼
Orbital mechanics:
    ├── Geometric mean longitude & anomaly
    ├── Equation of center
    ├── Obliquity of ecliptic
    └── Solar declination
    │
    ▼
Time corrections:
    ├── Equation of time (±15 min)
    └── True solar time
    │
    ▼
Hour angle → Solar zenith → Altitude
                          → Azimuth
```

The key outputs:
- **Altitude**: Degrees above horizon. Negative = nighttime. Solar noon in summer at lat 40°N ≈ 70-73°.
- **Azimuth**: Degrees clockwise from north. Due south = 180° (Northern Hemisphere noon).
- **Accuracy**: Sub-degree for 1800-2100, derived from NOAA spreadsheet formulas.

### 4.2 Lux Estimation from Camera EXIF

**File**: `light_classifier.py:estimate_lux_from_ev()` and `compute_ev_from_exif()`

The camera's auto-exposure settings encode ambient light information:

```
Step 1: EXIF → Exposure Value
    EV = log₂(f² / t) + log₂(ISO / 100)

    where f = f-number (aperture), t = shutter speed (seconds)

Step 2: EV → Lux
    Lux ≈ 2.5 × 2^EV
```

This is the **incident light metering equation** (ISO 2720). It's approximate because:
- Camera meters reflected light, not incident light
- Auto-exposure targets 18% gray (can be off for very bright/dark scenes)
- Different phone cameras have different metering algorithms

Typical values:
| Scene | EV | Lux |
|---|---|---|
| Direct sunlight | 15 | ~82,000 |
| Overcast outdoors | 12 | ~10,000 |
| Well-lit office | 8-9 | ~640-1,280 |
| Dim room | 5 | ~80 |

### 4.3 Sensor Fusion and Classification

**File**: `light_classifier.py:classify_light()`

```
Inputs:
  sun_altitude (from NOAA)
  sun_azimuth  (from NOAA)
  camera_heading (from phone magnetometer/compass)
  estimated_lux (from EXIF)
  latitude (from GPS)

Decision logic:
  1. facing_sun = angle_between(camera_heading, sun_azimuth) < 45°
  2. if sun_altitude ≤ 0 → "Nighttime"
  3. if facing_sun AND lux > 10,800 → "Direct Sunlight"
  4. if lux > 2,700 → "Bright Indirect"
  5. if lux > 500 → "Medium Light"
  6. else → "Low Light / Shade"
```

The lux thresholds are derived from **horticultural foot-candle standards**:
- Direct sun: >1,000 fc (10,800 lux)
- Bright indirect: 250-1,000 fc (2,700-10,800 lux)
- Medium: 50-250 fc (500-2,700 lux)
- Low light: <50 fc (<500 lux)

The `facing_sun` check is the fusion component: high lux alone doesn't mean direct sunlight (could be reflected light). The camera must be pointing within 45° of the sun's azimuth for the "Direct Sunlight" classification.

### 4.4 Window Direction Recommendations

The classifier cross-references compass direction with hemisphere to give daily light estimates:

| Direction (N. Hemisphere) | Daily Light |
|---|---|
| South | 6-8 hours direct + indirect |
| East | 3-5 hours gentle morning |
| West | 3-5 hours afternoon (hot) |
| North | 1-2 hours indirect only |

In the Southern Hemisphere, N/S logic is swapped (north-facing windows get the most sun).

### 4.5 Android Implementation Notes

This Python code is a **reference implementation and test harness**. For the Android app:
- The NOAA algorithm reimplements cleanly in Kotlin (~200 lines of math).
- Lux can come from `SensorManager.SENSOR_TYPE_LIGHT` (ambient light sensor, more accurate) with EXIF as fallback.
- Compass heading comes from `SensorManager.SENSOR_TYPE_ROTATION_VECTOR` (fused magnetometer + accelerometer + gyroscope).
- GPS from `FusedLocationProviderClient`.

---

## 5. Preprocessing Contract (Critical for Android)

Both TFLite models expect **identical preprocessing**. Any mismatch between training-time and inference-time normalization will silently degrade accuracy.

```
Input: RGB image from camera (any resolution)
    │
    ▼
Resize to 224 × 224 (bilinear interpolation)
    │
    ▼
Convert to float32 tensor, scale to [0, 1]
    (divide pixel values by 255.0)
    │
    ▼
Channel-wise normalization:
    R' = (R - 0.485) / 0.229
    G' = (G - 0.456) / 0.224
    B' = (B - 0.406) / 0.225
    │
    ▼
Tensor shape: [1, 3, 224, 224]  (NCHW format for PyTorch)
              [1, 224, 224, 3]  (NHWC format for TFLite)
```

The TFLite conversion via `litert_torch` handles the NCHW → NHWC transpose internally, so the Android-side LiteRT interpreter expects NHWC input.

**On Android** (Kotlin/LiteRT):
```kotlin
val mean = floatArrayOf(0.485f, 0.456f, 0.406f)
val std  = floatArrayOf(0.229f, 0.224f, 0.225f)
// For each pixel: normalized = (pixel / 255.0 - mean) / std
```

---

## 6. Knowledge Bases

### 6.1 `data/disease_remedies.json`

38 entries keyed by PlantVillage class name. Each entry contains:

```json
{
  "display_name": "Apple Scab",
  "symptoms": "...",
  "severity_levels": ["mild", "moderate", "severe"],
  "home_remedies": ["...", "..."],
  "prevention": "..."
}
```

Linked to disease model output by label string matching.

### 6.2 `data/plant_care.json`

47 entries keyed by species name (matching the ViT teacher's `id2label` naming). Each entry:

```json
{
  "light": "Bright Indirect",
  "water": "Water when top inch of soil is dry...",
  "humidity": "Medium to High (50-60%)",
  "temperature": "18-24°C (65-75°F)",
  "common_problems": ["Root rot from overwatering", "..."]
}
```

The `light` field uses the same categories as the sunlight classifier, enabling cross-referencing: "Your plant needs Bright Indirect light, and your current spot provides Medium Light."

---

## 7. On-Device Deployment Summary

| Component | Format | Size | Input | Output |
|---|---|---|---|---|
| Disease Detection | TFLite (FP32) | ~10 MB | 224x224 RGB | 38-class probabilities |
| Species ID | TFLite (FP32) | ~12 MB | 224x224 RGB | 47-class probabilities |
| Sunlight | Kotlin code | 0 MB | GPS + time + compass + EXIF | Light category + recommendations |
| Disease Remedies | JSON asset | ~50 KB | Disease label | Symptoms + remedies |
| Plant Care | JSON asset | ~30 KB | Species label | Care instructions |
| **Total** | | **~23 MB** | | |

---

## 8. Known Limitations and Future Work

1. **No quantization**: Both models are FP32. INT8 post-training quantization via `litert_torch.quantize` would reduce sizes by ~4x with <1% accuracy loss for MobileNetV2.

2. **Disease model domain gap**: Trained on PlantVillage (clean leaf-on-white-background images). Real phone photos with complex backgrounds, multiple leaves, or partial occlusion will likely perform worse. Fine-tuning on in-the-wild images would help.

3. **No confidence thresholding**: The models will always return a prediction even on non-plant images. A confidence threshold (e.g., reject predictions below 30%) should be added in the Android app layer.

4. **Lux estimation accuracy**: EXIF-derived lux is approximate (reflected vs incident light). The Android ambient light sensor (`TYPE_LIGHT`) is more reliable when available.

5. **Single-image classification**: Both models classify single images. No object detection or segmentation — the user needs to frame a single plant/leaf.
