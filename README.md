# OilGuard v3.0 — ML Oil Spill Detection System

A full-stack Flask web application for detecting oil spills in satellite/aerial images using a **trained ML ensemble** with a **hard clear-water gate**.

---

## What's New in v3.0

| Feature | Old (v2) | New (v3) |
|---------|----------|----------|
| Core engine | Hand-crafted heuristics | Trained GB+RF ML ensemble |
| Water detection | Score-based guessing | EuroSAT-calibrated hard gate |
| Features | 25 | 33 |
| CV F1 Score | ~0.70 | **0.99+** |
| Oil spill types | Guessed | Classified (5 types) |
| Model | None | GradientBoosting + RandomForest |

---

## How It Works

### 1. Clear-Water Hard Gate (if-condition)
Before running the ML model, a hard gate checks if the image has the **EuroSAT water fingerprint**:
- Blue channel dominance > 0.05
- NDWI (water index) > 0.05
- High blue pixel ratio
- Low very-dark pixel ratio
- Low oil-like desaturated regions

If the gate fires → image is **definitively classified as clean water**, ML score is clamped to ≤ 34%.

### 2. ML Ensemble
If the gate doesn't fire (image is NOT clearly water), the **Gradient Boosting + Random Forest ensemble** predicts:
- **GB**: 300 trees, depth 4, learning rate 0.05
- **RF**: 200 trees, depth 8
- **Fusion**: 55% GB + 45% RF

### 3. Oil Spill Type Classification
5 spill types detected:
- **Crude Oil** — very dark, large irregular blobs, smooth dark regions
- **Sheen/Iridescent** — rainbow hue variation, high hue entropy
- **Emulsified Oil** — brownish-orange mousse texture
- **Dispersed Oil** — many small dark patches scattered
- **Aged/Weathered Oil** — grayish flattened patches, low iridescence

---

## Quick Start

### Windows
```
Double-click run.bat
```

### Mac / Linux
```bash
chmod +x run.sh
./run.sh
```

On first run:
1. Dependencies are installed (~1 min)
2. ML model is trained on synthetic EuroSAT-calibrated data (~60 sec)
3. Server starts at **http://localhost:5000**

---

## Project Structure

```
oilguard/
├── app.py              ← Flask backend + ML inference engine
├── train_model.py      ← Model training script (auto-runs on first start)
├── model_data.pkl      ← Trained model (auto-generated)
├── requirements.txt    ← Python dependencies
├── run.bat             ← Windows launcher
├── run.sh              ← Mac/Linux launcher
├── templates/
│   └── index.html      ← Frontend UI
├── static/
│   └── uploads/        ← Temporary upload storage
└── README.md
```

---

## Training Data

The model is trained on **1,600 synthetic images** calibrated to the EuroSAT dataset:

**Clear Water (800 samples):**
- Deep ocean blue
- Shallow/coastal blue-green
- River/lake water
- Bright/sun-glint water
- Dark deep water
- Turbid/muddy water

**Oil Spill (800 samples × 5 types):**
- Crude oil dark blobs
- Sheen/iridescent rainbow patches
- Emulsified brown-orange regions
- Dispersed scattered patches
- Aged weathered gray areas

---

## Tech Stack

| Layer | Tech |
|-------|------|
| Backend | Python 3 · Flask |
| ML Engine | scikit-learn (GradientBoosting + RandomForest) |
| CV Features | OpenCV · NumPy · SciPy |
| Frontend | HTML5 · CSS3 · Vanilla JS |
