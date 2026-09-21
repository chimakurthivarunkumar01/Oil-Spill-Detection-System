#!/bin/bash
echo "=================================================="
echo "  OilGuard v4.0 — ML Oil Spill Detection"
echo "  EuroSAT-calibrated | GB+ET | 52 features"
echo "=================================================="
cd "$(dirname "$0")"
command -v python3 &>/dev/null || { echo "ERROR: Python 3 required"; exit 1; }
pip install flask numpy opencv-python scikit-learn scipy pillow werkzeug -q
if [ ! -f model_data.pkl ]; then
    echo "Training ML model (~90 seconds, first run only)..."
    python3 build_model.py
fi
echo "Starting at http://localhost:5000"
python3 app.py
