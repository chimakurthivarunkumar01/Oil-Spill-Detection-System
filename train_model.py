"""Run this to retrain the model. Called automatically by run.sh on first launch."""
import subprocess, sys
subprocess.run([sys.executable, 'build_model.py'], check=True)
