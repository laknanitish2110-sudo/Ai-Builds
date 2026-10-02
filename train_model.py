#!/usr/bin/env python3
"""
SensAI Model Training — entry point.

Usage:
    python train_model.py                                    # synthetic data (validate pipeline)
    python train_model.py --data fer2013 --csv fer2013.csv   # real training
    python train_model.py --data folder --dir ./data/faces/  # custom dataset
    python train_model.py --model mobile_emotion --epochs 30 # MobileNet variant
"""

import sys
from pathlib import Path

# make sensai-model importable as sensai_model
root = Path(__file__).parent
model_dir = root / "sensai-model"
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

# create importable alias
import importlib
import types

sensai_pkg = types.ModuleType("sensai_model")
sensai_pkg.__path__ = [str(model_dir)]
sensai_pkg.__package__ = "sensai_model"
sys.modules["sensai_model"] = sensai_pkg

# now we can import submodules
from sensai_model.training.train import main

if __name__ == "__main__":
    main()
