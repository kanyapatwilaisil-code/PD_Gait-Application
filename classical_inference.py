"""Adapter that lets the existing GUI pipeline use saved pdml Classical models."""
from pathlib import Path
import json
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference_augmented import AugmentedPredictor
from frozen_pdml.protocol import Protocol
from model_loader import load_model


class ClassicalPredictor(AugmentedPredictor):
    """Use the saved SVM or Random Forest model with the same F3 windows."""

    def __init__(self, model_name):
        if model_name not in ("SVM", "Random Forest"):
            raise ValueError(f"Unsupported classical model: {model_name}")

        model = load_model(model_name)
        if getattr(model, "feature_set", None) != "F3":
            raise ValueError(
                f"{model_name} uses feature_set={getattr(model, 'feature_set', None)!r}; "
                "this GUI adapter expects F3."
            )
        if not callable(getattr(model, "scores", None)):
            raise TypeError(f"Saved {model_name} model does not expose scores(windows).")

        preprocessing_path = HERE / "model" / "preprocessing.json"
        if not preprocessing_path.is_file():
            raise FileNotFoundError(f"Missing preprocessing config: {preprocessing_path}")
        self.cfg = Protocol(**json.loads(preprocessing_path.read_text(encoding="utf-8")))
        if self.cfg.window != 45 or self.cfg.stride != 22:
            raise ValueError(
                f"Expected F3 window/stride 45/22, got {self.cfg.window}/{self.cfg.stride}."
            )

        self.model = model
        self.threshold = float(getattr(model, "threshold", 0.0 if model_name == "SVM" else 0.5))
        self.reference_max_error = None
        self.manifest = {
            "model": model_name.lower().replace(" ", "_"),
            "display_name": model_name,
            "feature_set": model.feature_set,
            "threshold": self.threshold,
            "score_kind": getattr(model, "score_kind", "unknown"),
            "model_path": str(__import__("model_loader").MODEL_PATHS[model_name]),
            "bundle_type": "pdml_classical",
        }

    def scores(self, windows):
        windows = np.asarray(windows)
        if len(windows) == 0:
            return np.empty(0, dtype=np.float32)
        return np.asarray(self.model.scores(windows), dtype=np.float64).reshape(-1)
