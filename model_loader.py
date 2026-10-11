
from pathlib import Path
import sys
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RUN_DIR = (
    PROJECT_ROOT
    #/ "experiments_v2"
    / "artifacts"
    / "runs"
    / "20260929T115842_DNE_full_958574"
    / "conditions"
)

MODEL_PATHS = {
    "SVM": (
        RUN_DIR / "svm" / "F3"
        / "outer_2" / "candidate_0" / "inner_2"
        / "model.joblib"
    ),

    #"XGBoost": (
     #   RUN_DIR / "xgboost" / "F3"
      #  / "outer_3" / "candidate_0" / "inner_1"
       # / "model.joblib"
    #),

    "Random Forest": (
        RUN_DIR / "random_forest" / "F3"
        / "outer_1" / "candidate_2" / "inner_2"
        / "model.joblib"
    ),
}

def load_model(model_name):

    if model_name not in MODEL_PATHS:
        raise ValueError(f"ไม่รู้จักโมเดล: {model_name}")

    path = MODEL_PATHS[model_name]

    if not path.is_file():
        raise FileNotFoundError(
            f"ไม่พบไฟล์โมเดล: {path}"
        )

    print(f"[DEBUG] Model: {model_name}")
    print(f"[DEBUG] Path: {path}")
    print(f"[DEBUG] File exists: {path.is_file()}")
    return joblib.load(path)