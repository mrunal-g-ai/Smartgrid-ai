"""
Model persistence – save and load trained models to/from disk.
All models are stored in the project's models/ directory.
"""
import pickle
import joblib
from pathlib import Path
from datetime import datetime

# ── paths ──────────────────────────────────────────────────────────────────────
_BASE     = Path(__file__).resolve().parent.parent
MODELS_DIR = _BASE / "models"
MODELS_DIR.mkdir(exist_ok=True)

XGB_PATH  = MODELS_DIR / "xgboost_model.joblib"
SARI_PATH = MODELS_DIR / "sarimax_model.pkl"
QL_PATH   = MODELS_DIR / "q_learning_policy.pkl"
META_PATH = MODELS_DIR / "model_metadata.pkl"


def models_exist() -> bool:
    """Return True only when the minimum required models are on disk."""
    return XGB_PATH.exists() and QL_PATH.exists()


def save_models(xgb_model, sarimax_tuple, agent, metrics: dict) -> None:
    """Persist all three models plus metadata."""
    joblib.dump(xgb_model, XGB_PATH)

    if sarimax_tuple is not None:
        with open(SARI_PATH, "wb") as fh:
            pickle.dump(sarimax_tuple, fh)

    with open(QL_PATH, "wb") as fh:
        pickle.dump(agent, fh)

    metadata = {
        "trained_at":   datetime.now().strftime("%Y-%m-%d %H:%M"),
        "metrics":      metrics,
        "train_rows":   metrics.get("train_rows", "—") if isinstance(metrics, dict) else "—",
        "test_rows":    metrics.get("test_rows", "—") if isinstance(metrics, dict) else "—",
        "has_sarimax":  sarimax_tuple is not None,
    }
    with open(META_PATH, "wb") as fh:
        pickle.dump(metadata, fh)


def load_models():
    """
    Load all models from disk.

    Returns
    -------
    tuple  (xgb_model, sarimax_tuple_or_None, agent, metadata_dict)
           or None if models are missing or corrupt.
    """
    if not models_exist():
        return None

    try:
        xgb_model = joblib.load(XGB_PATH)

        sarimax_tuple = None
        if SARI_PATH.exists():
            with open(SARI_PATH, "rb") as fh:
                sarimax_tuple = pickle.load(fh)

        with open(QL_PATH, "rb") as fh:
            agent = pickle.load(fh)

        metadata = {}
        if META_PATH.exists():
            with open(META_PATH, "rb") as fh:
                metadata = pickle.load(fh)

        return xgb_model, sarimax_tuple, agent, metadata

    except Exception as exc:
        # Corrupt files – caller should retrain
        return None


def load_metadata() -> dict:
    """Return stored metadata or an empty dict."""
    if META_PATH.exists():
        try:
            with open(META_PATH, "rb") as fh:
                return pickle.load(fh)
        except Exception:
            pass
    return {}
