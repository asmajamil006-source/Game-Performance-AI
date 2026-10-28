"""Load a trained model from disk, or train and save it.

Training takes roughly half a minute, so the trained bundle is stored with joblib
and re-used.  The stored model is trusted only if it matches the current dataset
fingerprint, the schema version and the installed scikit-learn version; otherwise
it is rebuilt automatically.  If the artifact folder is read-only the model is
still trained and used in memory.

Command line:  python model_manager.py            # train/refresh and print a report
               python model_manager.py --retrain  # force retraining
"""
from __future__ import annotations

import sys

import joblib
import sklearn

import config as cfg
from model_engine import GamePerformancePredictor, train_bundle
from utils import DataValidationError, GamePerfError, file_fingerprint, get_logger

log = get_logger()


def _bundle_is_current(bundle: object, fingerprint: str) -> bool:
    return (
        isinstance(bundle, dict)
        and bundle.get("schema_version") == cfg.BUNDLE_SCHEMA_VERSION
        and bundle.get("sklearn_version") == sklearn.__version__
        and bundle.get("dataset_hash") == fingerprint
    )


def load_or_train(force: bool = False) -> GamePerformancePredictor:
    """Return a ready predictor. Raises GamePerfError subclasses with user-friendly messages."""
    if not cfg.DATA_PATH.exists():
        raise DataValidationError(f"Dataset file '{cfg.DATA_PATH.name}' was not found next to app.py.")
    fingerprint = file_fingerprint(cfg.DATA_PATH)

    if not force and cfg.MODEL_PATH.exists():
        try:
            bundle = joblib.load(cfg.MODEL_PATH)
            if _bundle_is_current(bundle, fingerprint):
                log.info("loaded saved model from %s", cfg.MODEL_PATH.name)
                return GamePerformancePredictor(bundle)
            log.info("saved model is outdated; retraining")
        except Exception:                                       # corrupt / incompatible file
            log.warning("saved model could not be read; retraining", exc_info=True)

    bundle = train_bundle(fingerprint)
    try:
        cfg.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, cfg.MODEL_PATH, compress=3)
        log.info("saved model to %s", cfg.MODEL_PATH)
    except OSError:
        log.warning("could not save the model artifact; continuing in memory", exc_info=True)
    return GamePerformancePredictor(bundle)


if __name__ == "__main__":
    try:
        predictor = load_or_train(force="--retrain" in sys.argv)
    except GamePerfError as err:
        print(f"ERROR: {err}")
        sys.exit(1)
    b = predictor.b
    m = b["test_metrics"]
    print(f"Selected model : {b['model_name']} ({b['feature_set']['label']})")
    print(f"Why            : {b['selection_reason']}")
    print(f"Test metrics   : MAE={m['MAE']:.2f}  RMSE={m['RMSE']:.2f}  R2={m['R2']:.3f}")
    print(f"80% interval   : observed test coverage {b['interval']['coverage_test']:.1%}")
