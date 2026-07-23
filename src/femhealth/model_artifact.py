"""Final model artifact persistence and validated loading."""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.base import clone

from femhealth.data import WDBC_FEATURE_COUNT, WDBC_FEATURE_NAMES, load_wdbc_data
from femhealth.data_split import RANDOM_STATE, TEST_SIZE, split_development_test
from femhealth.final_evaluation import DEVELOPMENT_CLASS_DISTRIBUTION, DEVELOPMENT_SAMPLE_COUNT
from femhealth.final_selection import (
    SELECTED_CALIBRATION,
    SELECTED_MODEL,
    SELECTED_THRESHOLD,
    SELECTED_VARIANT,
    build_selected_estimator,
)
from femhealth.model_evaluation import BENIGN_LABEL, MALIGNANT_LABEL, validate_development_data

MODEL_ARTIFACT_VERSION = "1.0.0"
ARTIFACT_DIRECTORY = Path("artifacts/model")
MODEL_FILENAME = "femhealth_svm_sigmoid.joblib"
METADATA_FILENAME = "femhealth_svm_sigmoid.metadata.json"
MODEL_PATH = ARTIFACT_DIRECTORY / MODEL_FILENAME
METADATA_PATH = ARTIFACT_DIRECTORY / METADATA_FILENAME
DEFAULT_FINAL_SUMMARY_PATH = Path("reports/results/final_holdout_summary.json")
DATASET_NAME = "Breast Cancer Wisconsin Diagnostic"
DATASET_SOURCE = "sklearn.datasets.load_breast_cancer(as_frame=True)"


def fit_selected_artifact_estimator(
    X_development: pd.DataFrame,
    y_development: pd.Series,
    estimator=None,
) -> object:
    """Fit the frozen selected estimator using only development data."""
    _validate_artifact_training_data(X_development, y_development)
    selected_estimator = build_selected_estimator() if estimator is None else estimator
    fitted_estimator = clone(selected_estimator)
    fitted_estimator.fit(X_development, y_development)
    return fitted_estimator


def build_artifact_metadata(
    fitted_estimator: object,
    model_sha256: str,
    final_summary_path: Path = DEFAULT_FINAL_SUMMARY_PATH,
) -> dict[str, object]:
    """Build reproducible metadata for the final model artifact."""
    final_holdout_metrics = _load_validated_final_summary(final_summary_path)

    return {
        "artifact_version": MODEL_ARTIFACT_VERSION,
        "selected_variant": SELECTED_VARIANT,
        "selected_model": SELECTED_MODEL,
        "selected_calibration": SELECTED_CALIBRATION,
        "threshold": SELECTED_THRESHOLD,
        "class_labels": {
            "malignant": MALIGNANT_LABEL,
            "benign": BENIGN_LABEL,
        },
        "feature_count": WDBC_FEATURE_COUNT,
        "feature_names": WDBC_FEATURE_NAMES,
        "training_sample_count": DEVELOPMENT_SAMPLE_COUNT,
        "training_class_distribution": {
            str(label): count for label, count in DEVELOPMENT_CLASS_DISTRIBUTION.items()
        },
        "dataset_name": DATASET_NAME,
        "dataset_source": DATASET_SOURCE,
        "split": {
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "stratified": True,
        },
        "pipeline": _build_pipeline_metadata(fitted_estimator),
        "final_holdout_metrics": final_holdout_metrics,
        "python_version": platform.python_version(),
        "pandas_version": pd.__version__,
        "scikit_learn_version": sklearn.__version__,
        "joblib_version": joblib.__version__,
        "model_sha256": model_sha256,
    }


def build_and_persist_final_artifact(
    output_directory: Path = ARTIFACT_DIRECTORY,
    final_summary_path: Path = DEFAULT_FINAL_SUMMARY_PATH,
) -> tuple[Path, Path, dict[str, object]]:
    """Build and persist the final model artifact without using holdout predictions."""
    output_directory = Path(output_directory)
    model_path = output_directory / MODEL_FILENAME
    metadata_path = output_directory / METADATA_FILENAME

    _ensure_artifact_paths_are_new(model_path, metadata_path)
    output_directory.mkdir(parents=True, exist_ok=True)

    temporary_model_path = model_path.with_name(f"{model_path.name}.tmp")
    temporary_metadata_path = metadata_path.with_name(f"{metadata_path.name}.tmp")
    created_final_paths: list[Path] = []

    try:
        X, y = load_wdbc_data()
        X_development, X_test, y_development, y_test = split_development_test(X, y)
        del X_test, y_test

        fitted_estimator = fit_selected_artifact_estimator(X_development, y_development)
        joblib.dump(fitted_estimator, temporary_model_path)
        model_sha256 = _sha256_file(temporary_model_path)
        metadata = build_artifact_metadata(
            fitted_estimator,
            model_sha256,
            final_summary_path,
        )
        temporary_metadata_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        temporary_model_path.replace(model_path)
        created_final_paths.append(model_path)
        temporary_metadata_path.replace(metadata_path)
        created_final_paths.append(metadata_path)
    except Exception:
        _remove_paths(
            temporary_model_path,
            temporary_metadata_path,
            *created_final_paths,
        )
        raise

    return model_path, metadata_path, metadata


def load_model_artifact(
    model_path: Path = MODEL_PATH,
    metadata_path: Path = METADATA_PATH,
) -> tuple[object, dict[str, object]]:
    """Load a validated model artifact. Joblib files must come from trusted sources."""
    model_path = Path(model_path)
    metadata_path = Path(metadata_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")

    if not metadata_path.exists():
        raise FileNotFoundError(f"Model metadata not found: {metadata_path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    _validate_loaded_metadata(metadata)

    calculated_sha256 = _sha256_file(model_path)
    if calculated_sha256 != metadata["model_sha256"]:
        raise ValueError("Model artifact hash mismatch")

    estimator = joblib.load(model_path)
    _validate_loaded_estimator(estimator)
    return estimator, metadata


def main() -> None:
    """Build the final artifact and print a compact metadata summary."""
    model_path, metadata_path, metadata = build_and_persist_final_artifact()
    compact_metadata = {
        "artifact_version": metadata["artifact_version"],
        "selected_variant": metadata["selected_variant"],
        "threshold": metadata["threshold"],
        "training_sample_count": metadata["training_sample_count"],
        "model_sha256": metadata["model_sha256"],
    }

    print(f"Modelo salvo em: {model_path}")
    print(f"Metadados salvos em: {metadata_path}")
    print(json.dumps(compact_metadata, indent=2, ensure_ascii=False))


def _validate_artifact_training_data(X: pd.DataFrame, y: pd.Series) -> None:
    validate_development_data(X, y)

    if len(X) != DEVELOPMENT_SAMPLE_COUNT:
        raise ValueError("Unexpected development sample count")

    if y.value_counts().sort_index().to_dict() != DEVELOPMENT_CLASS_DISTRIBUTION:
        raise ValueError("Unexpected development class distribution")


def _ensure_artifact_paths_are_new(model_path: Path, metadata_path: Path) -> None:
    existing_paths = [path for path in (model_path, metadata_path) if path.exists()]
    if existing_paths:
        existing = ", ".join(str(path) for path in existing_paths)
        raise FileExistsError(f"Model artifact already exists: {existing}")


def _load_validated_final_summary(final_summary_path: Path) -> dict[str, object]:
    summary = json.loads(Path(final_summary_path).read_text(encoding="utf-8"))

    if summary.get("selected_variant") != SELECTED_VARIANT:
        raise ValueError("Unexpected final summary variant")

    if summary.get("threshold") != SELECTED_THRESHOLD:
        raise ValueError("Unexpected final summary threshold")

    if summary.get("test_sample_count") != 114:
        raise ValueError("Unexpected final summary sample count")

    return summary


def _build_pipeline_metadata(fitted_estimator: object) -> dict[str, object]:
    pipeline = getattr(fitted_estimator, "estimator", None)
    steps = getattr(pipeline, "named_steps", {})
    scaler = steps.get("scaler")
    classifier = steps.get("model")
    calibration_cv = getattr(fitted_estimator, "cv", None)
    classifier_parameters = classifier.get_params() if hasattr(classifier, "get_params") else {}
    probability = classifier_parameters.get("probability", getattr(classifier, "probability", None))
    if probability == "deprecated":
        probability = False

    return {
        "scaler": type(scaler).__name__ if scaler is not None else None,
        "classifier": type(classifier).__name__ if classifier is not None else None,
        "kernel": getattr(classifier, "kernel", None),
        "C": getattr(classifier, "C", None),
        "gamma": getattr(classifier, "gamma", None),
        "class_weight": getattr(classifier, "class_weight", None),
        "probability": probability,
        "calibration_method": getattr(fitted_estimator, "method", SELECTED_CALIBRATION),
        "calibration_ensemble": getattr(fitted_estimator, "ensemble", None),
        "calibration_cv_splits": getattr(calibration_cv, "n_splits", None),
    }


def _validate_loaded_metadata(metadata: dict[str, object]) -> None:
    if metadata.get("artifact_version") != MODEL_ARTIFACT_VERSION:
        raise ValueError("Unexpected artifact version")

    if metadata.get("selected_variant") != SELECTED_VARIANT:
        raise ValueError("Unexpected selected variant")

    if metadata.get("threshold") != SELECTED_THRESHOLD:
        raise ValueError("Unexpected threshold")

    if metadata.get("feature_count") != WDBC_FEATURE_COUNT:
        raise ValueError("Unexpected feature count")

    if metadata.get("feature_names") != WDBC_FEATURE_NAMES:
        raise ValueError("Unexpected feature names")

    if metadata.get("scikit_learn_version") != sklearn.__version__:
        raise ValueError(
            "Incompatible scikit-learn version: "
            f"artifact uses {metadata.get('scikit_learn_version')}, "
            f"environment uses {sklearn.__version__}"
        )


def _validate_loaded_estimator(estimator: object) -> None:
    if not hasattr(estimator, "classes_"):
        raise ValueError("Model artifact is not fitted")

    if set(estimator.classes_) != {MALIGNANT_LABEL, BENIGN_LABEL}:
        raise ValueError("Unexpected model classes")


def _sha256_file(path: Path) -> str:
    sha256 = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def _remove_paths(*paths: Path) -> None:
    for path in paths:
        path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
