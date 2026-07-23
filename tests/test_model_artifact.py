import importlib
import inspect
import json
from pathlib import Path

import joblib
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import femhealth.model_artifact as model_artifact
from femhealth.data import WDBC_FEATURE_NAMES
from femhealth.final_evaluation import DEVELOPMENT_CLASS_DISTRIBUTION, DEVELOPMENT_SAMPLE_COUNT
from femhealth.final_selection import SELECTED_THRESHOLD, SELECTED_VARIANT
from femhealth.model_evaluation import BENIGN_LABEL, MALIGNANT_LABEL


class RecordingArtifactEstimator(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.estimator = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    SVC(
                        kernel="rbf",
                        C=1.0,
                        gamma="scale",
                        class_weight="balanced",
                        probability=False,
                    ),
                ),
            ]
        )
        self.method = "sigmoid"
        self.ensemble = False
        self.cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.fit_index_ = X.index.copy()
        self.fit_target_index_ = y.index.copy()
        self.fit_sample_count_ = len(X)
        self.classes_ = [BENIGN_LABEL, MALIGNANT_LABEL]
        return self


@pytest.fixture()
def development_data() -> tuple[pd.DataFrame, pd.Series]:
    X = pd.DataFrame(
        0.0,
        index=range(DEVELOPMENT_SAMPLE_COUNT),
        columns=WDBC_FEATURE_NAMES,
    )
    y = pd.Series(
        [MALIGNANT_LABEL] * DEVELOPMENT_CLASS_DISTRIBUTION[MALIGNANT_LABEL]
        + [BENIGN_LABEL] * DEVELOPMENT_CLASS_DISTRIBUTION[BENIGN_LABEL],
        index=X.index,
        name="diagnosis",
    )
    return X, y


@pytest.fixture()
def holdout_data() -> tuple[pd.DataFrame, pd.Series]:
    X = pd.DataFrame(
        1.0,
        index=range(1000, 1114),
        columns=WDBC_FEATURE_NAMES,
    )
    y = pd.Series(
        [MALIGNANT_LABEL] * 42 + [BENIGN_LABEL] * 72,
        index=X.index,
        name="diagnosis",
    )
    return X, y


@pytest.fixture()
def final_summary_path(tmp_path) -> Path:
    summary = {
        "selected_variant": SELECTED_VARIANT,
        "threshold": SELECTED_THRESHOLD,
        "test_sample_count": 114,
        "accuracy": 0.9736842105263158,
    }
    path = tmp_path / "final_holdout_summary.json"
    path.write_text(json.dumps(summary), encoding="utf-8")
    return path


def test_artifact_constants_and_default_paths_are_expected() -> None:
    assert model_artifact.MODEL_ARTIFACT_VERSION == "1.0.0"
    assert model_artifact.ARTIFACT_DIRECTORY == Path("artifacts/model")
    assert model_artifact.MODEL_FILENAME == "femhealth_svm_sigmoid.joblib"
    assert model_artifact.METADATA_FILENAME == "femhealth_svm_sigmoid.metadata.json"
    assert model_artifact.MODEL_PATH == Path("artifacts/model/femhealth_svm_sigmoid.joblib")
    assert model_artifact.METADATA_PATH == Path(
        "artifacts/model/femhealth_svm_sigmoid.metadata.json"
    )


def test_fit_selected_artifact_estimator_uses_only_development(development_data) -> None:
    X_development, y_development = development_data

    fitted = model_artifact.fit_selected_artifact_estimator(
        X_development,
        y_development,
        estimator=RecordingArtifactEstimator(),
    )

    assert fitted.fit_sample_count_ == DEVELOPMENT_SAMPLE_COUNT
    assert fitted.fit_index_.equals(X_development.index)
    assert fitted.fit_target_index_.equals(y_development.index)


def test_fit_selected_artifact_estimator_validates_sample_count(development_data) -> None:
    X_development, y_development = development_data

    with pytest.raises(ValueError, match="Unexpected development sample count"):
        model_artifact.fit_selected_artifact_estimator(
            X_development.iloc[:-1].copy(),
            y_development.iloc[:-1].copy(),
            estimator=RecordingArtifactEstimator(),
        )


def test_fit_selected_artifact_estimator_validates_distribution(development_data) -> None:
    X_development, y_development = development_data
    y_development = y_development.copy()
    y_development.iloc[0] = BENIGN_LABEL

    with pytest.raises(ValueError, match="Unexpected development class distribution"):
        model_artifact.fit_selected_artifact_estimator(
            X_development,
            y_development,
            estimator=RecordingArtifactEstimator(),
        )


def test_fit_selected_artifact_estimator_does_not_modify_inputs(development_data) -> None:
    X_development, y_development = development_data
    original_X = X_development.copy(deep=True)
    original_y = y_development.copy(deep=True)

    model_artifact.fit_selected_artifact_estimator(
        X_development,
        y_development,
        estimator=RecordingArtifactEstimator(),
    )

    assert X_development.equals(original_X)
    assert y_development.equals(original_y)


def test_persist_final_artifact_creates_joblib_and_metadata(
    monkeypatch,
    tmp_path,
    development_data,
    holdout_data,
    final_summary_path,
) -> None:
    X_development, y_development = development_data
    X_holdout, y_holdout = holdout_data
    X = pd.concat([X_development, X_holdout])
    y = pd.concat([y_development, y_holdout])

    monkeypatch.setattr(model_artifact, "load_wdbc_data", lambda: (X, y))
    monkeypatch.setattr(
        model_artifact,
        "split_development_test",
        lambda loaded_X, loaded_y: (X_development, X_holdout, y_development, y_holdout),
    )
    monkeypatch.setattr(
        model_artifact,
        "build_selected_estimator",
        RecordingArtifactEstimator,
    )

    model_path, metadata_path, metadata = model_artifact.build_and_persist_final_artifact(
        tmp_path,
        final_summary_path,
    )
    loaded_estimator = joblib.load(model_path)

    assert model_path.exists()
    assert metadata_path.exists()
    assert loaded_estimator.fit_index_.equals(X_development.index)
    assert set(loaded_estimator.fit_index_).isdisjoint(set(X_holdout.index))
    assert metadata["final_holdout_metrics"]["accuracy"] == 0.9736842105263158
    assert metadata["model_sha256"] == model_artifact._sha256_file(model_path)

    required_keys = {
        "artifact_version",
        "selected_variant",
        "selected_model",
        "selected_calibration",
        "threshold",
        "class_labels",
        "feature_count",
        "feature_names",
        "training_sample_count",
        "training_class_distribution",
        "dataset_name",
        "dataset_source",
        "split",
        "pipeline",
        "final_holdout_metrics",
        "python_version",
        "pandas_version",
        "scikit_learn_version",
        "joblib_version",
        "model_sha256",
    }
    assert required_keys.issubset(metadata)
    assert metadata["pipeline"]["scaler"] == "StandardScaler"
    assert metadata["pipeline"]["classifier"] == "SVC"
    assert metadata["pipeline"]["kernel"] == "rbf"
    assert metadata["pipeline"]["probability"] is False
    assert metadata["pipeline"]["calibration_cv_splits"] == 5


def test_persist_final_artifact_refuses_overwrite(tmp_path) -> None:
    model_path = tmp_path / model_artifact.MODEL_FILENAME
    model_path.write_bytes(b"existing")

    with pytest.raises(FileExistsError, match="Model artifact already exists"):
        model_artifact.build_and_persist_final_artifact(tmp_path)


def test_build_artifact_metadata_rejects_summary_variant(final_summary_path) -> None:
    summary = json.loads(final_summary_path.read_text(encoding="utf-8"))
    summary["selected_variant"] = "other"
    final_summary_path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(ValueError, match="Unexpected final summary variant"):
        model_artifact.build_artifact_metadata(
            RecordingArtifactEstimator(),
            "abc",
            final_summary_path,
        )


def test_build_artifact_metadata_rejects_summary_threshold(final_summary_path) -> None:
    summary = json.loads(final_summary_path.read_text(encoding="utf-8"))
    summary["threshold"] = 0.5
    final_summary_path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(ValueError, match="Unexpected final summary threshold"):
        model_artifact.build_artifact_metadata(
            RecordingArtifactEstimator(),
            "abc",
            final_summary_path,
        )


def test_load_model_artifact_validates_hash_and_fitted_estimator(
    tmp_path,
    final_summary_path,
    development_data,
) -> None:
    X_development, y_development = development_data
    estimator = model_artifact.fit_selected_artifact_estimator(
        X_development,
        y_development,
        estimator=RecordingArtifactEstimator(),
    )
    model_path = tmp_path / model_artifact.MODEL_FILENAME
    metadata_path = tmp_path / model_artifact.METADATA_FILENAME
    joblib.dump(estimator, model_path)
    metadata = model_artifact.build_artifact_metadata(
        estimator,
        model_artifact._sha256_file(model_path),
        final_summary_path,
    )
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    loaded_estimator, loaded_metadata = model_artifact.load_model_artifact(
        model_path,
        metadata_path,
    )

    assert hasattr(loaded_estimator, "classes_")
    assert set(loaded_estimator.classes_) == {MALIGNANT_LABEL, BENIGN_LABEL}
    assert loaded_metadata == metadata


def test_load_model_artifact_detects_tampered_joblib(
    tmp_path,
    final_summary_path,
    development_data,
) -> None:
    X_development, y_development = development_data
    estimator = model_artifact.fit_selected_artifact_estimator(
        X_development,
        y_development,
        estimator=RecordingArtifactEstimator(),
    )
    model_path = tmp_path / model_artifact.MODEL_FILENAME
    metadata_path = tmp_path / model_artifact.METADATA_FILENAME
    joblib.dump(estimator, model_path)
    metadata = model_artifact.build_artifact_metadata(
        estimator,
        model_artifact._sha256_file(model_path),
        final_summary_path,
    )
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with model_path.open("ab") as file:
        file.write(b"tampered")

    with pytest.raises(ValueError, match="Model artifact hash mismatch"):
        model_artifact.load_model_artifact(model_path, metadata_path)


def test_load_model_artifact_rejects_incompatible_version(
    tmp_path,
    final_summary_path,
    development_data,
) -> None:
    model_path, metadata_path = _write_loadable_artifact(
        tmp_path,
        final_summary_path,
        development_data,
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["artifact_version"] = "0.0.0"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="Unexpected artifact version"):
        model_artifact.load_model_artifact(model_path, metadata_path)


def test_load_model_artifact_rejects_sklearn_version_mismatch_before_joblib_load(
    monkeypatch,
    tmp_path,
    final_summary_path,
    development_data,
) -> None:
    model_path, metadata_path = _write_loadable_artifact(
        tmp_path,
        final_summary_path,
        development_data,
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["scikit_learn_version"] = "0.0.0"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    def fail_if_loaded(path):
        raise AssertionError("joblib.load must not run with incompatible sklearn version")

    monkeypatch.setattr(model_artifact.joblib, "load", fail_if_loaded)

    with pytest.raises(ValueError, match="Incompatible scikit-learn version"):
        model_artifact.load_model_artifact(model_path, metadata_path)


def test_load_model_artifact_rejects_feature_order(
    tmp_path,
    final_summary_path,
    development_data,
) -> None:
    model_path, metadata_path = _write_loadable_artifact(
        tmp_path,
        final_summary_path,
        development_data,
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["feature_names"] = list(reversed(metadata["feature_names"]))
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="Unexpected feature names"):
        model_artifact.load_model_artifact(model_path, metadata_path)


def test_model_artifact_module_has_no_holdout_evaluation_gridsearch_or_import_training(
    monkeypatch,
    tmp_path,
) -> None:
    module_source = inspect.getsource(model_artifact)
    monkeypatch.chdir(tmp_path)
    importlib.reload(model_artifact)

    assert "evaluate_final_holdout" not in module_source
    assert "GridSearchCV" not in module_source
    assert not Path("artifacts/model/femhealth_svm_sigmoid.joblib").exists()


def _write_loadable_artifact(
    tmp_path,
    final_summary_path,
    development_data,
) -> tuple[Path, Path]:
    X_development, y_development = development_data
    estimator = model_artifact.fit_selected_artifact_estimator(
        X_development,
        y_development,
        estimator=RecordingArtifactEstimator(),
    )
    model_path = tmp_path / model_artifact.MODEL_FILENAME
    metadata_path = tmp_path / model_artifact.METADATA_FILENAME
    joblib.dump(estimator, model_path)
    metadata = model_artifact.build_artifact_metadata(
        estimator,
        model_artifact._sha256_file(model_path),
        final_summary_path,
    )
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    return model_path, metadata_path
