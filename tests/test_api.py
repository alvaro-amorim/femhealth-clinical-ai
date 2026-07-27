import importlib
import inspect

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

import femhealth.api as api_module
import femhealth.api_schemas as api_schemas_module
import femhealth.explainability_artifacts as explainability_artifacts_module
from femhealth.api import create_app
from femhealth.api_schemas import ACADEMIC_DISCLAIMER, PredictionRequest
from femhealth.data import WDBC_FEATURE_NAMES
from femhealth.demo_cases_artifact import (
    DEMO_BENIGN_CASE_COUNT,
    DEMO_CASE_COUNT,
    DEMO_EXPECTED_REFERENCE_LABELS,
    DEMO_EXPECTED_SAMPLE_INDICES,
    DEMO_MALIGNANT_CASE_COUNT,
    DEMO_OFFICIAL_HOLDOUT_ACCURACY,
    DEMO_SELECTION_RULE,
)
from femhealth.explainability_artifacts import PNG_SIGNATURE
from femhealth.final_selection import SELECTED_THRESHOLD, SELECTED_VARIANT

EXPECTED_HEALTH_FIELDS = {
    "status",
    "model_loaded",
    "artifact_version",
    "selected_variant",
}
EXPECTED_MODEL_FIELDS = {
    "artifact_version",
    "selected_variant",
    "selected_model",
    "selected_calibration",
    "threshold",
    "class_labels",
    "feature_count",
    "feature_names",
    "training_sample_count",
    "model_sha256",
    "final_holdout_metrics",
    "disclaimer",
}
EXPECTED_PREDICTION_FIELDS = {
    "probability_malignant",
    "probability_benign",
    "predicted_label",
    "predicted_class",
    "threshold",
    "disclaimer",
}
EXPECTED_EXPLAINABILITY_FIELDS = {
    "method",
    "scorer",
    "selected_variant",
    "selected_model",
    "selected_calibration",
    "selected_threshold",
    "development_sample_count",
    "cv_splits",
    "permutation_repeats",
    "feature_count",
    "detail_row_count",
    "holdout_used",
    "mean_fold_roc_auc",
    "std_fold_roc_auc",
    "features",
    "fold_scores",
    "limitations",
    "plot_endpoint",
    "disclaimer",
}
EXPECTED_DEMO_CASES_FIELDS = {
    "artifact_version",
    "source_dataset",
    "source_split",
    "selection_rule",
    "used_for_training",
    "used_for_model_selection",
    "created_after_final_evaluation",
    "training_sample_count",
    "holdout_sample_count",
    "official_holdout_accuracy",
    "case_count",
    "malignant_case_count",
    "benign_case_count",
    "feature_names",
    "cases",
    "disclaimer",
}


class FakeEstimator:
    classes_ = [1, 0]

    def fit(self, X, y):
        raise AssertionError("API must not fit the estimator")


@pytest.fixture()
def metadata() -> dict[str, object]:
    return {
        "artifact_version": "1.0.0",
        "selected_variant": SELECTED_VARIANT,
        "selected_model": "svm",
        "selected_calibration": "sigmoid",
        "threshold": SELECTED_THRESHOLD,
        "class_labels": {"malignant": 0, "benign": 1},
        "feature_count": len(WDBC_FEATURE_NAMES),
        "feature_names": WDBC_FEATURE_NAMES,
        "training_sample_count": 455,
        "model_sha256": "abc123",
        "final_holdout_metrics": {
            "accuracy": 0.9736842105263158,
            "true_malignant": 41,
        },
    }


@pytest.fixture()
def valid_features() -> dict[str, float]:
    return {feature_name: float(index + 1) for index, feature_name in enumerate(WDBC_FEATURE_NAMES)}


@pytest.fixture()
def explainability_payload() -> dict[str, object]:
    return {
        "method": "cross_validated_permutation_importance",
        "scorer": "roc_auc_malignant",
        "selected_variant": SELECTED_VARIANT,
        "selected_model": "svm",
        "selected_calibration": "sigmoid",
        "selected_threshold": SELECTED_THRESHOLD,
        "development_sample_count": 455,
        "cv_splits": 5,
        "permutation_repeats": 10,
        "feature_count": len(WDBC_FEATURE_NAMES),
        "detail_row_count": 1500,
        "holdout_used": False,
        "mean_fold_roc_auc": 0.996285,
        "std_fold_roc_auc": 0.002,
        "features": [
            {
                "rank": index,
                "feature_name": feature_name,
                "feature_position": WDBC_FEATURE_NAMES.index(feature_name) + 1,
                "mean_importance": 1.0 / index,
                "std_importance": 0.001,
                "median_importance": 1.0 / index,
                "min_importance": -0.001,
                "max_importance": 0.01,
                "positive_fraction": 0.8,
                "fold_count": 5,
                "observation_count": 50,
            }
            for index, feature_name in enumerate(WDBC_FEATURE_NAMES, start=1)
        ],
        "fold_scores": [
            {
                "fold": fold,
                "train_sample_count": 364,
                "validation_sample_count": 91,
                "train_malignant_count": 136,
                "train_benign_count": 228,
                "validation_malignant_count": 34,
                "validation_benign_count": 57,
                "baseline_roc_auc": 0.99,
            }
            for fold in range(1, 6)
        ],
        "limitations": ["Limitação sintética para teste."],
    }


@pytest.fixture()
def explainability_loader(explainability_payload):
    return lambda: (explainability_payload, PNG_SIGNATURE + b"plot")


@pytest.fixture()
def demo_cases_payload(valid_features) -> dict[str, object]:
    cases = []
    for position, sample_index in enumerate(DEMO_EXPECTED_SAMPLE_INDICES, start=1):
        reference_label = DEMO_EXPECTED_REFERENCE_LABELS[sample_index]
        cases.append(
            {
                "case_id": f"demo-{position:02d}",
                "sample_index": sample_index,
                "reference_label": reference_label,
                "reference_class": "malignant" if reference_label == 0 else "benign",
                "features": valid_features,
            }
        )

    return {
        "artifact_version": "1.0.0",
        "source_dataset": "WDBC",
        "source_split": "final_holdout",
        "selection_rule": DEMO_SELECTION_RULE,
        "used_for_training": False,
        "used_for_model_selection": False,
        "created_after_final_evaluation": True,
        "training_sample_count": 455,
        "holdout_sample_count": 114,
        "official_holdout_accuracy": DEMO_OFFICIAL_HOLDOUT_ACCURACY,
        "case_count": DEMO_CASE_COUNT,
        "malignant_case_count": DEMO_MALIGNANT_CASE_COUNT,
        "benign_case_count": DEMO_BENIGN_CASE_COUNT,
        "feature_names": WDBC_FEATURE_NAMES,
        "cases": cases,
    }


@pytest.fixture()
def demo_cases_loader(demo_cases_payload):
    return lambda: demo_cases_payload


def test_create_app_does_not_load_artifact_or_explainability_immediately(
    metadata,
    explainability_payload,
    demo_cases_payload,
) -> None:
    calls = {"artifact": 0, "explainability": 0, "demo": 0}

    def fake_loader():
        calls["artifact"] += 1
        return FakeEstimator(), metadata

    def fake_explainability_loader():
        calls["explainability"] += 1
        return explainability_payload, PNG_SIGNATURE + b"plot"

    def fake_demo_cases_loader():
        calls["demo"] += 1
        return demo_cases_payload

    create_app(
        artifact_loader=fake_loader,
        explainability_loader=fake_explainability_loader,
        demo_cases_loader=fake_demo_cases_loader,
    )

    assert calls == {"artifact": 0, "explainability": 0, "demo": 0}


def test_testclient_lifespan_loads_once_and_removes_state(
    metadata,
    explainability_payload,
    demo_cases_payload,
) -> None:
    calls = {"artifact": 0, "explainability": 0, "demo": 0}
    app = create_app(
        artifact_loader=lambda: _load_once(calls, metadata),
        explainability_loader=lambda: _load_explainability_once(calls, explainability_payload),
        demo_cases_loader=lambda: _load_demo_cases_once(calls, demo_cases_payload),
    )

    with TestClient(app) as client:
        assert calls == {"artifact": 1, "explainability": 1, "demo": 1}
        assert hasattr(app.state, "estimator")
        assert hasattr(app.state, "explainability_payload")
        assert hasattr(app.state, "demo_cases_payload")
        assert client.get("/health").status_code == 200
        assert client.get("/model").status_code == 200
        assert client.get("/explainability").status_code == 200
        assert client.get("/explainability/plot").status_code == 200
        assert client.get("/demo-cases").status_code == 200
        assert client.get("/demo-cases").status_code == 200
        assert calls == {"artifact": 1, "explainability": 1, "demo": 1}

    assert not hasattr(app.state, "estimator")
    assert not hasattr(app.state, "metadata")
    assert not hasattr(app.state, "explainability_payload")
    assert not hasattr(app.state, "explainability_plot_bytes")
    assert not hasattr(app.state, "demo_cases_payload")


def test_health_returns_loaded_model_information(
    metadata,
    explainability_loader,
    demo_cases_loader,
) -> None:
    app = create_app(
        artifact_loader=lambda: (FakeEstimator(), metadata),
        explainability_loader=explainability_loader,
        demo_cases_loader=demo_cases_loader,
    )

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == EXPECTED_HEALTH_FIELDS
    assert payload["status"] == "ok"
    assert payload["model_loaded"] is True
    assert payload["artifact_version"] == metadata["artifact_version"]
    assert payload["selected_variant"] == SELECTED_VARIANT


def test_model_returns_safe_metadata_fields(
    metadata,
    explainability_loader,
    demo_cases_loader,
) -> None:
    app = create_app(
        artifact_loader=lambda: (FakeEstimator(), metadata),
        explainability_loader=explainability_loader,
        demo_cases_loader=demo_cases_loader,
    )

    with TestClient(app) as client:
        response = client.get("/model")

    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == EXPECTED_MODEL_FIELDS
    assert payload["feature_names"] == WDBC_FEATURE_NAMES
    assert payload["disclaimer"] == ACADEMIC_DISCLAIMER
    assert "estimator" not in payload


def test_predict_reorders_features_and_calls_inference_once(
    monkeypatch,
    metadata,
    explainability_loader,
    demo_cases_loader,
    valid_features,
) -> None:
    calls = {"predict": 0}
    captured = {}

    def fake_predict_with_artifact(X, estimator, received_metadata):
        calls["predict"] += 1
        captured["X"] = X.copy()
        captured["estimator"] = estimator
        captured["metadata"] = received_metadata
        return pd.DataFrame(
            {
                "probability_malignant": [0.70],
                "probability_benign": [0.30],
                "predicted_label": [0],
                "predicted_class": ["malignant"],
            },
            index=X.index,
        )

    monkeypatch.setattr(api_module, "predict_with_artifact", fake_predict_with_artifact)
    app = create_app(
        artifact_loader=lambda: (FakeEstimator(), metadata),
        explainability_loader=explainability_loader,
        demo_cases_loader=demo_cases_loader,
    )
    reversed_features = dict(reversed(list(valid_features.items())))
    original_payload = {"features": reversed_features.copy()}

    with TestClient(app) as client:
        response = client.post("/predict", json=original_payload)

    payload = response.json()
    assert response.status_code == 200
    assert calls["predict"] == 1
    assert captured["X"].shape == (1, len(WDBC_FEATURE_NAMES))
    assert list(captured["X"].columns) == WDBC_FEATURE_NAMES
    assert list(captured["X"].index) == ["request"]
    assert captured["X"].iloc[0].to_dict() == valid_features
    assert payload == {
        "probability_malignant": 0.70,
        "probability_benign": 0.30,
        "predicted_label": 0,
        "predicted_class": "malignant",
        "threshold": SELECTED_THRESHOLD,
        "disclaimer": ACADEMIC_DISCLAIMER,
    }
    assert original_payload == {"features": reversed_features}


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param(
            {"features": {feature: 1.0 for feature in WDBC_FEATURE_NAMES[1:]}},
            id="missing-feature",
        ),
        pytest.param(
            {"features": {**{feature: 1.0 for feature in WDBC_FEATURE_NAMES}, "extra": 1.0}},
            id="extra-feature",
        ),
        pytest.param(
            {"features": {**{feature: 1.0 for feature in WDBC_FEATURE_NAMES}, "mean radius": "1"}},
            id="string",
        ),
        pytest.param(
            {"features": {**{feature: 1.0 for feature in WDBC_FEATURE_NAMES}, "mean radius": True}},
            id="bool",
        ),
        pytest.param(
            {"features": {**{feature: 1.0 for feature in WDBC_FEATURE_NAMES}, "mean radius": None}},
            id="null",
        ),
        pytest.param(
            {"features": {feature: 1.0 for feature in WDBC_FEATURE_NAMES}, "extra": "field"},
            id="top-level-extra",
        ),
        pytest.param({}, id="missing-features"),
    ],
)
def test_predict_rejects_invalid_payloads(
    metadata,
    explainability_loader,
    demo_cases_loader,
    payload,
) -> None:
    app = create_app(
        artifact_loader=lambda: (FakeEstimator(), metadata),
        explainability_loader=explainability_loader,
        demo_cases_loader=demo_cases_loader,
    )

    with TestClient(app) as client:
        response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_predict_missing_features_returns_standard_validation_detail(
    metadata,
    explainability_loader,
    demo_cases_loader,
) -> None:
    app = create_app(
        artifact_loader=lambda: (FakeEstimator(), metadata),
        explainability_loader=explainability_loader,
        demo_cases_loader=demo_cases_loader,
    )

    with TestClient(app) as client:
        response = client.post("/predict", json={})

    assert response.status_code == 422
    detail = response.json()["detail"]
    first_error = detail[0]

    assert isinstance(detail, list)
    assert detail
    assert {"type", "loc", "msg"}.issubset(first_error)
    assert "body" in first_error["loc"]
    assert "features" in first_error["loc"]


@pytest.mark.parametrize("invalid_value", [float("nan"), float("inf"), float("-inf")])
def test_prediction_request_rejects_non_finite_values(invalid_value) -> None:
    features = {feature: 1.0 for feature in WDBC_FEATURE_NAMES}
    features["mean radius"] = invalid_value

    with pytest.raises(ValidationError):
        PredictionRequest.model_validate({"features": features})


def test_explainability_returns_payload_and_plot(
    metadata,
    explainability_loader,
    demo_cases_loader,
) -> None:
    app = create_app(
        artifact_loader=lambda: (FakeEstimator(), metadata),
        explainability_loader=explainability_loader,
        demo_cases_loader=demo_cases_loader,
    )

    with TestClient(app) as client:
        response = client.get("/explainability")
        plot_response = client.get("/explainability/plot")

    payload = response.json()
    assert response.status_code == 200
    assert set(payload) == EXPECTED_EXPLAINABILITY_FIELDS
    assert len(payload["features"]) == 30
    assert len(payload["fold_scores"]) == 5
    assert payload["holdout_used"] is False
    assert payload["plot_endpoint"] == "/explainability/plot"
    assert payload["disclaimer"] == ACADEMIC_DISCLAIMER
    assert plot_response.status_code == 200
    assert plot_response.headers["content-type"] == "image/png"
    assert plot_response.content.startswith(PNG_SIGNATURE)


def test_demo_cases_returns_validated_payload_without_inference(
    monkeypatch,
    metadata,
    explainability_loader,
    demo_cases_loader,
) -> None:
    calls = {"predict": 0}

    def fake_predict_with_artifact(*args, **kwargs):
        calls["predict"] += 1
        raise AssertionError("GET /demo-cases must not execute inference")

    monkeypatch.setattr(api_module, "predict_with_artifact", fake_predict_with_artifact)
    app = create_app(
        artifact_loader=lambda: (FakeEstimator(), metadata),
        explainability_loader=explainability_loader,
        demo_cases_loader=demo_cases_loader,
    )

    with TestClient(app) as client:
        response = client.get("/demo-cases")

    payload = response.json()
    assert response.status_code == 200
    assert set(payload) == EXPECTED_DEMO_CASES_FIELDS
    assert payload["case_count"] == DEMO_CASE_COUNT
    assert payload["malignant_case_count"] == DEMO_MALIGNANT_CASE_COUNT
    assert payload["benign_case_count"] == DEMO_BENIGN_CASE_COUNT
    assert payload["selection_rule"] == DEMO_SELECTION_RULE
    assert payload["used_for_training"] is False
    assert payload["used_for_model_selection"] is False
    assert payload["official_holdout_accuracy"] == DEMO_OFFICIAL_HOLDOUT_ACCURACY
    assert payload["feature_names"] == WDBC_FEATURE_NAMES
    assert len(payload["cases"]) == DEMO_CASE_COUNT
    assert len(payload["cases"][0]["features"]) == 30
    assert payload["disclaimer"] == ACADEMIC_DISCLAIMER
    assert "artifacts" not in str(payload)
    assert "reports" not in str(payload)
    assert calls["predict"] == 0


def test_endpoints_return_503_without_lifespan_state(
    metadata,
    explainability_loader,
    demo_cases_loader,
) -> None:
    app = create_app(
        artifact_loader=lambda: (FakeEstimator(), metadata),
        explainability_loader=explainability_loader,
        demo_cases_loader=demo_cases_loader,
    )
    client = TestClient(app)

    assert client.get("/health").status_code == 503
    assert client.get("/model").status_code == 503
    assert client.get("/explainability").status_code == 503
    assert client.get("/explainability/plot").status_code == 503
    assert client.get("/demo-cases").status_code == 503
    assert client.post(
        "/predict",
        json={"features": {feature: 1.0 for feature in WDBC_FEATURE_NAMES}},
    ).status_code == 503


def test_import_does_not_load_artifact_or_train(monkeypatch, tmp_path) -> None:
    calls = {"load": 0}

    def fake_load_model_artifact():
        calls["load"] += 1
        return FakeEstimator(), {}

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(api_module, "load_model_artifact", fake_load_model_artifact)
    importlib.reload(api_module)

    assert calls["load"] == 0
    assert not hasattr(api_module.app.state, "estimator")


def test_api_modules_do_not_contain_forbidden_operations() -> None:
    source = (
        inspect.getsource(api_module)
        + inspect.getsource(api_schemas_module)
        + inspect.getsource(explainability_artifacts_module)
    )
    forbidden_terms = [
        ".fit(",
        "permutation_importance(",
        "compute_cross_validated_permutation_importance(",
        "run_explainability_once(",
        "load_wdbc_data",
        "split_development_test",
        "evaluate_final_holdout",
        "GridSearchCV",
        "joblib.dump",
        "build_selected_estimator",
        "build_and_persist_final_artifact",
    ]

    for term in forbidden_terms:
        assert term not in source


def _load_once(calls: dict[str, int], metadata: dict[str, object]) -> tuple[FakeEstimator, dict]:
    calls["artifact"] += 1
    return FakeEstimator(), metadata


def _load_explainability_once(
    calls: dict[str, int],
    payload: dict[str, object],
) -> tuple[dict[str, object], bytes]:
    calls["explainability"] += 1
    return payload, PNG_SIGNATURE + b"plot"


def _load_demo_cases_once(
    calls: dict[str, int],
    payload: dict[str, object],
) -> dict[str, object]:
    calls["demo"] += 1
    return payload
