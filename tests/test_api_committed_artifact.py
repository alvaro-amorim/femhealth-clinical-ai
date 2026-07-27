import pytest
from fastapi.testclient import TestClient

from femhealth.api import app
from femhealth.api_schemas import ACADEMIC_DISCLAIMER
from femhealth.data import WDBC_FEATURE_NAMES
from femhealth.explainability_artifacts import PNG_SIGNATURE
from femhealth.final_selection import SELECTED_THRESHOLD


def test_api_uses_committed_artifact_for_synthetic_prediction() -> None:
    features = {feature_name: 1.0 for feature_name in WDBC_FEATURE_NAMES}

    with TestClient(app) as client:
        health_response = client.get("/health")
        model_response = client.get("/model")
        explainability_response = client.get("/explainability")
        explainability_plot_response = client.get("/explainability/plot")
        prediction_response = client.post("/predict", json={"features": features})

    assert health_response.status_code == 200
    assert model_response.status_code == 200
    assert explainability_response.status_code == 200
    assert explainability_plot_response.status_code == 200
    assert prediction_response.status_code == 200
    assert health_response.json()["model_loaded"] is True
    assert model_response.json()["feature_names"] == WDBC_FEATURE_NAMES
    assert len(explainability_response.json()["features"]) == 30
    assert len(explainability_response.json()["fold_scores"]) == 5
    assert explainability_plot_response.content.startswith(PNG_SIGNATURE)

    prediction = prediction_response.json()
    assert 0 <= prediction["probability_malignant"] <= 1
    assert 0 <= prediction["probability_benign"] <= 1
    assert prediction["probability_malignant"] + prediction["probability_benign"] == pytest.approx(
        1.0
    )
    assert prediction["predicted_label"] in {0, 1}
    assert prediction["predicted_class"] in {"malignant", "benign"}
    assert prediction["threshold"] == SELECTED_THRESHOLD
    assert prediction["disclaimer"] == ACADEMIC_DISCLAIMER
