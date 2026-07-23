import pandas as pd
import pytest

from femhealth.data import WDBC_FEATURE_NAMES
from femhealth.inference import predict_with_artifact
from femhealth.model_artifact import METADATA_PATH, MODEL_PATH, _sha256_file, load_model_artifact


def test_committed_model_artifact_loads_and_predicts_on_synthetic_input() -> None:
    assert MODEL_PATH.exists()
    assert METADATA_PATH.exists()

    estimator, metadata = load_model_artifact(MODEL_PATH, METADATA_PATH)

    assert _sha256_file(MODEL_PATH) == metadata["model_sha256"]

    X = pd.DataFrame(
        [[1.0] * len(WDBC_FEATURE_NAMES)],
        index=["synthetic_sample"],
        columns=WDBC_FEATURE_NAMES,
    )

    predictions = predict_with_artifact(X, estimator, metadata)

    assert predictions.index.equals(X.index)
    assert list(predictions.columns) == [
        "probability_malignant",
        "probability_benign",
        "predicted_label",
        "predicted_class",
    ]
    assert predictions["probability_malignant"].between(0, 1).all()
    assert predictions["probability_benign"].between(0, 1).all()
    probability_sum = (
        predictions.loc["synthetic_sample", "probability_malignant"]
        + predictions.loc["synthetic_sample", "probability_benign"]
    )
    assert probability_sum == pytest.approx(1.0)
    assert predictions.loc["synthetic_sample", "predicted_label"] in {0, 1}
    assert predictions.loc["synthetic_sample", "predicted_class"] in {"malignant", "benign"}
