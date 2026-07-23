"""Inference contract for the persisted FemHealth model artifact."""

from __future__ import annotations

import pandas as pd
from pandas.api.types import is_numeric_dtype

from femhealth.data import WDBC_FEATURE_COUNT, WDBC_FEATURE_NAMES
from femhealth.model_evaluation import BENIGN_LABEL, MALIGNANT_LABEL

CLASS_NAMES = {
    MALIGNANT_LABEL: "malignant",
    BENIGN_LABEL: "benign",
}


def predict_with_artifact(
    X: pd.DataFrame,
    estimator,
    metadata: dict,
) -> pd.DataFrame:
    """Predict malignant probabilities with an already loaded artifact estimator."""
    _validate_inference_input(X, metadata)

    probability_malignant = pd.Series(
        _predict_malignant_probability(estimator, X),
        index=X.index,
        name="probability_malignant",
    )
    threshold = metadata["threshold"]
    predicted_label = pd.Series(
        [
            MALIGNANT_LABEL if probability >= threshold else BENIGN_LABEL
            for probability in probability_malignant
        ],
        index=X.index,
        name="predicted_label",
    )

    return pd.DataFrame(
        {
            "probability_malignant": probability_malignant,
            "probability_benign": 1 - probability_malignant,
            "predicted_label": predicted_label,
            "predicted_class": predicted_label.map(CLASS_NAMES),
        },
        index=X.index,
    )


def _validate_inference_input(X: pd.DataFrame, metadata: dict) -> None:
    if not isinstance(X, pd.DataFrame):
        raise ValueError("X must be a DataFrame")

    if X.empty:
        raise ValueError("X must contain at least one row")

    if X.shape[1] != WDBC_FEATURE_COUNT:
        raise ValueError("Unexpected feature count")

    if list(X.columns) != WDBC_FEATURE_NAMES:
        raise ValueError("Unexpected feature columns")

    if metadata.get("feature_names") != WDBC_FEATURE_NAMES:
        raise ValueError("Unexpected metadata feature names")

    if X.isna().any().any():
        raise ValueError("Null values found")

    if not all(is_numeric_dtype(dtype) for dtype in X.dtypes):
        raise ValueError("Features must be numeric")


def _predict_malignant_probability(estimator, X: pd.DataFrame):
    probabilities = estimator.predict_proba(X)
    malignant_index = list(estimator.classes_).index(MALIGNANT_LABEL)
    return probabilities[:, malignant_index]
