"""Dataset loading and validation for WDBC."""

from __future__ import annotations

import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.datasets import load_breast_cancer

WDBC_SAMPLE_COUNT = 569
WDBC_FEATURE_COUNT = 30
WDBC_TARGET_NAME = "diagnosis"
WDBC_CLASS_NAMES = {
    0: "malignant",
    1: "benign",
}
WDBC_CLASS_DISTRIBUTION = {
    0: 212,
    1: 357,
}
WDBC_FEATURE_NAMES = [
    "mean radius",
    "mean texture",
    "mean perimeter",
    "mean area",
    "mean smoothness",
    "mean compactness",
    "mean concavity",
    "mean concave points",
    "mean symmetry",
    "mean fractal dimension",
    "radius error",
    "texture error",
    "perimeter error",
    "area error",
    "smoothness error",
    "compactness error",
    "concavity error",
    "concave points error",
    "symmetry error",
    "fractal dimension error",
    "worst radius",
    "worst texture",
    "worst perimeter",
    "worst area",
    "worst smoothness",
    "worst compactness",
    "worst concavity",
    "worst concave points",
    "worst symmetry",
    "worst fractal dimension",
]


def load_wdbc_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load the WDBC dataset from scikit-learn."""
    dataset = load_breast_cancer(as_frame=True)
    X = dataset.data
    y = dataset.target.rename(WDBC_TARGET_NAME)

    validate_wdbc_data(X, y)

    return X, y


def validate_wdbc_data(X: pd.DataFrame, y: pd.Series) -> None:
    """Validate the expected WDBC dataset contract."""
    if not isinstance(X, pd.DataFrame):
        raise ValueError("X must be a DataFrame")

    if not isinstance(y, pd.Series):
        raise ValueError("y must be a Series")

    if X.shape != (WDBC_SAMPLE_COUNT, WDBC_FEATURE_COUNT):
        raise ValueError("Unexpected feature shape")

    if len(y) != WDBC_SAMPLE_COUNT:
        raise ValueError("Unexpected target length")

    if list(X.columns) != WDBC_FEATURE_NAMES:
        raise ValueError("Unexpected feature columns")

    if not X.index.equals(y.index):
        raise ValueError("Index mismatch")

    if X.isna().any().any() or y.isna().any():
        raise ValueError("Null values found")

    if not all(is_numeric_dtype(dtype) for dtype in X.dtypes):
        raise ValueError("Features must be numeric")

    if set(y.unique()) != set(WDBC_CLASS_NAMES):
        raise ValueError("Unexpected target classes")

    if y.value_counts().sort_index().to_dict() != WDBC_CLASS_DISTRIBUTION:
        raise ValueError("Unexpected class distribution")

    if y.name != WDBC_TARGET_NAME:
        raise ValueError("Unexpected target name")