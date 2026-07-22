"""Reproducible development/test split for tabular data."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from femhealth.data import validate_wdbc_data

TEST_SIZE = 0.20
RANDOM_STATE = 42


def split_development_test(
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split validated WDBC data into development and final test sets."""
    validate_wdbc_data(X, y)

    X_development, X_test, y_development, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    return X_development, X_test, y_development, y_test
