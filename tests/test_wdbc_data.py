import pandas as pd
import pytest

from femhealth.data import (
    WDBC_CLASS_DISTRIBUTION,
    WDBC_CLASS_NAMES,
    WDBC_FEATURE_NAMES,
    load_wdbc_data,
    validate_wdbc_data,
)


def test_load_wdbc_data_shape() -> None:
    X, y = load_wdbc_data()

    assert X.shape == (569, 30)
    assert y.shape == (569,)
    assert y.name == "diagnosis"


def test_wdbc_feature_names_and_order() -> None:
    X, _ = load_wdbc_data()

    assert list(X.columns) == WDBC_FEATURE_NAMES


def test_wdbc_classes_and_distribution() -> None:
    _, y = load_wdbc_data()

    assert set(y.unique()) == set(WDBC_CLASS_NAMES)
    assert y.value_counts().sort_index().to_dict() == WDBC_CLASS_DISTRIBUTION


def test_wdbc_has_no_null_values() -> None:
    X, y = load_wdbc_data()

    assert not X.isna().any().any()
    assert not y.isna().any()


def test_wdbc_features_are_numeric() -> None:
    X, _ = load_wdbc_data()

    assert all(pd.api.types.is_numeric_dtype(dtype) for dtype in X.dtypes)


def test_validate_wdbc_data_rejects_missing_column() -> None:
    X, y = load_wdbc_data()
    invalid_X = X.drop(columns=[WDBC_FEATURE_NAMES[0]])

    with pytest.raises(ValueError, match="Unexpected feature shape"):
        validate_wdbc_data(invalid_X, y)


def test_validate_wdbc_data_rejects_null_value() -> None:
    X, y = load_wdbc_data()
    invalid_X = X.copy()
    invalid_X.loc[invalid_X.index[0], WDBC_FEATURE_NAMES[0]] = None

    with pytest.raises(ValueError, match="Null values found"):
        validate_wdbc_data(invalid_X, y)


def test_validate_wdbc_data_rejects_invalid_class() -> None:
    X, y = load_wdbc_data()
    invalid_y = y.copy()
    invalid_y.iloc[0] = 2

    with pytest.raises(ValueError, match="Unexpected target classes"):
        validate_wdbc_data(X, invalid_y)


def test_validate_wdbc_data_rejects_unexpected_target_name() -> None:
    X, y = load_wdbc_data()
    invalid_y = y.rename("target")

    with pytest.raises(ValueError, match="Unexpected target name"):
        validate_wdbc_data(X, invalid_y)


def test_validate_wdbc_data_rejects_index_mismatch() -> None:
    X, y = load_wdbc_data()
    invalid_y = y.copy()
    invalid_y.index = invalid_y.index + 1

    with pytest.raises(ValueError, match="Index mismatch"):
        validate_wdbc_data(X, invalid_y)
