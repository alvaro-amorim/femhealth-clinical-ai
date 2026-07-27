import pandas as pd
import pytest

from femhealth.data import WDBC_FEATURE_NAMES
from femhealth.ui_logic import (
    build_confusion_matrix,
    build_explainability_feature_table,
    build_explainability_fold_table,
    format_decimal_pt_br,
    format_probability,
    model_variant_pt_br,
    prediction_class_pt_br,
    validate_api_feature_contract,
)


def test_validate_api_feature_contract_accepts_canonical_features() -> None:
    validate_api_feature_contract(WDBC_FEATURE_NAMES)


def test_validate_api_feature_contract_rejects_missing_feature() -> None:
    with pytest.raises(ValueError, match="Unexpected feature count"):
        validate_api_feature_contract(WDBC_FEATURE_NAMES[:-1])


def test_validate_api_feature_contract_rejects_extra_feature() -> None:
    with pytest.raises(ValueError, match="Unexpected feature count"):
        validate_api_feature_contract([*WDBC_FEATURE_NAMES, "extra"])


def test_validate_api_feature_contract_rejects_incorrect_order() -> None:
    with pytest.raises(ValueError, match="Unexpected feature order"):
        validate_api_feature_contract(list(reversed(WDBC_FEATURE_NAMES)))


def test_validate_api_feature_contract_rejects_duplicate() -> None:
    duplicated = [*WDBC_FEATURE_NAMES[:-1], WDBC_FEATURE_NAMES[0]]

    with pytest.raises(ValueError, match="Duplicated feature names"):
        validate_api_feature_contract(duplicated)


def test_format_probability_uses_pt_br_percent() -> None:
    assert format_probability(0.9762) == "97,62%"


@pytest.mark.parametrize("value", [-0.01, 1.01])
def test_format_probability_rejects_invalid_probability(value) -> None:
    with pytest.raises(ValueError, match="Probability must be between 0 and 1"):
        format_probability(value)


def test_format_decimal_pt_br_uses_comma() -> None:
    assert format_decimal_pt_br(0.51) == "0,51"
    assert format_decimal_pt_br(1.0) == "1,00"


def test_format_decimal_pt_br_accepts_zero_decimal_places() -> None:
    assert format_decimal_pt_br(1.49, decimal_places=0) == "1"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True])
def test_format_decimal_pt_br_rejects_invalid_values(value) -> None:
    with pytest.raises(ValueError, match="Value must be a finite number"):
        format_decimal_pt_br(value)


def test_format_decimal_pt_br_rejects_negative_decimal_places() -> None:
    with pytest.raises(ValueError, match="Decimal places must be greater than or equal to zero"):
        format_decimal_pt_br(1.0, decimal_places=-1)


def test_model_variant_pt_br_translates_without_exposing_technical_key() -> None:
    translated = model_variant_pt_br("svm_sigmoid")

    assert translated == "SVM calibrado por sigmoid"
    assert translated != "svm_sigmoid"


def test_model_variant_pt_br_rejects_unknown_variant() -> None:
    with pytest.raises(ValueError, match="Unexpected selected variant"):
        model_variant_pt_br("unknown")


def test_prediction_class_pt_br_translates_classes() -> None:
    assert prediction_class_pt_br("malignant") == "Padrão classificado como maligno"
    assert prediction_class_pt_br("benign") == "Padrão classificado como benigno"


def test_prediction_class_pt_br_rejects_unknown_class() -> None:
    with pytest.raises(ValueError, match="Unexpected predicted class"):
        prediction_class_pt_br("unknown")


def test_build_confusion_matrix_uses_persisted_counts() -> None:
    final_metrics = {
        "true_malignant": 41,
        "false_negative_malignant": 1,
        "false_positive_malignant": 2,
        "true_benign": 70,
    }

    matrix = build_confusion_matrix(final_metrics)

    expected = pd.DataFrame(
        {
            "Previsto: maligno": [41, 2],
            "Previsto: benigno": [1, 70],
        },
        index=["Real: maligno", "Real: benigno"],
    )
    assert matrix.equals(expected)


def test_build_explainability_feature_table_translates_and_orders_by_rank() -> None:
    features = [
        _feature(rank=2, feature_name="mean texture"),
        _feature(rank=1, feature_name="mean radius"),
    ]

    table = build_explainability_feature_table(features)

    assert table.columns.tolist() == [
        "Posição no ranking",
        "Variável",
        "Chave canônica",
        "Importância média",
        "Desvio-padrão",
        "Fração positiva",
    ]
    assert table["Posição no ranking"].tolist() == [1, 2]
    assert table["Variável"].tolist() == ["Raio médio", "Textura média"]
    assert table["Chave canônica"].tolist() == ["mean radius", "mean texture"]
    assert table["Importância média"].tolist() == [0.1, 0.1]


def test_build_explainability_feature_table_rejects_duplicate_rank() -> None:
    features = [
        _feature(rank=1, feature_name="mean radius"),
        _feature(rank=1, feature_name="mean texture"),
    ]

    with pytest.raises(ValueError, match="Duplicated explainability ranks"):
        build_explainability_feature_table(features)


def test_build_explainability_feature_table_rejects_unknown_feature() -> None:
    features = [_feature(rank=1, feature_name="unknown")]

    with pytest.raises(ValueError, match="Feature without Portuguese label"):
        build_explainability_feature_table(features)


def test_build_explainability_fold_table_orders_folds() -> None:
    fold_scores = [
        _fold_score(fold=2),
        _fold_score(fold=1),
    ]

    table = build_explainability_fold_table(fold_scores)

    assert table.columns.tolist() == [
        "Fold",
        "Amostras de treinamento",
        "Amostras de validação",
        "Malignos na validação",
        "Benignos na validação",
        "ROC AUC maligno",
    ]
    assert table["Fold"].tolist() == [1, 2]
    assert table["ROC AUC maligno"].tolist() == [0.99, 0.99]


def test_build_explainability_fold_table_rejects_missing_values() -> None:
    with pytest.raises(ValueError, match="Missing explainability fold values"):
        build_explainability_fold_table([{"fold": 1}])


def _feature(rank: int, feature_name: str) -> dict:
    return {
        "rank": rank,
        "feature_name": feature_name,
        "mean_importance": 0.1,
        "std_importance": 0.01,
        "positive_fraction": 0.8,
    }


def _fold_score(fold: int) -> dict:
    return {
        "fold": fold,
        "train_sample_count": 364,
        "validation_sample_count": 91,
        "validation_malignant_count": 34,
        "validation_benign_count": 57,
        "baseline_roc_auc": 0.99,
    }
