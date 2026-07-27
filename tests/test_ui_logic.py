import base64
import json

import pandas as pd
import pytest

from femhealth.data import WDBC_FEATURE_NAMES
from femhealth.ui_logic import (
    build_confusion_matrix,
    build_demo_feature_table,
    build_demo_scoreboard,
    build_explainability_feature_table,
    build_explainability_fold_table,
    compare_demo_prediction,
    deserialize_demo_progress,
    format_decimal_pt_br,
    format_probability,
    model_variant_pt_br,
    prediction_class_pt_br,
    reference_class_pt_br,
    serialize_demo_progress,
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


def test_reference_class_pt_br_translates_classes() -> None:
    assert reference_class_pt_br("malignant") == "Maligno"
    assert reference_class_pt_br("benign") == "Benigno"


def test_reference_class_pt_br_rejects_unknown_class() -> None:
    with pytest.raises(ValueError, match="Unexpected reference class"):
        reference_class_pt_br("unknown")


def test_build_demo_feature_table_uses_canonical_order_and_translations() -> None:
    features = _demo_features()

    table = build_demo_feature_table(features)

    assert table.columns.tolist() == ["Número", "Variável", "Chave canônica", "Valor"]
    assert table.shape == (30, 4)
    assert table["Número"].tolist() == list(range(1, 31))
    assert table["Chave canônica"].tolist() == WDBC_FEATURE_NAMES
    assert table["Variável"].iloc[0] == "Raio médio"
    assert table["Valor"].tolist() == [float(index + 1) for index in range(30)]


def test_build_demo_feature_table_rejects_missing_feature() -> None:
    features = _demo_features()
    del features[WDBC_FEATURE_NAMES[0]]

    with pytest.raises(ValueError, match="feature names"):
        build_demo_feature_table(features)


def test_build_demo_feature_table_rejects_extra_feature() -> None:
    features = _demo_features()
    features["extra"] = 1.0

    with pytest.raises(ValueError, match="feature names"):
        build_demo_feature_table(features)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, "1.0"])
def test_build_demo_feature_table_rejects_invalid_values(value) -> None:
    features = _demo_features()
    features[WDBC_FEATURE_NAMES[0]] = value

    with pytest.raises(ValueError, match="finite numbers"):
        build_demo_feature_table(features)


def test_compare_demo_prediction_returns_label_match() -> None:
    assert compare_demo_prediction(0, 0) is True
    assert compare_demo_prediction(0, 1) is False


@pytest.mark.parametrize(
    ("reference_label", "predicted_label"),
    [(2, 0), (0, 2), (True, 0), (0, False)],
)
def test_compare_demo_prediction_rejects_invalid_labels(reference_label, predicted_label) -> None:
    with pytest.raises(ValueError):
        compare_demo_prediction(reference_label, predicted_label)


def test_build_demo_scoreboard_handles_empty_session() -> None:
    assert build_demo_scoreboard({}) == {
        "tested": 0,
        "correct": 0,
        "divergences": 0,
        "accuracy": "—",
    }


def test_build_demo_scoreboard_counts_unique_cases() -> None:
    scoreboard = build_demo_scoreboard(
        {
            "demo-01": True,
            "demo-04": False,
        }
    )

    assert scoreboard == {
        "tested": 2,
        "correct": 1,
        "divergences": 1,
        "accuracy": 0.5,
    }


def test_serialize_and_deserialize_demo_progress_round_trip() -> None:
    results = {
        "demo-01": _demo_result("demo-01", predicted_label=0),
        "demo-04": _demo_result("demo-04", predicted_label=0),
    }

    encoded = serialize_demo_progress(results, selected_case_id="demo-04")
    restored_results, selected_case_id = deserialize_demo_progress(encoded, _demo_cases())

    assert selected_case_id == "demo-04"
    assert restored_results["demo-01"]["sample_index"] == 256
    assert restored_results["demo-01"]["reference_label"] == 0
    assert restored_results["demo-01"]["correct"] is True
    assert restored_results["demo-04"]["sample_index"] == 363
    assert restored_results["demo-04"]["reference_label"] == 1
    assert restored_results["demo-04"]["correct"] is False
    assert restored_results["demo-04"]["probability_malignant"] == 0.62


def test_serialize_demo_progress_is_deterministic() -> None:
    results = {
        "demo-04": _demo_result("demo-04", predicted_label=0),
        "demo-01": _demo_result("demo-01", predicted_label=0),
    }

    assert serialize_demo_progress(results, "demo-04") == serialize_demo_progress(
        dict(reversed(list(results.items()))),
        "demo-04",
    )


def test_demo_progress_accepts_eight_results() -> None:
    results = {
        f"demo-{position:02d}": _demo_result(
            f"demo-{position:02d}",
            predicted_label=_demo_cases()[position - 1]["reference_label"],
        )
        for position in range(1, 9)
    }

    encoded = serialize_demo_progress(results, selected_case_id="demo-08")
    restored_results, selected_case_id = deserialize_demo_progress(encoded, _demo_cases())

    assert selected_case_id == "demo-08"
    assert len(restored_results) == 8


def test_serialize_demo_progress_rejects_more_than_eight_results() -> None:
    results = {f"demo-{position:02d}": _demo_result("demo-01") for position in range(1, 10)}

    with pytest.raises(ValueError, match="too many"):
        serialize_demo_progress(results, selected_case_id="demo-01")


def test_deserialize_demo_progress_rejects_more_than_eight_results() -> None:
    payload = _progress_payload()
    payload["results"] = {
        f"demo-{position:02d}": _persisted_result()
        for position in range(1, 10)
    }

    with pytest.raises(ValueError, match="too many"):
        deserialize_demo_progress(_encode_progress_payload(payload), _demo_cases())


def test_deserialize_demo_progress_rejects_unknown_case() -> None:
    payload = _progress_payload()
    payload["results"]["demo-99"] = _persisted_result()

    with pytest.raises(ValueError, match="case id"):
        deserialize_demo_progress(_encode_progress_payload(payload), _demo_cases())


def test_deserialize_demo_progress_rejects_unknown_selected_case() -> None:
    payload = _progress_payload(selected_case_id="demo-99")

    with pytest.raises(ValueError, match="selected"):
        deserialize_demo_progress(_encode_progress_payload(payload), _demo_cases())


def test_deserialize_demo_progress_rejects_unknown_version() -> None:
    payload = _progress_payload()
    payload["version"] = 2

    with pytest.raises(ValueError, match="version"):
        deserialize_demo_progress(_encode_progress_payload(payload), _demo_cases())


def test_deserialize_demo_progress_rejects_invalid_json() -> None:
    encoded = base64.urlsafe_b64encode(b"not-json").decode("ascii")

    with pytest.raises(ValueError, match="JSON"):
        deserialize_demo_progress(encoded, _demo_cases())


def test_deserialize_demo_progress_rejects_invalid_base64() -> None:
    with pytest.raises(ValueError, match="base64"):
        deserialize_demo_progress("!!!!", _demo_cases())


@pytest.mark.parametrize("invalid_label", [2, 0.0, True])
def test_deserialize_demo_progress_rejects_invalid_predicted_label(invalid_label) -> None:
    payload = _progress_payload()
    payload["results"]["demo-01"]["predicted_label"] = invalid_label

    with pytest.raises(ValueError, match="predicted label"):
        deserialize_demo_progress(_encode_progress_payload(payload), _demo_cases())


def test_deserialize_demo_progress_rejects_inconsistent_predicted_class() -> None:
    payload = _progress_payload()
    payload["results"]["demo-01"]["predicted_label"] = 0
    payload["results"]["demo-01"]["predicted_class"] = "benign"

    with pytest.raises(ValueError, match="inconsistent"):
        deserialize_demo_progress(_encode_progress_payload(payload), _demo_cases())


@pytest.mark.parametrize("value", [-0.01, 1.01])
def test_deserialize_demo_progress_rejects_probability_outside_range(value) -> None:
    payload = _progress_payload()
    payload["results"]["demo-01"]["probability_malignant"] = value
    payload["results"]["demo-01"]["probability_benign"] = 1 - value

    with pytest.raises(ValueError, match="probability"):
        deserialize_demo_progress(_encode_progress_payload(payload), _demo_cases())


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_deserialize_demo_progress_rejects_non_finite_probabilities(value) -> None:
    payload = _progress_payload()
    payload["results"]["demo-01"]["probability_malignant"] = value

    with pytest.raises(ValueError, match="probability"):
        deserialize_demo_progress(_encode_progress_payload(payload), _demo_cases())


def test_deserialize_demo_progress_rejects_invalid_probability_sum() -> None:
    payload = _progress_payload()
    payload["results"]["demo-01"]["probability_malignant"] = 0.60
    payload["results"]["demo-01"]["probability_benign"] = 0.30

    with pytest.raises(ValueError, match="sum"):
        deserialize_demo_progress(_encode_progress_payload(payload), _demo_cases())


def test_deserialize_demo_progress_rejects_unexpected_threshold() -> None:
    payload = _progress_payload()
    payload["results"]["demo-01"]["threshold"] = 0.50

    with pytest.raises(ValueError, match="threshold"):
        deserialize_demo_progress(_encode_progress_payload(payload), _demo_cases())


def test_deserialize_demo_progress_rejects_missing_key() -> None:
    payload = _progress_payload()
    del payload["results"]["demo-01"]["threshold"]

    with pytest.raises(ValueError, match="missing"):
        deserialize_demo_progress(_encode_progress_payload(payload), _demo_cases())


def test_deserialize_demo_progress_rejects_additional_key() -> None:
    payload = _progress_payload()
    payload["results"]["demo-01"]["correct"] = True

    with pytest.raises(ValueError, match="unexpected"):
        deserialize_demo_progress(_encode_progress_payload(payload), _demo_cases())


def test_serialize_demo_progress_does_not_persist_features_reference_or_correct() -> None:
    result = _demo_result("demo-01")
    result["features"] = _demo_features()

    with pytest.raises(ValueError, match="unexpected"):
        serialize_demo_progress({"demo-01": result}, selected_case_id="demo-01")

    result = _demo_result("demo-01")
    encoded = serialize_demo_progress({"demo-01": result}, selected_case_id="demo-01")
    raw_payload = _decode_progress_payload(encoded)
    raw_result = raw_payload["results"]["demo-01"]

    assert "features" not in raw_result
    assert "reference_label" not in raw_result
    assert "reference_class" not in raw_result
    assert "sample_index" not in raw_result
    assert "correct" not in raw_result


def test_deserialize_demo_progress_recalculates_correct_from_official_case() -> None:
    payload = _progress_payload(selected_case_id="demo-04")
    payload["results"] = {
        "demo-04": _persisted_result(
            predicted_label=0,
            predicted_class="malignant",
            probability_malignant=0.62,
            probability_benign=0.38,
        )
    }

    restored_results, _ = deserialize_demo_progress(
        _encode_progress_payload(payload),
        _demo_cases(),
    )

    assert restored_results["demo-04"]["reference_label"] == 1
    assert restored_results["demo-04"]["correct"] is False


def test_deserialize_demo_progress_discards_manually_altered_result_safely() -> None:
    payload = _progress_payload()
    payload["results"]["demo-01"]["predicted_class"] = "malignant<script>"

    with pytest.raises(ValueError, match="predicted class"):
        deserialize_demo_progress(_encode_progress_payload(payload), _demo_cases())


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


def _demo_features() -> dict[str, float]:
    return {
        feature_name: float(index + 1)
        for index, feature_name in enumerate(WDBC_FEATURE_NAMES)
    }


def _demo_cases() -> list[dict]:
    sample_indices = [256, 428, 501, 363, 564, 464, 358, 343]
    reference_labels = [0, 1, 0, 1, 0, 1, 1, 0]
    return [
        {
            "case_id": f"demo-{position:02d}",
            "sample_index": sample_index,
            "reference_label": reference_label,
            "reference_class": "malignant" if reference_label == 0 else "benign",
            "features": _demo_features(),
        }
        for position, (sample_index, reference_label) in enumerate(
            zip(sample_indices, reference_labels, strict=True),
            start=1,
        )
    ]


def _demo_result(case_id: str, predicted_label: int = 0) -> dict:
    official_case = {case["case_id"]: case for case in _demo_cases()}[case_id]
    predicted_class = "malignant" if predicted_label == 0 else "benign"
    probability_malignant = 0.62 if case_id == "demo-04" and predicted_label == 0 else 0.99
    probability_benign = 1 - probability_malignant
    if predicted_label == 1:
        probability_malignant = 0.01
        probability_benign = 0.99

    return {
        "case_id": case_id,
        "sample_index": official_case["sample_index"],
        "reference_label": official_case["reference_label"],
        "reference_class": official_case["reference_class"],
        "predicted_label": predicted_label,
        "predicted_class": predicted_class,
        "probability_malignant": probability_malignant,
        "probability_benign": probability_benign,
        "threshold": 0.51,
        "correct": official_case["reference_label"] == predicted_label,
    }


def _persisted_result(
    predicted_label: int = 0,
    predicted_class: str = "malignant",
    probability_malignant: float = 0.99,
    probability_benign: float = 0.01,
) -> dict:
    return {
        "predicted_label": predicted_label,
        "predicted_class": predicted_class,
        "probability_malignant": probability_malignant,
        "probability_benign": probability_benign,
        "threshold": 0.51,
    }


def _progress_payload(selected_case_id: str | None = "demo-01") -> dict:
    return {
        "version": 1,
        "selected_case_id": selected_case_id,
        "results": {
            "demo-01": _persisted_result(),
        },
    }


def _encode_progress_payload(payload: dict) -> str:
    raw_json = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw_json).decode("ascii").rstrip("=")


def _decode_progress_payload(encoded: str) -> dict:
    padded = encoded + ("=" * (-len(encoded) % 4))
    return json.loads(base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8"))
