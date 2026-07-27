"""Pure presentation logic for the Streamlit interface."""

from __future__ import annotations

import base64
import binascii
import json
import math
from typing import Any

import pandas as pd

from femhealth.ui_labels import FEATURE_LABELS_PT_BR, translate_feature_name

DEMO_PROGRESS_VERSION = 1
DEMO_PROGRESS_THRESHOLD = 0.51
DEMO_PROGRESS_MAX_RESULTS = 8
DEMO_PROGRESS_PROBABILITY_TOLERANCE = 1e-6
_PREDICTED_LABEL_TO_CLASS = {
    0: "malignant",
    1: "benign",
}
_DEMO_PROGRESS_TOP_LEVEL_KEYS = {
    "version",
    "selected_case_id",
    "results",
}
_DEMO_PROGRESS_RESULT_KEYS = {
    "predicted_label",
    "predicted_class",
    "probability_malignant",
    "probability_benign",
    "threshold",
}
_DEMO_SESSION_RESULT_KEYS = _DEMO_PROGRESS_RESULT_KEYS | {
    "case_id",
    "sample_index",
    "reference_label",
    "reference_class",
    "correct",
}


def validate_api_feature_contract(feature_names: list[str]) -> None:
    """Validate that API feature metadata matches the UI translation contract."""
    expected_features = list(FEATURE_LABELS_PT_BR)

    if len(feature_names) != len(expected_features):
        raise ValueError("Unexpected feature count")

    if len(set(feature_names)) != len(feature_names):
        raise ValueError("Duplicated feature names")

    if set(feature_names) != set(expected_features):
        raise ValueError("Unexpected feature names")

    if feature_names != expected_features:
        raise ValueError("Unexpected feature order")


def format_probability(value: float) -> str:
    """Format a probability as a Brazilian Portuguese percentage."""
    if not 0 <= value <= 1:
        raise ValueError("Probability must be between 0 and 1")

    return f"{value * 100:.2f}%".replace(".", ",")


def format_decimal_pt_br(
    value: float,
    decimal_places: int = 2,
) -> str:
    """Format a finite decimal number using Brazilian Portuguese separators."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError("Value must be a finite number")

    if not math.isfinite(float(value)):
        raise ValueError("Value must be a finite number")

    if decimal_places < 0:
        raise ValueError("Decimal places must be greater than or equal to zero")

    return f"{value:.{decimal_places}f}".replace(".", ",")


def model_variant_pt_br(selected_variant: str) -> str:
    """Translate selected model variant identifiers for presentation."""
    if selected_variant == "svm_sigmoid":
        return "SVM calibrado por sigmoid"

    raise ValueError("Unexpected selected variant")


def prediction_class_pt_br(predicted_class: str) -> str:
    """Translate API predicted classes for presentation."""
    if predicted_class == "malignant":
        return "Padrão classificado como maligno"

    if predicted_class == "benign":
        return "Padrão classificado como benigno"

    raise ValueError("Unexpected predicted class")


def reference_class_pt_br(reference_class: str) -> str:
    """Translate reference classes for demonstration cases."""
    if reference_class == "malignant":
        return "Maligno"

    if reference_class == "benign":
        return "Benigno"

    raise ValueError("Unexpected reference class")


def build_demo_feature_table(features: dict[str, float]) -> pd.DataFrame:
    """Build a display table for one demonstration case."""
    if not isinstance(features, dict):
        raise ValueError("Demo features must be an object")

    expected_features = list(FEATURE_LABELS_PT_BR)
    if list(features) != expected_features:
        raise ValueError("Unexpected demo feature names")

    rows = []
    for position, feature_name in enumerate(expected_features, start=1):
        value = features[feature_name]
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError("Demo feature values must be finite numbers")

        numeric_value = float(value)
        if not math.isfinite(numeric_value):
            raise ValueError("Demo feature values must be finite numbers")

        rows.append(
            {
                "Número": position,
                "Variável": translate_feature_name(feature_name),
                "Chave canônica": feature_name,
                "Valor": numeric_value,
            }
        )

    return pd.DataFrame(rows)


def compare_demo_prediction(
    reference_label: int,
    predicted_label: int,
) -> bool:
    """Compare reference and predicted labels for a demonstration case."""
    if isinstance(reference_label, bool) or reference_label not in {0, 1}:
        raise ValueError("Unexpected reference label")

    if isinstance(predicted_label, bool) or predicted_label not in {0, 1}:
        raise ValueError("Unexpected predicted label")

    return reference_label == predicted_label


def build_demo_scoreboard(case_results: dict[str, bool]) -> dict[str, int | float | str]:
    """Build unique-case scoreboard values for the current Streamlit session."""
    for result in case_results.values():
        if not isinstance(result, bool):
            raise ValueError("Demo scoreboard values must be booleans")

    tested = len(case_results)
    correct = sum(case_results.values())
    divergences = tested - correct
    accuracy = "—" if tested == 0 else correct / tested

    return {
        "tested": tested,
        "correct": correct,
        "divergences": divergences,
        "accuracy": accuracy,
    }


def serialize_demo_progress(
    results: dict[str, dict],
    selected_case_id: str | None,
) -> str:
    """Serialize demonstration progress into a compact URL-safe value."""
    if not isinstance(results, dict):
        raise ValueError("Demo progress results must be an object")

    if len(results) > DEMO_PROGRESS_MAX_RESULTS:
        raise ValueError("Demo progress has too many results")

    if selected_case_id is not None and not isinstance(selected_case_id, str):
        raise ValueError("Unexpected selected demo case id")

    persisted_results = {}
    for case_id in sorted(results):
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("Unexpected demo case id")

        result = results[case_id]
        persisted_results[case_id] = _sanitize_demo_progress_result(case_id, result)

    payload = {
        "version": DEMO_PROGRESS_VERSION,
        "selected_case_id": selected_case_id,
        "results": persisted_results,
    }
    raw_json = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    encoded = base64.urlsafe_b64encode(raw_json.encode("utf-8")).decode("ascii")
    return encoded.rstrip("=")


def deserialize_demo_progress(
    encoded_progress: str,
    demo_cases: list[dict],
) -> tuple[dict[str, dict], str | None]:
    """Deserialize and validate URL-persisted demonstration progress."""
    if not isinstance(encoded_progress, str) or not encoded_progress:
        raise ValueError("Demo progress must be a non-empty string")

    official_cases = _build_official_demo_case_index(demo_cases)
    payload = _decode_demo_progress_payload(encoded_progress)

    _validate_exact_keys(payload, _DEMO_PROGRESS_TOP_LEVEL_KEYS, "Demo progress")

    if payload["version"] != DEMO_PROGRESS_VERSION:
        raise ValueError("Unsupported demo progress version")

    selected_case_id = payload["selected_case_id"]
    if selected_case_id is not None:
        if not isinstance(selected_case_id, str) or selected_case_id not in official_cases:
            raise ValueError("Unexpected selected demo case id")

    raw_results = payload["results"]
    if not isinstance(raw_results, dict):
        raise ValueError("Demo progress results must be an object")

    if len(raw_results) > DEMO_PROGRESS_MAX_RESULTS:
        raise ValueError("Demo progress has too many results")

    reconstructed_results = {}
    for case_id, raw_result in raw_results.items():
        if not isinstance(case_id, str) or case_id not in official_cases:
            raise ValueError("Unexpected demo case id")

        persisted_result = _validate_persisted_demo_progress_result(raw_result)
        official_case = official_cases[case_id]
        predicted_label = persisted_result["predicted_label"]
        correct = compare_demo_prediction(official_case["reference_label"], predicted_label)
        reconstructed_results[case_id] = {
            "case_id": case_id,
            "sample_index": official_case["sample_index"],
            "reference_label": official_case["reference_label"],
            "reference_class": official_case["reference_class"],
            "predicted_label": predicted_label,
            "predicted_class": persisted_result["predicted_class"],
            "probability_malignant": persisted_result["probability_malignant"],
            "probability_benign": persisted_result["probability_benign"],
            "threshold": persisted_result["threshold"],
            "correct": correct,
        }

    return reconstructed_results, selected_case_id


def build_confusion_matrix(final_metrics: dict) -> pd.DataFrame:
    """Build a display table from persisted confusion counts."""
    return pd.DataFrame(
        {
            "Previsto: maligno": [
                final_metrics["true_malignant"],
                final_metrics["false_positive_malignant"],
            ],
            "Previsto: benigno": [
                final_metrics["false_negative_malignant"],
                final_metrics["true_benign"],
            ],
        },
        index=["Real: maligno", "Real: benigno"],
    )


def build_explainability_feature_table(features: list[dict]) -> pd.DataFrame:
    """Build a display table for global permutation importance features."""
    if not features:
        raise ValueError("Explainability features must not be empty")

    ranks = [feature.get("rank") for feature in features]
    if len(set(ranks)) != len(ranks):
        raise ValueError("Duplicated explainability ranks")

    rows = []
    required_keys = {
        "rank",
        "feature_name",
        "mean_importance",
        "std_importance",
        "positive_fraction",
    }
    for feature in features:
        if not required_keys.issubset(feature):
            raise ValueError("Missing explainability feature values")

        feature_name = feature["feature_name"]
        rows.append(
            {
                "Posição no ranking": feature["rank"],
                "Variável": translate_feature_name(feature_name),
                "Chave canônica": feature_name,
                "Importância média": feature["mean_importance"],
                "Desvio-padrão": feature["std_importance"],
                "Fração positiva": feature["positive_fraction"],
            }
        )

    return pd.DataFrame(rows).sort_values("Posição no ranking").reset_index(drop=True)


def build_explainability_fold_table(fold_scores: list[dict]) -> pd.DataFrame:
    """Build a display table for explainability validation fold scores."""
    if not fold_scores:
        raise ValueError("Explainability fold scores must not be empty")

    required_keys = {
        "fold",
        "train_sample_count",
        "validation_sample_count",
        "validation_malignant_count",
        "validation_benign_count",
        "baseline_roc_auc",
    }
    rows = []
    for fold_score in fold_scores:
        if not required_keys.issubset(fold_score):
            raise ValueError("Missing explainability fold values")

        rows.append(
            {
                "Fold": fold_score["fold"],
                "Amostras de treinamento": fold_score["train_sample_count"],
                "Amostras de validação": fold_score["validation_sample_count"],
                "Malignos na validação": fold_score["validation_malignant_count"],
                "Benignos na validação": fold_score["validation_benign_count"],
                "ROC AUC maligno": fold_score["baseline_roc_auc"],
            }
        )

    return pd.DataFrame(rows).sort_values("Fold").reset_index(drop=True)


def _sanitize_demo_progress_result(case_id: str, result: Any) -> dict[str, int | float | str]:
    if not isinstance(result, dict):
        raise ValueError("Demo progress result must be an object")

    unexpected_keys = set(result) - _DEMO_SESSION_RESULT_KEYS
    if unexpected_keys:
        raise ValueError("Demo progress result contains unexpected keys")

    if "features" in result:
        raise ValueError("Demo progress must not persist features")

    embedded_case_id = result.get("case_id")
    if embedded_case_id is not None and embedded_case_id != case_id:
        raise ValueError("Demo progress case id mismatch")

    missing_keys = _DEMO_PROGRESS_RESULT_KEYS - set(result)
    if missing_keys:
        raise ValueError("Demo progress result is missing required keys")

    return _validate_persisted_demo_progress_result(
        {key: result[key] for key in _DEMO_PROGRESS_RESULT_KEYS}
    )


def _decode_demo_progress_payload(encoded_progress: str) -> dict:
    if len(encoded_progress) % 4 == 1:
        raise ValueError("Demo progress is not valid base64")

    padded_progress = encoded_progress + ("=" * (-len(encoded_progress) % 4))
    try:
        raw_json = base64.b64decode(
            padded_progress.encode("ascii"),
            altchars=b"-_",
            validate=True,
        ).decode("utf-8")
    except (binascii.Error, UnicodeDecodeError) as exc:
        raise ValueError("Demo progress is not valid base64") from exc

    try:
        payload = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise ValueError("Demo progress is not valid JSON") from exc

    if not isinstance(payload, dict):
        raise ValueError("Demo progress must be an object")

    return payload


def _validate_persisted_demo_progress_result(raw_result: Any) -> dict[str, int | float | str]:
    if not isinstance(raw_result, dict):
        raise ValueError("Demo progress result must be an object")

    _validate_exact_keys(raw_result, _DEMO_PROGRESS_RESULT_KEYS, "Demo progress result")

    predicted_label = raw_result["predicted_label"]
    if isinstance(predicted_label, bool) or not isinstance(predicted_label, int):
        raise ValueError("Unexpected predicted label")

    if predicted_label not in _PREDICTED_LABEL_TO_CLASS:
        raise ValueError("Unexpected predicted label")

    predicted_class = raw_result["predicted_class"]
    if predicted_class not in {"malignant", "benign"}:
        raise ValueError("Unexpected predicted class")

    if predicted_class != _PREDICTED_LABEL_TO_CLASS[predicted_label]:
        raise ValueError("Predicted label and class are inconsistent")

    probability_malignant = _validate_probability(raw_result["probability_malignant"])
    probability_benign = _validate_probability(raw_result["probability_benign"])
    if not math.isclose(
        probability_malignant + probability_benign,
        1.0,
        rel_tol=0.0,
        abs_tol=DEMO_PROGRESS_PROBABILITY_TOLERANCE,
    ):
        raise ValueError("Demo progress probabilities must sum to one")

    threshold = _validate_finite_number(raw_result["threshold"], "Unexpected threshold")
    if not math.isclose(threshold, DEMO_PROGRESS_THRESHOLD, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("Unexpected threshold")

    return {
        "predicted_label": predicted_label,
        "predicted_class": predicted_class,
        "probability_malignant": probability_malignant,
        "probability_benign": probability_benign,
        "threshold": threshold,
    }


def _build_official_demo_case_index(demo_cases: list[dict]) -> dict[str, dict]:
    if not isinstance(demo_cases, list):
        raise ValueError("Demo cases must be a list")

    official_cases = {}
    for case in demo_cases:
        if not isinstance(case, dict):
            raise ValueError("Demo case must be an object")

        required_keys = {"case_id", "sample_index", "reference_label", "reference_class"}
        if not required_keys.issubset(case):
            raise ValueError("Demo case is missing required keys")

        case_id = case["case_id"]
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("Unexpected demo case id")

        if case_id in official_cases:
            raise ValueError("Duplicated demo case id")

        reference_label = case["reference_label"]
        if isinstance(reference_label, bool) or reference_label not in _PREDICTED_LABEL_TO_CLASS:
            raise ValueError("Unexpected reference label")

        reference_class = case["reference_class"]
        if reference_class != _PREDICTED_LABEL_TO_CLASS[reference_label]:
            raise ValueError("Reference label and class are inconsistent")

        sample_index = case["sample_index"]
        if isinstance(sample_index, bool) or not isinstance(sample_index, int):
            raise ValueError("Unexpected demo sample index")

        official_cases[case_id] = {
            "case_id": case_id,
            "sample_index": sample_index,
            "reference_label": reference_label,
            "reference_class": reference_class,
        }

    return official_cases


def _validate_probability(value: Any) -> float:
    probability = _validate_finite_number(value, "Unexpected probability")
    if not 0 <= probability <= 1:
        raise ValueError("Unexpected probability")

    return probability


def _validate_finite_number(value: Any, message: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(message)

    numeric_value = float(value)
    if not math.isfinite(numeric_value):
        raise ValueError(message)

    return numeric_value


def _validate_exact_keys(payload: dict, expected_keys: set[str], context: str) -> None:
    missing_keys = expected_keys - set(payload)
    extra_keys = set(payload) - expected_keys

    if missing_keys:
        raise ValueError(f"{context} is missing required keys")

    if extra_keys:
        raise ValueError(f"{context} contains unexpected keys")
