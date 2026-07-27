import pytest

from femhealth.explainability_artifacts import PNG_SIGNATURE, load_explainability_artifacts


def test_committed_explainability_artifacts_load_successfully() -> None:
    payload, plot_bytes = load_explainability_artifacts()

    assert len(payload["features"]) == 30
    assert len(payload["fold_scores"]) == 5
    assert payload["mean_fold_roc_auc"] == pytest.approx(0.996285)
    assert payload["detail_row_count"] == 1500
    assert payload["holdout_used"] is False
    assert plot_bytes.startswith(PNG_SIGNATURE)
