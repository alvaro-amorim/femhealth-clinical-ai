"""Single-use runner for global permutation importance artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from femhealth.data import WDBC_FEATURE_COUNT, WDBC_FEATURE_NAMES, load_wdbc_data
from femhealth.data_split import RANDOM_STATE, TEST_SIZE, split_development_test
from femhealth.final_evaluation import DEVELOPMENT_CLASS_DISTRIBUTION, DEVELOPMENT_SAMPLE_COUNT
from femhealth.final_selection import (
    SELECTED_CALIBRATION,
    SELECTED_MODEL,
    SELECTED_THRESHOLD,
    SELECTED_VARIANT,
)
from femhealth.model_evaluation import BENIGN_LABEL, MALIGNANT_LABEL
from femhealth.model_explainability import (
    EXPLAINABILITY_CV_SPLITS,
    EXPLAINABILITY_RANDOM_STATE,
    EXPLAINABILITY_SCORER,
    PERMUTATION_REPEATS,
    compute_cross_validated_permutation_importance,
)

EXPLAINABILITY_DIRECTORY = Path("reports/explainability")
DETAILS_FILENAME = "permutation_importance_details.csv"
SUMMARY_FILENAME = "permutation_importance_summary.csv"
FOLD_SCORES_FILENAME = "permutation_importance_fold_scores.csv"
METADATA_FILENAME = "permutation_importance_metadata.json"
PLOT_FILENAME = "permutation_importance_top15.png"

LIMITATIONS = [
    "Variáveis correlacionadas podem dividir importância entre si.",
    "Importância por permutação mede dependência preditiva, não causalidade.",
    "Resultados pertencem ao dataset WDBC.",
    "A análise não constitui explicação clínica.",
    "A amostra é limitada e não representa validação externa.",
]


def run_explainability_once(
    output_directory: Path = EXPLAINABILITY_DIRECTORY,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object], dict[str, Path]]:
    """Run explainability analysis once and persist CSV, JSON, and PNG artifacts."""
    output_directory = Path(output_directory)
    paths = _build_output_paths(output_directory)
    _ensure_outputs_are_new(paths)
    output_directory.mkdir(parents=True, exist_ok=True)
    temporary_paths = {name: _temporary_path(path) for name, path in paths.items()}
    created_final_paths: list[Path] = []

    try:
        X, y = load_wdbc_data()
        X_development, X_test, y_development, y_test = split_development_test(X, y)
        del X_test, y_test

        details, summary, fold_scores = compute_cross_validated_permutation_importance(
            X_development,
            y_development,
        )
        metadata = _build_metadata(details)

        details.to_csv(temporary_paths["details"], index=False)
        summary.to_csv(temporary_paths["summary"], index=False)
        fold_scores.to_csv(temporary_paths["fold_scores"], index=False)
        temporary_paths["metadata"].write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        _save_top15_plot(summary, temporary_paths["plot"])

        for name, path in paths.items():
            temporary_paths[name].replace(path)
            created_final_paths.append(path)
    except Exception:
        _remove_paths(*temporary_paths.values(), *created_final_paths)
        raise

    return details, summary, fold_scores, metadata, paths


def main() -> None:
    """Run explainability artifacts generation and print a concise summary."""
    details, summary, fold_scores, _, paths = run_explainability_once()

    print("Arquivos criados:")
    for path in paths.values():
        print(f"- {path}")

    print()
    print(f"ROC AUC médio entre folds: {fold_scores['baseline_roc_auc'].mean():.6f}")
    print()
    print("Top 10 features:")
    print(
        summary[
            [
                "rank",
                "feature_name",
                "mean_importance",
                "std_importance",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )
    print()
    print(f"Linhas detalhadas: {len(details)}")


def _build_output_paths(output_directory: Path) -> dict[str, Path]:
    return {
        "details": output_directory / DETAILS_FILENAME,
        "summary": output_directory / SUMMARY_FILENAME,
        "fold_scores": output_directory / FOLD_SCORES_FILENAME,
        "metadata": output_directory / METADATA_FILENAME,
        "plot": output_directory / PLOT_FILENAME,
    }


def _temporary_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}.tmp{path.suffix}")


def _ensure_outputs_are_new(paths: dict[str, Path]) -> None:
    existing_paths = [path for path in paths.values() if path.exists()]
    if existing_paths:
        existing = ", ".join(str(path) for path in existing_paths)
        raise FileExistsError(f"Explainability outputs already exist: {existing}")


def _build_metadata(details: pd.DataFrame) -> dict[str, object]:
    return {
        "method": "cross_validated_permutation_importance",
        "scorer": EXPLAINABILITY_SCORER,
        "selected_variant": SELECTED_VARIANT,
        "selected_model": SELECTED_MODEL,
        "selected_calibration": SELECTED_CALIBRATION,
        "selected_threshold": SELECTED_THRESHOLD,
        "malignant_label": MALIGNANT_LABEL,
        "benign_label": BENIGN_LABEL,
        "development_sample_count": DEVELOPMENT_SAMPLE_COUNT,
        "development_class_distribution": {
            str(label): count for label, count in DEVELOPMENT_CLASS_DISTRIBUTION.items()
        },
        "cv_splits": EXPLAINABILITY_CV_SPLITS,
        "shuffle": True,
        "random_state": EXPLAINABILITY_RANDOM_STATE,
        "test_size": TEST_SIZE,
        "split_random_state": RANDOM_STATE,
        "permutation_repeats": PERMUTATION_REPEATS,
        "feature_count": WDBC_FEATURE_COUNT,
        "feature_names": WDBC_FEATURE_NAMES,
        "detail_row_count": len(details),
        "holdout_used": False,
        "final_model_artifact_modified": False,
        "limitations": LIMITATIONS,
    }


def _save_top15_plot(summary: pd.DataFrame, path: Path) -> None:
    top15 = summary.head(15).sort_values("mean_importance", ascending=True)
    figure, axis = plt.subplots(figsize=(9, 6))
    axis.barh(
        top15["feature_name"],
        top15["mean_importance"],
        xerr=top15["std_importance"],
        color="#4B5563",
    )
    axis.set_xlabel("Redução média no ROC AUC maligno")
    axis.set_title("Importância global por permutação — validação cruzada")
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _remove_paths(*paths: Path) -> None:
    for path in paths:
        path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
