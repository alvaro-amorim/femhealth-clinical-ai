"""Single-use command for final holdout evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from femhealth.data import load_wdbc_data
from femhealth.data_split import split_development_test
from femhealth.final_evaluation import evaluate_final_holdout

FINAL_RESULTS_DIRECTORY = Path("reports/results")
FINAL_SUMMARY_PATH = FINAL_RESULTS_DIRECTORY / "final_holdout_summary.json"
FINAL_PREDICTIONS_PATH = FINAL_RESULTS_DIRECTORY / "final_holdout_predictions.csv"


def run_final_holdout_once(
    output_directory: Path = FINAL_RESULTS_DIRECTORY,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the frozen final holdout evaluation once and persist its results."""
    output_directory = Path(output_directory)
    summary_path = output_directory / FINAL_SUMMARY_PATH.name
    predictions_path = output_directory / FINAL_PREDICTIONS_PATH.name

    existing_paths = [path for path in (summary_path, predictions_path) if path.exists()]
    if existing_paths:
        existing = ", ".join(str(path) for path in existing_paths)
        raise FileExistsError(f"Final holdout results already exist: {existing}")

    X, y = load_wdbc_data()
    X_development, X_test, y_development, y_test = split_development_test(X, y)
    summary, predictions, _ = evaluate_final_holdout(
        X_development,
        y_development,
        X_test,
        y_test,
    )

    output_directory.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(_summary_record(summary), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    predictions.to_csv(predictions_path, index=True, index_label="sample_index")

    return summary, predictions


def main() -> None:
    """Run the final holdout command and print the persisted outputs."""
    summary, predictions = run_final_holdout_once()

    print("RESUMO FINAL")
    print(summary.to_string(index=False))
    print()
    print("ERROS")
    print(predictions.loc[~predictions["correct"]].to_string())
    print()
    print(f"Resumo salvo em: {FINAL_SUMMARY_PATH}")
    print(f"Predições salvas em: {FINAL_PREDICTIONS_PATH}")


def _summary_record(summary: pd.DataFrame) -> dict[str, object]:
    record = summary.iloc[0].to_dict()
    return {key: _json_value(value) for key, value in record.items()}


def _json_value(value: object) -> object:
    if hasattr(value, "item"):
        return value.item()

    return value


if __name__ == "__main__":
    main()
