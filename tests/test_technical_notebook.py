import json
from pathlib import Path

NOTEBOOK_PATH = Path("notebooks/02_modelagem_e_explicabilidade.ipynb")

REQUIRED_TITLES = [
    "Dataset e contrato",
    "Separação desenvolvimento/teste",
    "Benchmark dos cinco modelos",
    "Ajuste de hiperparâmetros",
    "Calibração e análise de threshold",
    "Seleção congelada",
    "Resultado final do holdout",
    "Artefato persistido",
    "Explicabilidade global",
    "Discussão crítica",
    "Conclusão",
]

REQUIRED_REFERENCES = [
    "notebooks/01_exploracao_wdbc.ipynb",
    "final_holdout_summary.json",
    "permutation_importance_summary.csv",
    "permutation_importance_top15.png",
]

FORBIDDEN_PATTERNS = [
    "evaluate_final_holdout(",
    "run_final_holdout_once(",
    "run_explainability_once(",
    "build_and_persist_final_artifact(",
    "joblib.load",
    "joblib.dump",
    ".fit(",
    "GridSearchCV(",
    "permutation_importance(",
]

FORBIDDEN_WRITES = [
    "to_csv(",
    "to_json(",
    "write_text(",
    "savefig(",
]


def _load_notebook() -> dict:
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def _cell_source_text(notebook: dict) -> str:
    sources = []
    for cell in notebook["cells"]:
        source = cell.get("source", "")
        if isinstance(source, list):
            sources.append("".join(source))
        else:
            sources.append(source)
    return "\n".join(sources)


def test_technical_notebook_exists_and_uses_nbformat_4() -> None:
    assert NOTEBOOK_PATH.exists()
    notebook = _load_notebook()

    assert notebook["nbformat"] == 4
    assert NOTEBOOK_PATH.stat().st_size < 5 * 1024 * 1024


def test_technical_notebook_has_expected_cell_structure() -> None:
    notebook = _load_notebook()
    markdown_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "markdown"]
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]

    assert len(markdown_cells) >= 12
    assert len(code_cells) >= 8
    assert all(cell["execution_count"] is not None for cell in code_cells)


def test_technical_notebook_has_no_error_outputs() -> None:
    notebook = _load_notebook()

    assert not [
        output
        for cell in notebook["cells"]
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]


def test_technical_notebook_preserves_outputs() -> None:
    notebook = _load_notebook()
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    cells_with_outputs = [cell for cell in code_cells if cell.get("outputs")]
    output_count = sum(len(cell.get("outputs", [])) for cell in code_cells)

    assert len(cells_with_outputs) >= 8
    assert output_count >= 10


def test_technical_notebook_contains_required_sections_and_references() -> None:
    notebook_text = _cell_source_text(_load_notebook())

    for title in REQUIRED_TITLES:
        assert title in notebook_text

    for reference in REQUIRED_REFERENCES:
        assert reference in notebook_text


def test_technical_notebook_avoids_forbidden_operations() -> None:
    notebook_text = _cell_source_text(_load_notebook())

    for pattern in FORBIDDEN_PATTERNS:
        assert pattern not in notebook_text

    for pattern in FORBIDDEN_WRITES:
        assert pattern not in notebook_text


def test_technical_notebook_does_not_use_holdout_after_split() -> None:
    notebook = _load_notebook()
    code_sources = []
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = cell.get("source", "")
        code_sources.append("".join(source) if isinstance(source, list) else source)

    split_cell_index = next(
        index
        for index, source in enumerate(code_sources)
        if "split_development_test(X, y)" in source
    )
    split_cell = code_sources[split_cell_index]
    later_code = "\n".join(code_sources[split_cell_index + 1 :])

    assert "del X_test, y_test" in split_cell
    assert "X_test" not in later_code
    assert "y_test" not in later_code
    assert 'final_summary["probability"]' not in later_code
    assert 'final_summary["predictions"]' not in later_code


def test_technical_notebook_contains_key_recorded_results() -> None:
    notebook_text = json.dumps(_load_notebook(), ensure_ascii=False)

    for expected_text in ["svm_sigmoid", "0.51", "114", "1500", "worst texture"]:
        assert expected_text in notebook_text
