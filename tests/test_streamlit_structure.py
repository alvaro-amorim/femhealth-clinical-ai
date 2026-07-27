from pathlib import Path

FORBIDDEN_TERMS = [
    "load_model_artifact",
    "predict_with_artifact",
    "load_explainability_artifacts",
    "model_explainability",
    "joblib",
    "sklearn",
    "load_wdbc_data",
    "split_development_test",
    "evaluate_final_holdout",
    "build_selected_estimator",
    "build_and_persist_final_artifact",
    ".fit(",
    "GridSearchCV",
    "pandas.read_csv",
    "read_csv",
    "read_bytes",
    "json.load",
    "Path(\"reports",
    "reports/explainability",
    "compute_cross_validated_permutation_importance",
    "run_explainability_once",
]


def test_streamlit_layers_do_not_use_forbidden_model_operations() -> None:
    combined_source = "\n".join(
        [
            Path("streamlit_app.py").read_text(encoding="utf-8"),
            Path("src/femhealth/streamlit_pages.py").read_text(encoding="utf-8"),
            Path("src/femhealth/api_client.py").read_text(encoding="utf-8"),
        ]
    )

    for term in FORBIDDEN_TERMS:
        assert term not in combined_source


def test_streamlit_entrypoint_uses_multipage_navigation() -> None:
    source = Path("streamlit_app.py").read_text(encoding="utf-8")

    assert "st.set_page_config" in source
    assert "st.Page" in source
    assert "st.navigation" in source
    assert ".run()" in source
    assert "Explicabilidade" in source
    assert "render_explainability_page" in source
    assert "Casos de demonstração" in source
    assert "casos-demonstracao" in source
    assert "render_demo_cases_page" in source
    assert source.index("render_demo_cases_page") < source.index("render_simulator_page")


def test_streamlit_pages_use_api_client_and_form() -> None:
    source = Path("src/femhealth/streamlit_pages.py").read_text(encoding="utf-8")

    assert "st.form" in source
    assert "st.form_submit_button" in source
    assert "get_health" in source
    assert "get_model_info" in source
    assert "get_explainability" in source
    assert "get_explainability_plot" in source
    assert "get_demo_cases" in source
    assert "request_prediction" in source
    assert "render_explainability_page" in source
    assert "render_demo_cases_page" in source
    assert "Reiniciar placar da demonstração" in source
    assert "Consultar os 30 valores do caso" in source


def test_streamlit_demo_page_uses_api_and_preserves_manual_simulator() -> None:
    source = Path("src/femhealth/streamlit_pages.py").read_text(encoding="utf-8")
    demo_source = source[
        source.index("def render_demo_cases_page") : source.index("def _render_prediction_form")
    ]

    assert "get_demo_cases()" in demo_source
    assert demo_source.count("get_demo_cases()") == 1
    assert "request_prediction(selected_case[\"features\"])" in demo_source
    assert "build_demo_feature_table" in demo_source
    assert "compare_demo_prediction" in demo_source
    assert "reference_label" in demo_source
    assert "render_simulator_page" in source
    assert "st.form_submit_button(\"Executar classificação acadêmica\")" in source
