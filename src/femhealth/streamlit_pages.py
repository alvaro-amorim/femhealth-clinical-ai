"""Streamlit pages for the FemHealth academic interface."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from femhealth.api_client import (
    FemHealthApiError,
    get_demo_cases,
    get_explainability,
    get_explainability_plot,
    get_health,
    get_model_info,
    request_prediction,
)
from femhealth.ui_labels import group_feature_names, translate_feature_name
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

_DEMO_RESULTS_STATE_KEY = "demo_cases_page_results"
_DEMO_LAST_CASE_STATE_KEY = "demo_cases_page_last_case_id"
_DEMO_PROGRESS_QUERY_PARAM = "demo_progress"


def render_presentation_page() -> None:
    """Render the project presentation page."""
    st.title("FemHealth Clinical AI")
    st.subheader("Projeto acadêmico de classificação com apoio de aprendizado de máquina")
    st.warning(
        "Este projeto não possui validade clínica. A saída não é diagnóstico médico e não "
        "substitui avaliação por profissional de saúde."
    )

    st.markdown(
        """
        **Arquitetura resumida**

        Dados públicos → desenvolvimento e avaliação → artefato congelado → FastAPI → Streamlit
        """
    )

    try:
        health = get_health()
    except FemHealthApiError as exc:
        st.warning(str(exc))
    else:
        st.success("API conectada")
        columns = st.columns(2)
        columns[0].metric("Versão do artefato", health["artifact_version"])
        columns[1].metric(
            "Variante selecionada",
            model_variant_pt_br(health["selected_variant"]),
        )


def render_model_page() -> None:
    """Render model metadata and final results."""
    st.title("Modelo e resultados")

    try:
        model_info = get_model_info()
    except FemHealthApiError as exc:
        st.error(str(exc))
        st.info("Inicie o serviço FastAPI antes de consultar os resultados do modelo.")
        return

    final_metrics = model_info["final_holdout_metrics"]
    st.write("Resumo seguro do artefato final exposto pela FastAPI.")

    columns = st.columns(5)
    columns[0].metric("Modelo", model_info["selected_model"].upper())
    columns[1].metric("Calibração", "Sigmoid")
    columns[2].metric("Limiar de decisão", format_decimal_pt_br(model_info["threshold"]))
    columns[3].metric("Amostras de treinamento", str(model_info["training_sample_count"]))
    columns[4].metric("Amostras do teste final", str(final_metrics["test_sample_count"]))

    metric_columns = st.columns(5)
    metric_columns[0].metric("Acurácia", format_probability(final_metrics["accuracy"]))
    metric_columns[1].metric(
        "Recall maligno",
        format_probability(final_metrics["recall_malignant"]),
    )
    metric_columns[2].metric(
        "Especificidade benigna",
        format_probability(final_metrics["specificity_benign"]),
    )
    metric_columns[3].metric("F1 maligno", format_probability(final_metrics["f1_malignant"]))
    metric_columns[4].metric("ROC AUC", format_probability(final_metrics["roc_auc_malignant"]))

    st.subheader("Matriz de confusão")
    st.dataframe(build_confusion_matrix(final_metrics), use_container_width=True)

    with st.expander("Contrato das variáveis"):
        st.dataframe(
            _build_feature_contract_table(model_info["feature_names"]),
            hide_index=True,
            use_container_width=True,
        )

    st.caption(f"Versão do artefato: {model_info['artifact_version']}")
    st.caption(f"SHA-256: {model_info['model_sha256']}")
    st.warning(model_info["disclaimer"])


def render_explainability_page() -> None:
    """Render global explainability artifacts served by the API."""
    st.title("Explicabilidade global")
    st.warning(
        "A análise mostra dependência preditiva global do modelo. Não representa "
        "causalidade, diagnóstico ou relevância clínica."
    )

    try:
        explainability = get_explainability()
        plot_bytes = get_explainability_plot()
    except FemHealthApiError as exc:
        st.error(str(exc))
        st.info("Inicie o serviço FastAPI para consultar a explicabilidade.")
        return

    try:
        _validate_explainability_payload(explainability)
        feature_table = build_explainability_feature_table(explainability["features"])
        fold_table = build_explainability_fold_table(explainability["fold_scores"])
    except ValueError as exc:
        st.error(str(exc))
        return

    columns = st.columns(5)
    columns[0].metric(
        "Amostras de desenvolvimento",
        str(explainability["development_sample_count"]),
    )
    columns[1].metric("Folds", str(explainability["cv_splits"]))
    columns[2].metric("Repetições por variável", str(explainability["permutation_repeats"]))
    columns[3].metric("ROC AUC médio", format_probability(explainability["mean_fold_roc_auc"]))
    columns[4].metric(
        "Desvio-padrão entre folds",
        format_decimal_pt_br(explainability["std_fold_roc_auc"], decimal_places=4),
    )

    st.image(plot_bytes, caption="Importância média por permutação, com barras de erro.")
    st.write(
        "Maior importância média não implica causalidade. Diferenças pequenas devem "
        "ser interpretadas com cautela, e features correlacionadas podem compartilhar "
        "importância."
    )

    st.subheader("Principais variáveis")
    st.dataframe(
        feature_table.head(15),
        hide_index=True,
        use_container_width=True,
        column_config=_explainability_feature_column_config(),
    )

    with st.expander("Ranking completo das 30 variáveis"):
        st.dataframe(
            feature_table,
            hide_index=True,
            use_container_width=True,
            column_config=_explainability_feature_column_config(),
        )

    st.subheader("Folds de validação")
    st.dataframe(
        fold_table,
        hide_index=True,
        use_container_width=True,
        column_config={
            "ROC AUC maligno": st.column_config.NumberColumn(
                "ROC AUC maligno",
                format="%.6f",
            ),
        },
    )

    st.subheader("Limitações registradas")
    for limitation in explainability["limitations"]:
        st.markdown(f"- {limitation}")

    st.warning(explainability["disclaimer"])


def render_simulator_page() -> None:
    """Render the academic prediction simulator."""
    st.title("Simulador acadêmico")

    try:
        model_info = get_model_info()
        feature_names = model_info["feature_names"]
        validate_api_feature_contract(feature_names)
    except (FemHealthApiError, ValueError) as exc:
        st.error(str(exc))
        st.info("Inicie o serviço FastAPI antes de usar o simulador.")
        return

    st.warning(
        "O simulador apresenta a saída de um modelo acadêmico. Ele não realiza diagnóstico "
        "e não substitui avaliação médica."
    )

    submitted, feature_values = _render_prediction_form(feature_names)

    if not submitted:
        return

    missing_count = sum(value is None for value in feature_values.values())
    if missing_count:
        st.error(f"Preencha todos os campos antes de executar. Campos ausentes: {missing_count}.")
        return

    payload = {feature_name: float(feature_values[feature_name]) for feature_name in feature_names}

    try:
        prediction = request_prediction(payload)
    except FemHealthApiError as exc:
        st.error(str(exc))
        return

    _render_prediction_result(prediction)


def render_demo_cases_page() -> None:
    """Render holdout demonstration cases served by the API."""
    st.title("Casos de demonstração")
    st.warning(
        "Os casos desta página pertencem ao holdout final e não foram usados no "
        "treinamento. Eles são apresentados somente para demonstração acadêmica após a "
        "avaliação final já congelada."
    )

    try:
        demo_cases = get_demo_cases()
        _validate_demo_cases_payload_for_page(demo_cases)
    except (FemHealthApiError, ValueError) as exc:
        st.error(str(exc))
        st.info("Inicie o serviço FastAPI antes de consultar os casos de demonstração.")
        return

    st.caption("Seleção: Primeiros oito registros na ordem congelada do holdout final.")
    metric_columns = st.columns(4)
    metric_columns[0].metric("Casos disponíveis", str(demo_cases["case_count"]))
    metric_columns[1].metric("Malignos", str(demo_cases["malignant_case_count"]))
    metric_columns[2].metric("Benignos", str(demo_cases["benign_case_count"]))
    metric_columns[3].metric(
        "Acurácia oficial do holdout completo",
        format_probability(demo_cases["official_holdout_accuracy"]),
    )
    st.warning(
        "A taxa desta página considera apenas os casos executados nesta sessão e não "
        "substitui a avaliação oficial dos 114 registros."
    )

    _initialize_demo_progress_state(demo_cases["cases"])
    results = _get_demo_results_state()
    case_options = list(enumerate(demo_cases["cases"], start=1))
    default_case_index = _resolve_demo_selectbox_index(
        case_options,
        _get_demo_last_case_id_state(),
    )
    selected_position, selected_case = st.selectbox(
        "Caso",
        case_options,
        index=default_case_index,
        format_func=lambda option: f"Caso {option[0]} — índice {option[1]['sample_index']}",
        key="demo_cases_page_selected_case",
    )
    del selected_position

    with st.expander("Consultar os 30 valores do caso"):
        st.dataframe(
            build_demo_feature_table(selected_case["features"]),
            hide_index=True,
            width="stretch",
            column_config={
                "Valor": st.column_config.NumberColumn(
                    "Valor",
                    format="%.6f",
                ),
            },
        )

    if st.button("Executar caso selecionado", key="demo_cases_page_run_button"):
        try:
            prediction = request_prediction(selected_case["features"])
            correct = compare_demo_prediction(
                selected_case["reference_label"],
                prediction["predicted_label"],
            )
        except (FemHealthApiError, ValueError) as exc:
            st.error(str(exc))
        else:
            results[selected_case["case_id"]] = _build_demo_case_result(
                selected_case,
                prediction,
                correct,
            )
            st.session_state[_DEMO_LAST_CASE_STATE_KEY] = selected_case["case_id"]
            _sync_demo_progress_query_param(results, selected_case["case_id"])

    _render_demo_reset_button(results)
    selected_result = results.get(selected_case["case_id"])
    if selected_result is not None:
        _render_demo_case_result(selected_result)

    _render_demo_scoreboard(results)


def _render_prediction_form(feature_names: list[str]) -> tuple[bool, dict[str, float | None]]:
    feature_values: dict[str, float | None] = {}

    with st.form("prediction_form"):
        for group_name, grouped_features in group_feature_names(feature_names).items():
            with st.expander(group_name, expanded=group_name == "Valores médios"):
                columns = st.columns(2)
                for index, feature_name in enumerate(grouped_features):
                    with columns[index % 2]:
                        feature_values[feature_name] = st.number_input(
                            translate_feature_name(feature_name),
                            value=None,
                            min_value=0.0,
                            step=0.01,
                            format="%.6f",
                            placeholder="Informe o valor",
                            help=feature_name,
                            key=f"feature::{feature_name}",
                        )

        submitted = st.form_submit_button("Executar classificação acadêmica")

    return submitted, feature_values


def _render_prediction_result(prediction: dict) -> None:
    classification = prediction_class_pt_br(prediction["predicted_class"])

    if prediction["predicted_class"] == "malignant":
        st.warning(classification)
    else:
        st.success(classification)

    columns = st.columns(3)
    columns[0].metric(
        "Probabilidade estimada — padrão maligno",
        format_probability(prediction["probability_malignant"]),
    )
    columns[1].metric(
        "Probabilidade estimada — padrão benigno",
        format_probability(prediction["probability_benign"]),
    )
    columns[2].metric("Limiar aplicado", format_decimal_pt_br(prediction["threshold"]))
    st.warning(prediction["disclaimer"])


def _validate_demo_cases_payload_for_page(demo_cases: dict) -> None:
    if demo_cases["case_count"] != 8:
        raise ValueError("Quantidade inesperada de casos de demonstração.")

    if demo_cases["malignant_case_count"] != 4:
        raise ValueError("Quantidade inesperada de casos malignos.")

    if demo_cases["benign_case_count"] != 4:
        raise ValueError("Quantidade inesperada de casos benignos.")

    if demo_cases["used_for_training"] is not False:
        raise ValueError("Casos de demonstração não podem ter sido usados no treinamento.")

    if demo_cases["used_for_model_selection"] is not False:
        raise ValueError("Casos de demonstração não podem ter sido usados na seleção.")


def _get_demo_results_state() -> dict[str, dict]:
    results = st.session_state.setdefault(_DEMO_RESULTS_STATE_KEY, {})
    if not isinstance(results, dict):
        st.session_state[_DEMO_RESULTS_STATE_KEY] = {}
        return st.session_state[_DEMO_RESULTS_STATE_KEY]

    return results


def _get_demo_last_case_id_state() -> str | None:
    if _DEMO_LAST_CASE_STATE_KEY not in st.session_state:
        return None

    value = st.session_state[_DEMO_LAST_CASE_STATE_KEY]
    return value if isinstance(value, str) else None


def _initialize_demo_progress_state(demo_cases: list[dict]) -> None:
    if _DEMO_RESULTS_STATE_KEY in st.session_state:
        return

    encoded_progress = _get_demo_progress_query_param()
    if encoded_progress is None:
        st.session_state[_DEMO_RESULTS_STATE_KEY] = {}
        st.session_state.pop(_DEMO_LAST_CASE_STATE_KEY, None)
        return

    try:
        results, selected_case_id = deserialize_demo_progress(encoded_progress, demo_cases)
    except ValueError:
        st.session_state[_DEMO_RESULTS_STATE_KEY] = {}
        st.session_state.pop(_DEMO_LAST_CASE_STATE_KEY, None)
        _remove_demo_progress_query_param()
        return

    st.session_state[_DEMO_RESULTS_STATE_KEY] = results
    if selected_case_id is None:
        st.session_state.pop(_DEMO_LAST_CASE_STATE_KEY, None)
    else:
        st.session_state[_DEMO_LAST_CASE_STATE_KEY] = selected_case_id


def _get_demo_progress_query_param() -> str | None:
    raw_value = st.query_params.get(_DEMO_PROGRESS_QUERY_PARAM)
    if raw_value is None:
        return None

    if isinstance(raw_value, list):
        if len(raw_value) != 1:
            _remove_demo_progress_query_param()
            return None

        raw_value = raw_value[0]

    if not isinstance(raw_value, str) or not raw_value:
        _remove_demo_progress_query_param()
        return None

    return raw_value


def _sync_demo_progress_query_param(results: dict[str, dict], selected_case_id: str) -> None:
    st.query_params[_DEMO_PROGRESS_QUERY_PARAM] = serialize_demo_progress(
        results,
        selected_case_id,
    )


def _remove_demo_progress_query_param() -> None:
    if _DEMO_PROGRESS_QUERY_PARAM in st.query_params:
        del st.query_params[_DEMO_PROGRESS_QUERY_PARAM]


def _resolve_demo_selectbox_index(
    case_options: list[tuple[int, dict]],
    selected_case_id: str | None,
) -> int:
    if selected_case_id is None:
        return 0

    for option_index, (_, demo_case) in enumerate(case_options):
        if demo_case["case_id"] == selected_case_id:
            return option_index

    return 0


def _build_demo_case_result(selected_case: dict, prediction: dict, correct: bool) -> dict:
    return {
        "case_id": selected_case["case_id"],
        "sample_index": selected_case["sample_index"],
        "reference_label": selected_case["reference_label"],
        "reference_class": selected_case["reference_class"],
        "predicted_label": int(prediction["predicted_label"]),
        "predicted_class": prediction["predicted_class"],
        "probability_malignant": float(prediction["probability_malignant"]),
        "probability_benign": float(prediction["probability_benign"]),
        "threshold": float(prediction["threshold"]),
        "correct": correct,
    }


def _render_demo_case_result(result: dict) -> None:
    st.subheader("Resultado do caso")
    status_label = "Acertou" if result["correct"] else "Divergiu"
    if result["correct"]:
        st.success(status_label)
    else:
        st.warning(status_label)

    columns = st.columns(3)
    columns[0].metric(
        "Rótulo de referência do dataset",
        f"{reference_class_pt_br(result['reference_class'])} ({result['reference_label']})",
    )
    columns[1].metric(
        "Classificação produzida pelo modelo",
        prediction_class_pt_br(result["predicted_class"]),
    )
    columns[2].metric("Acertou ou Divergiu", status_label)

    probability_columns = st.columns(3)
    probability_columns[0].metric(
        "Probabilidade estimada — padrão maligno",
        format_probability(result["probability_malignant"]),
    )
    probability_columns[1].metric(
        "Probabilidade estimada — padrão benigno",
        format_probability(result["probability_benign"]),
    )
    probability_columns[2].metric(
        "Limiar aplicado",
        format_decimal_pt_br(result["threshold"]),
    )


def _render_demo_reset_button(results: dict[str, dict]) -> None:
    if st.button("Reiniciar placar da demonstração", key="demo_cases_page_reset_button"):
        results.clear()
        st.session_state[_DEMO_RESULTS_STATE_KEY] = results
        st.session_state.pop(_DEMO_LAST_CASE_STATE_KEY, None)
        _remove_demo_progress_query_param()


def _render_demo_scoreboard(results: dict[str, dict]) -> None:
    st.subheader("Placar da sessão")
    scoreboard = build_demo_scoreboard(
        {case_id: result["correct"] for case_id, result in results.items()}
    )
    accuracy_value = scoreboard["accuracy"]
    formatted_accuracy = (
        accuracy_value if isinstance(accuracy_value, str) else format_probability(accuracy_value)
    )

    columns = st.columns(4)
    columns[0].metric("Casos únicos testados", str(scoreboard["tested"]))
    columns[1].metric("Acertos", str(scoreboard["correct"]))
    columns[2].metric("Divergências", str(scoreboard["divergences"]))
    columns[3].metric("Taxa de acerto da demonstração", formatted_accuracy)
    st.caption(
        "Esta taxa é descritiva da sessão atual. A avaliação oficial permanece sendo a "
        "do holdout completo de 114 registros."
    )


def _build_feature_contract_table(feature_names: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Número": range(1, len(feature_names) + 1),
            "Nome em português": [
                translate_feature_name(feature_name) for feature_name in feature_names
            ],
            "Chave técnica canônica": feature_names,
        }
    )


def _validate_explainability_payload(explainability: dict) -> None:
    if explainability["holdout_used"] is not False:
        raise ValueError("Explainability holdout flag is invalid")

    if explainability["feature_count"] != 30:
        raise ValueError("Explainability feature count is invalid")

    if explainability["detail_row_count"] != 1500:
        raise ValueError("Explainability detail count is invalid")


def _explainability_feature_column_config() -> dict:
    return {
        "Importância média": st.column_config.NumberColumn(
            "Importância média",
            format="%.6f",
        ),
        "Desvio-padrão": st.column_config.NumberColumn(
            "Desvio-padrão",
            format="%.6f",
        ),
        "Fração positiva": st.column_config.NumberColumn(
            "Fração positiva",
            format="percent",
        ),
    }
