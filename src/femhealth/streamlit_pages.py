"""Streamlit pages for the FemHealth academic interface."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from femhealth.api_client import (
    FemHealthApiError,
    get_explainability,
    get_explainability_plot,
    get_health,
    get_model_info,
    request_prediction,
)
from femhealth.ui_labels import group_feature_names, translate_feature_name
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
