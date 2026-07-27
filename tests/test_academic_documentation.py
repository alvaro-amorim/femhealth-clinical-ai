import re
from pathlib import Path

DOCUMENTS = {
    Path("docs/ACADEMIC_DOCUMENTATION.md"): "# Documentação acadêmica — FemHealth Clinical AI",
    Path("docs/MODEL_CARD.md"): "# Model Card — FemHealth SVM Sigmoid",
    Path("docs/ARCHITECTURE.md"): "# Arquitetura — FemHealth Clinical AI",
    Path("docs/RESPONSIBLE_USE.md"): "# Ética, limitações e uso responsável",
    Path("docs/AI_USAGE.md"): "# Uso de inteligência artificial no desenvolvimento",
}


def test_academic_documents_exist_and_are_non_empty() -> None:
    for path, title in DOCUMENTS.items():
        assert path.exists()
        text = path.read_text(encoding="utf-8")
        assert text.strip()
        assert title in text


def test_academic_documentation_contains_required_sections() -> None:
    text = Path("docs/ACADEMIC_DOCUMENTATION.md").read_text(encoding="utf-8")
    sections = [
        "## 1. Resumo do projeto",
        "## 2. Problema abordado",
        "## 3. Dados",
        "## 4. Protocolo experimental",
        "## 5. Modelos avaliados",
        "## 6. Modelo selecionado",
        "## 7. Resultado final",
        "## 8. Explicabilidade",
        "## 9. Arquitetura da aplicação",
        "## 10. Reprodutibilidade",
        "## 11. Limitações",
        "## 12. Conclusão",
        "## 13. Próximos passos",
    ]

    for section in sections:
        assert section in text


def test_model_card_contains_usage_metrics_risks_and_governance() -> None:
    text = Path("docs/MODEL_CARD.md").read_text(encoding="utf-8")
    normalized_text = " ".join(text.split())
    expected_terms = [
        "## Uso pretendido",
        "## Usos não pretendidos",
        "## Métricas",
        "## Limitações e riscos",
        "## Manutenção e governança",
        "classe maligna 0",
    ]

    for term in expected_terms:
        assert term in text

    assert "holdout ficou fora do treinamento" in normalized_text


def test_architecture_contains_flows_and_mermaid() -> None:
    text = Path("docs/ARCHITECTURE.md").read_text(encoding="utf-8")

    assert "```mermaid" in text
    assert "Dataset WDBC" in text
    assert "Joblib final" in text
    assert "FastAPI" in text
    assert "Streamlit" in text
    assert "Artefatos de explicabilidade" in text
    assert "Endpoints JSON e PNG" in text
    assert "API não treina modelos" in text
    assert "Streamlit não carrega Joblib" in text


def test_responsible_use_contains_required_topics() -> None:
    text = Path("docs/RESPONSIBLE_USE.md").read_text(encoding="utf-8")
    expected_terms = [
        "## Falsos negativos e falsos positivos",
        "## Interpretação das probabilidades",
        "## Supervisão humana",
        "## Privacidade e segurança",
        "## Usos proibidos",
    ]

    for term in expected_terms:
        assert term in text


def test_ai_usage_contains_support_verification_limits_and_responsibility() -> None:
    text = Path("docs/AI_USAGE.md").read_text(encoding="utf-8")
    expected_terms = [
        "## Atividades auxiliadas",
        "## Atividades verificadas por execução",
        "## Limites do uso de IA",
        "## Responsabilidade autoral",
        "revisão humana",
    ]

    for term in expected_terms:
        assert term in text


def test_readme_references_documentation_index() -> None:
    text = Path("README.md").read_text(encoding="utf-8")
    expected_links = [
        "docs/ACADEMIC_DOCUMENTATION.md",
        "docs/MODEL_CARD.md",
        "docs/ARCHITECTURE.md",
        "docs/RESPONSIBLE_USE.md",
        "docs/AI_USAGE.md",
        "notebooks/01_exploracao_wdbc.ipynb",
        "notebooks/02_modelagem_e_explicabilidade.ipynb",
    ]

    assert "## Documentação" in text
    for link in expected_links:
        assert link in text


def test_relative_markdown_links_point_to_existing_files() -> None:
    for path in [Path("README.md"), *DOCUMENTS]:
        text = path.read_text(encoding="utf-8")
        for link in re.findall(r"\]\(([^)]+)\)", text):
            if "://" in link:
                continue
            target = link.split("#", maxsplit=1)[0]
            if not target:
                continue

            resolved = (path.parent / target).resolve()
            assert resolved.exists(), f"Broken link {link} in {path}"


def test_critical_facts_are_documented_consistently() -> None:
    combined_text = _combined_documentation_text()
    expected_facts = [
        "569",
        "455",
        "114",
        "svm_sigmoid",
        "0.51",
        "41",
        "1 falso negativo",
        "2 falsos positivos",
        "70",
        "CC43CEC3BA58C5A4950217E80C8B286B0E7DB501FF663BA9BE9F91DF1F4B05B5",
        "worst texture",
        "holdout_used=false",
    ]

    for fact in expected_facts:
        assert fact in combined_text


def test_documentation_does_not_make_clinical_claims() -> None:
    combined_text = _combined_documentation_text().lower()
    forbidden_claims = [
        "está pronto para uso real",
        "validade clínica comprovada",
        "ferramenta clínica aprovada",
        "aprovado para uso médico",
        "diagnóstico automatizado",
        "diagnóstico autônomo",
        "substitui profissionais",
    ]

    for claim in forbidden_claims:
        assert claim not in combined_text


def test_documentation_references_notebooks_and_artifacts() -> None:
    combined_text = _combined_documentation_text()
    expected_references = [
        "notebooks/01_exploracao_wdbc.ipynb",
        "notebooks/02_modelagem_e_explicabilidade.ipynb",
        "artifacts/model/femhealth_svm_sigmoid.joblib",
        "reports/explainability",
    ]

    for reference in expected_references:
        assert reference in combined_text


def _combined_documentation_text() -> str:
    paths = [Path("README.md"), *DOCUMENTS]
    return "\n".join(path.read_text(encoding="utf-8") for path in paths)
