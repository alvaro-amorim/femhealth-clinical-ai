"""Portuguese labels for WDBC feature names."""

from __future__ import annotations

FEATURE_LABELS_PT_BR: dict[str, str] = {
    "mean radius": "Raio médio",
    "mean texture": "Textura média",
    "mean perimeter": "Perímetro médio",
    "mean area": "Área média",
    "mean smoothness": "Suavidade média",
    "mean compactness": "Compacidade média",
    "mean concavity": "Concavidade média",
    "mean concave points": "Pontos côncavos médios",
    "mean symmetry": "Simetria média",
    "mean fractal dimension": "Dimensão fractal média",
    "radius error": "Erro padrão do raio",
    "texture error": "Erro padrão da textura",
    "perimeter error": "Erro padrão do perímetro",
    "area error": "Erro padrão da área",
    "smoothness error": "Erro padrão da suavidade",
    "compactness error": "Erro padrão da compacidade",
    "concavity error": "Erro padrão da concavidade",
    "concave points error": "Erro padrão dos pontos côncavos",
    "symmetry error": "Erro padrão da simetria",
    "fractal dimension error": "Erro padrão da dimensão fractal",
    "worst radius": "Pior raio",
    "worst texture": "Pior textura",
    "worst perimeter": "Pior perímetro",
    "worst area": "Pior área",
    "worst smoothness": "Pior suavidade",
    "worst compactness": "Pior compacidade",
    "worst concavity": "Pior concavidade",
    "worst concave points": "Piores pontos côncavos",
    "worst symmetry": "Pior simetria",
    "worst fractal dimension": "Pior dimensão fractal",
}

MEAN_GROUP = "Valores médios"
ERROR_GROUP = "Erros padrão"
WORST_GROUP = "Piores valores"


def translate_feature_name(feature_name: str) -> str:
    """Translate a canonical WDBC feature name to Portuguese."""
    try:
        return FEATURE_LABELS_PT_BR[feature_name]
    except KeyError as exc:
        raise ValueError(f"Feature without Portuguese label: {feature_name}") from exc


def group_feature_names(feature_names: list[str]) -> dict[str, list[str]]:
    """Group canonical feature names while preserving input order."""
    groups = {
        MEAN_GROUP: [],
        ERROR_GROUP: [],
        WORST_GROUP: [],
    }

    for feature_name in feature_names:
        translate_feature_name(feature_name)

        if feature_name.startswith("mean "):
            groups[MEAN_GROUP].append(feature_name)
        elif feature_name.endswith(" error"):
            groups[ERROR_GROUP].append(feature_name)
        elif feature_name.startswith("worst "):
            groups[WORST_GROUP].append(feature_name)
        else:
            raise ValueError(f"Unknown feature group: {feature_name}")

    return groups
