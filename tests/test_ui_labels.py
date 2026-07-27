import pytest

from femhealth.data import WDBC_FEATURE_NAMES
from femhealth.ui_labels import FEATURE_LABELS_PT_BR, group_feature_names, translate_feature_name


def test_feature_labels_match_canonical_features() -> None:
    assert len(FEATURE_LABELS_PT_BR) == 30
    assert list(FEATURE_LABELS_PT_BR) == WDBC_FEATURE_NAMES
    assert all(label.strip() for label in FEATURE_LABELS_PT_BR.values())


def test_translate_feature_name_returns_portuguese_label() -> None:
    assert translate_feature_name("mean radius") == "Raio médio"


def test_translate_feature_name_rejects_unknown_feature() -> None:
    with pytest.raises(ValueError, match="Feature without Portuguese label"):
        translate_feature_name("unknown")


def test_group_feature_names_builds_three_ordered_groups() -> None:
    groups = group_feature_names(WDBC_FEATURE_NAMES)

    assert list(groups) == ["Valores médios", "Erros padrão", "Piores valores"]
    assert [len(features) for features in groups.values()] == [10, 10, 10]
    assert groups["Valores médios"] == WDBC_FEATURE_NAMES[:10]
    assert groups["Erros padrão"] == WDBC_FEATURE_NAMES[10:20]
    assert groups["Piores valores"] == WDBC_FEATURE_NAMES[20:]


def test_group_feature_names_rejects_unknown_feature() -> None:
    with pytest.raises(ValueError, match="Feature without Portuguese label"):
        group_feature_names(["unknown"])
