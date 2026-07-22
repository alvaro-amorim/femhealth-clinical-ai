from femhealth.data import load_wdbc_data
from femhealth.data_split import split_development_test


def test_split_development_test_sizes_and_shapes() -> None:
    X, y = load_wdbc_data()

    X_development, X_test, y_development, y_test = split_development_test(X, y)

    assert X_development.shape == (455, 30)
    assert X_test.shape == (114, 30)
    assert y_development.shape == (455,)
    assert y_test.shape == (114,)


def test_split_development_test_preserves_stratified_distribution() -> None:
    X, y = load_wdbc_data()

    _, _, y_development, y_test = split_development_test(X, y)

    assert y_development.value_counts().sort_index().to_dict() == {0: 170, 1: 285}
    assert y_test.value_counts().sort_index().to_dict() == {0: 42, 1: 72}


def test_split_development_test_keeps_indices_aligned() -> None:
    X, y = load_wdbc_data()

    X_development, X_test, y_development, y_test = split_development_test(X, y)

    assert X_development.index.equals(y_development.index)
    assert X_test.index.equals(y_test.index)


def test_split_development_test_has_no_index_overlap() -> None:
    X, y = load_wdbc_data()

    X_development, X_test, _, _ = split_development_test(X, y)

    assert set(X_development.index).isdisjoint(set(X_test.index))


def test_split_development_test_index_union_matches_original_data() -> None:
    X, y = load_wdbc_data()

    X_development, X_test, _, _ = split_development_test(X, y)

    split_indices = set(X_development.index).union(set(X_test.index))
    assert split_indices == set(X.index)


def test_split_development_test_is_reproducible() -> None:
    X, y = load_wdbc_data()

    first_split = split_development_test(X, y)
    second_split = split_development_test(X, y)

    for first_part, second_part in zip(first_split, second_split, strict=True):
        assert first_part.equals(second_part)


def test_split_development_test_does_not_modify_original_data() -> None:
    X, y = load_wdbc_data()
    original_X = X.copy(deep=True)
    original_y = y.copy(deep=True)

    split_development_test(X, y)

    assert X.equals(original_X)
    assert y.equals(original_y)
