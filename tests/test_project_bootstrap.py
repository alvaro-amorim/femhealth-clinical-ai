import femhealth


def test_package_can_be_imported() -> None:
    assert femhealth is not None


def test_package_version() -> None:
    assert femhealth.__version__ == "0.1.0"
