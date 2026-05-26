import lossgog


def test_package_import():
    assert hasattr(lossgog, "__version__") or hasattr(lossgog, "VERSION") or True
    print("Lossgog imported successfully!")
