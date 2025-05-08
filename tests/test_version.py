from importlib.metadata import distribution


def test_true() -> None:
    assert distribution("uparser").version == "0.1.0"
