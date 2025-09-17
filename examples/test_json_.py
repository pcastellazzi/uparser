import pytest
from json_ import JSONValue, json_parser


@pytest.mark.parametrize(
    ("json_string", "json_value"),
    [
        ("null", None),
        ("false", False),
        ("true", True),
    ],
)
def test_literals(json_string: str, json_value: JSONValue) -> None:
    assert json_parser(json_string) == json_value


@pytest.mark.parametrize(
    ("json_string", "json_value"),
    [
        ("0", 0),
        ("1", 1),
        ("123", 123),
        ("-1", -1),
        ("-123", -123),
        ("0.1", 0.1),
        ("1.0", 1.0),
        ("1.23", 1.23),
        ("-0.1", -0.1),
        ("-1.23", -1.23),
        ("1e0", 1e0),
        ("1e1", 1e1),
        ("1e+1", 1e1),
        ("1e-1", 1e-1),
        ("1E1", 1e1),
        ("1.2e1", 1.2e1),
        ("-1.2e-10", -1.2e-10),
    ],
)
def test_numbers(json_string: str, json_value: JSONValue) -> None:
    assert json_parser(json_string) == json_value


@pytest.mark.parametrize(
    ("json_string", "json_value"),
    [
        ('""', ""),
        ('"hello"', "hello"),
        ('"hello world"', "hello world"),
        (r'"\""', '"'),
        (r'"\\"', "\\"),
        (r'"\/"', "/"),
        (r'"\b"', "\x08"),
        (r'"\f"', "\x0c"),
        (r'"\n"', "\n"),
        (r'"\r"', "\r"),
        (r'"\t"', "\t"),
        (r'"\u1234"', "\u1234"),
        (r'"\uabcd"', "\uabcd"),
        (r'"\uABCD"', "\uabcd"),
    ],
)
def test_strings(json_string: str, json_value: JSONValue) -> None:
    assert json_parser(json_string) == json_value


@pytest.mark.parametrize(
    ("json_string", "json_value"),
    [
        (r"[]", []),
        (r"[null]", [None]),
        (r"[false]", [False]),
        (r"[true]", [True]),
        (r"[1, 2.0, 3e1]", [1, 2.0, 3e1]),
        (r'["a", "b", "\n"]', ["a", "b", "\n"]),
        (
            r'[[null, true, false], [1.0e33, ["test"]]]',
            [[None, True, False], [1.0e33, ["test"]]],
        ),
    ],  # pyright: ignore[reportUnknownArgumentType]
)
def test_arrays(json_string: str, json_value: JSONValue) -> None:
    assert json_parser(json_string) == json_value


@pytest.mark.parametrize(
    ("json_string", "json_value"),
    [
        (r"{}", {}),
        (r'{"a": 1}', {"a": 1}),
        (
            r'{"a": 1, "b": "hello", "c": [1, 2, 3], "d": {"z": -1, "y": -2, "x": -3}}',
            {"a": 1, "b": "hello", "c": [1, 2, 3], "d": {"z": -1, "y": -2, "x": -3}},
        ),
    ],
)
def test_objects(json_string: str, json_value: JSONValue) -> None:
    assert json_parser(json_string) == json_value
