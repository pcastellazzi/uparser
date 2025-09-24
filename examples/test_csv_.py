import pytest
from csv_ import field, record

import uparser as p


@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        ("  text  ", "  text  "),  # spaces are preserved
        ("some text", "some text"),
        ('""', ""),  # empty quoted field is ok
        ('"  text  "', "  text  "),  # spaces are preserved
        ('"some ""quoted"" text"', 'some "quoted" text'),  # quote escaping
    ],
)
def test_a_field(actual: str, expected: str) -> None:
    result = field(0, actual)
    assert isinstance(result, p.Success)
    assert result.value == expected


@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        ('""', [""]),  # a record with a quoted empty field
        ("a,b,c", ["a", "b", "c"]),  # multiple fields
        ('"a, b"', ["a, b"]),  # a comma inside a field
        ('"a\r\nb"', ["a\r\nb"]),  # crlf inside a field
    ],
)
def test_a_record(actual: str, expected: list[str]) -> None:
    result = record(0, actual)
    assert isinstance(result, p.Success)
    assert result.value == expected
