import pytest
from calculator import calculate


@pytest.mark.parametrize(
    ("equation", "expected"),
    [
        ("1", 1.0),
        ("+ 1 2", 3.0),
        ("- 5 2", 3.0),
        ("* 2 3", 6.0),
        ("/ 6 2", 3.0),
        ("^ 2 3", 8.0),
        ("+ 1 * 2 3", 7.0),
        ("- / 10 2 + 1 1", 3.0),
        ("+ 3 / * 4 2 ^ - 1 5 ^ 2 3", 3.0001220703125),
        ("  + 1   2  ", 3.0),
        ("-10", -10.0),
        ("1.5", 1.5),
        ("+ 1.5 2.5", 4.0),
    ],
)
def test_calculate_success(equation: str, expected: float) -> None:
    assert calculate(equation) == pytest.approx(expected)  # pyright: ignore[reportUnknownMemberType]


@pytest.mark.parametrize(
    ("equation", "error_message"),
    [
        ("", "Error: expected an expression at position 0"),
        ("1 2", "Error: expected an expression at position 2"),
        ("+ 1", "Error: expected an expression at position 0"),
        ("+ 1 2 3", "Error: expected an expression at position 6"),
        ("?", "Error: expected an expression at position 0"),
        ("+ 1 ?", "Error: expected an expression at position 0"),
    ],
)
def test_calculate_failure(equation: str, error_message: str) -> None:
    with pytest.raises(ValueError, match=error_message):
        calculate(equation)
