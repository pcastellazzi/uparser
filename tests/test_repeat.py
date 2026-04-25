import pytest

import uparser as p


@pytest.mark.timeout(1)
def test_repeat_infinite_loop_prevention() -> None:
    """
    Test that `repeat` does not enter an infinite loop when the underlying
    parser succeeds without consuming any input.
    """

    # Succeeds with "" without consuming input if "A" is not present.
    non_consuming_parser = p.option(p.atom("A"), default="")

    # If not protected, this will loop forever.
    parser = p.repeat(non_consuming_parser, 0, p.INFINITY)

    # Return Success if there is no inifinite loop.
    result = parser(0, "B")
    assert isinstance(result, p.Success)
    assert result.index == 0
    assert result.value == [""]
