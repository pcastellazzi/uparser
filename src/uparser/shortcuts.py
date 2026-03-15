from collections.abc import Callable
from typing import assert_never

from uparser.combinators import INFINITY, bind, repeat, transform
from uparser.core import Failure, Parser, State, Success, parser_hook
from uparser.parsers import eof

EOF: Parser[str, str] = eof()
"""Pre-built parser for end-of-input. Equivalent to ``eof()``."""


def many0[F, S](element: Parser[F, S]) -> Parser[F, list[S]]:
    """
    Repeat a parser zero or more times.

    Parameters:
        element: Parser to repeat.

    Examples:
        >>> # A parser to match an optional sequence of "A".
        >>> parser = p.many0(p.atom("A"))
        >>> parser(0, "BBBB")
        Success(index=0, value=[])
        >>> parser(0, "AAAA")
        Success(index=4, value=['A', 'A', 'A', 'A'])
    """
    return parser_hook.get()(many0)(repeat(element, 0, INFINITY))


def many1[F, S](element: Parser[F, S]) -> Parser[F, list[S]]:
    """
    Repeat a parser one or more times.

    Parameters:
        element: Parser to repeat.

    Examples:
        >>> # A parser to match at least one "A".
        >>> parser = p.many1(p.atom("A"))
        >>> parser(0, "BBBB")
        Failure(index=0, error='A')
        >>> parser(0, "AAAA")
        Success(index=4, value=['A', 'A', 'A', 'A'])
    """
    return parser_hook.get()(many1)(repeat(element, 1, INFINITY))


def option[F, S](element: Parser[F, S], *, default: S) -> Parser[F, S]:
    """
    Try a parser, returning a default value if it does not match.

    Parameters:
        element: Parser to try.
        default: Value to use if the parser did not match.

    Examples:
        >>> # A parser to match an optional "A".
        >>> parser = p.option(p.atom("A"), default="<default>")
        >>> parser(0, "BBBB")
        Success(index=0, value='<default>')
        >>> parser(0, "AAAA")
        Success(index=1, value='A')
    """

    return parser_hook.get()(option)(
        map_value(repeat(element, 0, 1), lambda v: v[0] if v else default)
    )


def map_error[F, F1, S](
    element: Parser[F, S], mapper: Callable[[F], F1]
) -> Parser[F1, S]:
    """
    Alters the error of a parser.

    Parameters:
        element: Parser to alter.
        mapper: Function ([collections.abc.Callable][]) making the alteration.

    Examples:
        >>> # A parser to match "A" with a custom error.
        >>> parser = p.atom("A")
        >>> parser = p.map_error(parser, lambda e: f"expected: {e!r}")
        >>> parser(0, "B")
        Failure(index=0, error="expected: 'A'")
        >>> parser(0, "A")
        Success(index=1, value='A')
    """

    def wrapper(state: State[F, S]) -> State[F1, S]:
        match state:
            case Failure(index, error):
                return Failure(index, mapper(error))
            case Success() as success:
                return success
            case _ as other:
                assert_never(other)

    return parser_hook.get()(map_error)(transform(element, wrapper))


def map_value[F, S, S1](
    element: Parser[F, S], mapper: Callable[[S], S1]
) -> Parser[F, S1]:
    """
    Alters the value of a parser.

    Parameters:
        element: Parser to alter.
        mapper: Function ([collections.abc.Callable][]) making the alteration.

    Examples:
        >>> # A parser to match "A" returning a custom value.
        >>> parser = p.atom("A")
        >>> parser = p.map_value(parser, lambda _: "got an A")
        >>> parser(0, "A")
        Success(index=1, value='got an A')
        >>> parser(0, "B")
        Failure(index=0, error='A')
    """

    def wrapper(state: State[F, S]) -> State[F, S1]:
        match state:
            case Success(index, value):
                return Success(index, mapper(value))
            case Failure() as failure:
                return failure
            case _ as other:
                assert_never(other)

    return parser_hook.get()(map_value)(transform(element, wrapper))


def set[F, S, F1, S1](element: Parser[F, S], error: F1, value: S1) -> Parser[F1, S1]:  # noqa: A001
    """
    Alters the error and value of a parser.

    Parameters:
        element: Parser to alter.
        error: New error.
        value: New value.

    Examples:
        >>> # A parser to match "A" returning a custom value.
        >>> parser = p.atom("A")
        >>> parser = p.set(parser, "some error", "some value")
        >>> parser(0, "B")
        Failure(index=0, error='some error')
        >>> parser(0, "A")
        Success(index=1, value='some value')
    """

    def mapper(state: State[F, S]) -> State[F1, S1]:
        match state:
            case Failure(index, _):
                return Failure(index, error)
            case Success(index, _):
                return Success(index, value)
            case _ as other:
                assert_never(other)

    return parser_hook.get()(set)(transform(element, mapper))


def set_error[F, F1, S](element: Parser[F, S], error: F1) -> Parser[F1, S]:
    """
    Alters the error of a parser.

    Parameters:
        element: Parser to alter.
        error: New error.

    Examples:
        >>> # A parser to match "A" with a custom error.
        >>> parser = p.atom("A")
        >>> parser = p.set_error(parser, "expected an A")
        >>> parser(0, "B")
        Failure(index=0, error='expected an A')
        >>> parser(0, "A")
        Success(index=1, value='A')
    """
    return parser_hook.get()(set_error)(map_error(element, lambda _: error))


def set_value[F, S, S1](element: Parser[F, S], value: S1) -> Parser[F, S1]:
    """
    Alters the value of a parser.

    Parameters:
        element: Parser to alter.
        value: New value.

    Examples:
        >>> # A parser to match "A" returning a custom value.
        >>> parser = p.atom("A")
        >>> parser = p.set_value(parser, "got an A")
        >>> parser(0, "A")
        Success(index=1, value='got an A')
        >>> parser(0, "B")
        Failure(index=0, error='A')
    """
    return parser_hook.get()(set_value)(map_value(element, lambda _: value))


def skip_left[F, S1, S2](one: Parser[F, S1], two: Parser[F, S2]) -> Parser[F, S2]:
    """
    Generates a parser combining two parsers sequentially. The value produced
    by the first (left) is discarded.

    Parameters:
        one: First parser.
        two: Second parser.

    Examples:
        >>> # A parser to match letters followed by numbers and keeping only
        >>> # the numbers.
        >>> letters = p.map_error(p.regex(r"[A-Z]+"), lambda e: "letters")
        >>> numbers = p.map_error(p.regex(r"[0-9]+"), lambda e: "numbers")
        >>> parser = p.skip_left(letters, numbers)
        >>> parser(0, "1234")
        Failure(index=0, error='letters')
        >>> parser(0, "ABCD")
        Failure(index=4, error='numbers')
        >>> parser(0, "ABCD1234")
        Success(index=8, value='1234')
    """
    return parser_hook.get()(skip_left)(bind(one, lambda _: two))


def skip_right[F, S1, S2](one: Parser[F, S1], two: Parser[F, S2]) -> Parser[F, S1]:
    """
    Generates a parser combining two parsers sequentially. The value produced
    by the second (right) is discarded.

    Parameters:
        one: First parser.
        two: Second parser.

    Examples:
        >>> # A parser to match letters followed by numbers and keeping only
        >>> # the letters.
        >>> letters = p.map_error(p.regex(r"[A-Z]+"), lambda e: "letters")
        >>> numbers = p.map_error(p.regex(r"[0-9]+"), lambda e: "numbers")
        >>> parser = p.skip_right(letters, numbers)
        >>> parser(0, "1234")
        Failure(index=0, error='letters')
        >>> parser(0, "ABCD")
        Failure(index=4, error='numbers')
        >>> parser(0, "ABCD1234")
        Success(index=8, value='ABCD')
    """
    return parser_hook.get()(skip_right)(
        bind(one, lambda value1: map_value(two, lambda _: value1))
    )
