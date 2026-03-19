from collections.abc import Callable
from sys import maxsize
from typing import assert_never

from uparser.core import Failure, Parser, State, Success, parser_hook

INFINITY = maxsize
"""Sentinel value for infinite repetitions."""


def bind[F, S, S1](
    element: Parser[F, S], fn: Callable[[S], Parser[F, S1]]
) -> Parser[F, S1]:
    """
    Chain two parsers sequentially, where the second parser is determined by
    the result of the first.

    Parameters:
        element: the first parser to apply
        fn: A function taking a Success and returns a new parser to apply.

    Examples:
        >>> # Parse a number, then parse that many 'A's.
        >>> def letter(count: int) -> p.Parser[str, str]:
        ...     return p.map_value(
        ...         p.repeat(p.atom("A"), count, count), lambda v: "".join(v)
        ...     )
        >>> count = p.map_value(p.regex(r"\\d+"), lambda v: int(v))
        >>> parser = p.bind(count, letter)
        >>> parser(0, "3AAA")
        Success(index=4, value='AAA')
        >>> parser(0, "#AAA")
        Failure(index=0, error='\\\\d+')
        >>> parser(0, "3AAB")
        Failure(index=3, error='A')
    """

    @parser_hook.get()(bind)
    def parser(index: int, text: str) -> State[F, S1]:
        match element(index, text):
            case Success(index, value):
                return fn(value)(index, text)
            case Failure() as failure:
                return failure
            case _ as other:
                assert_never(other)

    return parser


def choice[F, S](*options: Parser[F, S]) -> Parser[list[F], S]:
    """
    A parser for an ordered choice.

    The first [uparser.Success][] is returned immediately. Otherwise a
    [uparser.Failure][] with the accumulated errors is returned.

    Parameters:
        options: One or more parsers to evaluate in order.

    Examples:
        >>> # A parser to match "A" or "B".
        >>> parser = p.choice(p.atom("A"), p.atom("B"))
        >>> parser(0, "CC")
        Failure(index=0, error=['A', 'B'])
        >>> parser(0, "BB")
        Success(index=1, value='B')
        >>> parser(0, "AA")
        Success(index=1, value='A')

        >>> # An empty choice is an automatic failure.
        >>> parser = p.choice()
        >>> parser(0, "AA")
        Failure(index=0, error=[])
    """

    @parser_hook.get()(choice)
    def parser(index: int, text: str) -> State[list[F], S]:
        failures: list[F] = []
        for option in options:
            match option(index, text):
                case Failure(_, error):
                    failures.append(error)
                case Success() as success:
                    return success
                case _ as other:
                    assert_never(other)

        return Failure(index, failures)

    return parser


def repeat[F, S](
    element: Parser[F, S], minimum: int, maximum: int | None = None
) -> Parser[F, list[S]]:
    """
    A parser to handle repetitions.

    Parameters:
        element: Parser to repeat.
        minimum: Minimum number of iterations.
        maximum: An optional maximum number of iterations.

    Examples:
        >>> # Zero or more times.
        >>> parser1 = p.repeat(p.atom("A"), 0, p.INFINITY)
        >>> parser1(0, "BBBB")
        Success(index=0, value=[])
        >>> parser1(0, "AAAA")
        Success(index=4, value=['A', 'A', 'A', 'A'])

        >>> # One or more times.
        >>> parser2 = p.repeat(p.atom("A"), 1, p.INFINITY)
        >>> parser2(0, "BBBB")
        Failure(index=0, error='A')
        >>> parser2(0, "AAAA")
        Success(index=4, value=['A', 'A', 'A', 'A'])

        >>> # Zero or one time.
        >>> parser3 = p.repeat(p.atom("A"), 0, 1)
        >>> parser3(0, "BBBB")
        Success(index=0, value=[])
        >>> parser3(0, "AAAA")
        Success(index=1, value=['A'])

        >>> # Between 2 and 4 times.
        >>> parser4 = p.repeat(p.atom("A"), 2, 4)
        >>> parser4(0, "BBBB")
        Failure(index=0, error='A')
        >>> parser4(0, "AAAA")
        Success(index=4, value=['A', 'A', 'A', 'A'])

        >>> parser5 = p.repeat(p.atom("A"), 4, 2)
        Traceback (most recent call last):
        ...
        ValueError: minimum (4) must be <= maximum (2)
    """

    if maximum is None:
        maximum = minimum

    if minimum > maximum:
        message = f"minimum ({minimum}) must be <= maximum ({maximum})"
        raise ValueError(message)

    @parser_hook.get()(repeat)
    def parser(index: int, text: str) -> State[F, list[S]]:
        current_index = index
        iterations = 0
        values: list[S] = []

        while iterations < maximum:
            match element(current_index, text):
                case Success(index, value):
                    current_index = index
                    values.append(value)
                    iterations += 1
                case Failure() as failure:
                    if iterations >= minimum:
                        break
                    return failure
                case _ as other:
                    assert_never(other)

        return Success(current_index, values)

    return parser


def sequence[F, S](*elements: Parser[F, S]) -> Parser[F, list[S]]:
    """
    A parser for sequencing actions.

    The first [uparser.Failure][] is returned immediately. Otherwise a
    [uparser.Success][] with the accumulated values is returned.

    Parameters:
        elements: One or more parsers to evaluate in order.

    Examples:
        >>> # A parser to match "A" followed by "B".
        >>> parser = p.sequence(p.atom("A"), p.atom("B"))
        >>> parser(0, "CCC")
        Failure(index=0, error='A')
        >>> parser(0, "ACC")
        Failure(index=1, error='B')
        >>> parser(0, "ABC")
        Success(index=2, value=['A', 'B'])

        >>> # An empty sequence is an automatic success.
        >>> parser = p.sequence()
        >>> parser(0, "CCC")
        Success(index=0, value=[])
    """

    @parser_hook.get()(sequence)
    def parser(index: int, text: str) -> State[F, list[S]]:
        current_index = index
        successes: list[S] = []

        for element in elements:
            match element(current_index, text):
                case Success(index, value):
                    current_index = index
                    successes.append(value)
                case Failure() as failure:
                    return failure
                case _ as other:
                    assert_never(other)

        return Success(current_index, successes)

    return parser


def transform[F, S, F1, S1](
    element: Parser[F, S], fn: Callable[[State[F, S]], State[F1, S1]]
) -> Parser[F1, S1]:
    """
    Alters the state of a parser using a transformation function.

    Parameters:
        element: The parser whose state will be transformed.
        fn: The transformation function.

    Examples:
        >>> def str2int(state: p.State[str, str]) -> p.State[str, int]:
        ...     if isinstance(state, p.Success):
        ...         return p.Success(state.index, int(state.value))
        ...     return state
        >>> number = p.transform(p.regex(r"\\d+"), str2int)
        >>> number(0, "123")
        Success(index=3, value=123)
        >>> number(0, "abc")
        Failure(index=0, error='\\\\d+')
    """

    @parser_hook.get()(transform)
    def parser(index: int, text: str) -> State[F1, S1]:
        return fn(element(index, text))

    return parser
