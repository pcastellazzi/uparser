"""
*uparser* is a minimalistic parser combinator library using a functional
approach, sum types and generics. My idea of a test case for the new type
hinting features introduced in Python 3.12.

The examples provided below assume the following prelude:

``` py
import re
import uparser as p
```
"""

from collections.abc import Callable
from dataclasses import dataclass
from functools import cache, wraps
from re import Pattern
from re import compile as re_compile
from sys import maxsize
from typing import TYPE_CHECKING, assert_never, final

if TYPE_CHECKING:  # needed for pdoc
    from typing import TypeVar

    F = TypeVar("F")
    S = TypeVar("S")

__all__ = (
    "INFINITY",
    "Failure",
    "Parser",
    "Reference",
    "State",
    "Success",
    "atom",
    "bind",
    "choice",
    "eof",
    "many0",
    "many1",
    "map",
    "map_error",
    "map_value",
    "optional",
    "parser_hook",
    "regex",
    "repeat",
    "sequence",
    "set",
    "set_error",
    "set_value",
    "skip1",
    "skip2",
)

INFINITY = maxsize
"""Sentinel value for infinite repetitions."""


@final
@dataclass(frozen=True)
class Failure[F]:
    """
    Container for failed results, errors in this context.

    Attributes:
        index: Position on the input where the error was found.
        error: Error found.

    Examples:
        >>> # A Failure can be a string.
        >>> p.Failure(0, "a string literal")
        Failure(index=0, error='a string literal')

        >>> # A Failure can be an arbitrary object.
        >>> p.Failure(1, ["a", "list", "of", "strings"])
        Failure(index=1, error=['a', 'list', 'of', 'strings'])

        >>> # A Failure can be a custom object.
        >>> from enum import Enum, auto
        >>> class MyError(Enum):
        ...     EOF = auto()
        ...     INVALID = auto()
        >>> p.Failure(2, MyError.EOF)
        Failure(index=2, error=<MyError.EOF: 1>)
    """

    index: int
    error: F


@final
@dataclass(frozen=True)
class Success[S]:
    """
    Container for successful results, values in this context.

    Attributes:
        index: Position on the input where the value was found.
        value: Value parsed.

    Examples:
        >>> # A Success can be a string.
        >>> p.Success(0, "a string literal")
        Success(index=0, value='a string literal')

        >>> # A Success can be an arbitrary object.
        >>> p.Success(1, ["a", "list", "of", "strings"])
        Success(index=1, value=['a', 'list', 'of', 'strings'])

        >>> # A Success can be a custom object.
        >>> from enum import Enum, auto
        >>> class Token(Enum):
        ...     IF = auto()
        ...     WHILE = auto()
        >>> p.Success(2, Token.WHILE)
        Success(index=2, value=<Token.WHILE: 2>)
    """

    index: int
    value: S


type State[F, S] = Failure[F] | Success[S]
"""The expected result of a parser."""

type Parser[F, S] = Callable[[int, str], State[F, S]]
"""The parser contract."""


def parser_hook[F, S](
    caller: Callable[..., Parser[F, S]],
) -> Callable[[Parser[F, S]], Parser[F, S]]:
    """
    A decorator applied to all parsers created by the library.

    This implementation uses [functools.wraps][] and [functools.cache][] to
    enable a nice output when debugging and a caching mechanism to avoid
    recalculations.

    Parameters:
        caller: A parser generator.

    Examples:
        >>> import uparser as p
        >>> original_parser_hook = p.parser_hook

        >>> # Default behavior.
        >>> parser = p.atom("A")
        >>> parser.__name__
        'atom'
        >>> parser.cache_info()
        CacheInfo(hits=0, misses=0, maxsize=None, currsize=0)

        >>> # Disable the hook using an identity function.
        >>> def disable_hooks(caller):
        ...     return lambda p: p
        >>> p.parser_hook = disable_hooks
        >>> parser = p.atom("A")
        >>> parser.__name__
        'parser'
        >>> hasattr(parser, "cache_info")
        False

        >>> p.parser_hook = original_parser_hook
    """
    return lambda parser: wraps(caller)(cache(parser))


def atom(expected: str) -> Parser[str, str]:
    """
    A parser for a text literal.

    Parameters:
        expected: The text literal to search for.

    Examples:
        >>> # A parser to match the keyword "if".
        >>> parser = p.atom("if")
        >>> parser(0, "while")
        Failure(index=0, error='if')
        >>> parser(0, "if")
        Success(index=2, value='if')
    """

    @parser_hook(atom)
    def parser(index: int, text: str) -> State[str, str]:
        if text.startswith(expected, index):
            return Success(index + len(expected), expected)
        return Failure(index, expected)

    return parser


def eof() -> Parser[str, str]:
    """
    A parser to check if all the available input has been consumed.

    Examples:
        >>> parser = p.eof()

        >>> # There is input remaining.
        >>> parser(0, "some input")
        Failure(index=0, error='')

        >>> # No more input available.
        >>> parser(10, "some input")
        Success(index=10, value='')

        >>> # An empty string is considered consumed.
        >>> parser(0, "")
        Success(index=0, value='')
    """

    @parser_hook(eof)
    def parser(index: int, text: str) -> State[str, str]:
        if index >= len(text):
            return Success(index, "")
        return Failure(index, "")

    return parser


def regex(expression: str | Pattern[str]) -> Parser[str, str]:
    r"""
    A parser for a regular expression.

    This parser uses [re.match][] and it always returns the complete matched
    text ([re.Match.group][](0)). Using precompiled regular expression is
    allowed to use custom flags, like [re.VERBOSE][].

    Parameters:
        expression: A string to be compiled as a regular expression or an
            already compiled regular expression.

    Examples:
        >>> # A parser to match digits using a regular expression in string form.
        >>> parser = p.regex(r"\d+")
        >>> parser(0, "letters")
        Failure(index=0, error='\\d+')
        >>> parser(0, "1234")
        Success(index=4, value='1234')

        >>> # A parser to match digits using a precompiled regular expression.
        >>> parser = p.regex(re.compile(r"\d+ # comment", re.VERBOSE))
        >>> parser(0, "letters")
        Failure(index=0, error='\\d+ # comment')
        >>> parser(0, "1234")
        Success(index=4, value='1234')
    """

    pattern = re_compile(expression)

    @parser_hook(regex)
    def parser(index: int, text: str) -> State[str, str]:
        if match := pattern.match(text, index):
            return Success(match.end(), match.group(0))
        return Failure(index, pattern.pattern)

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

    @parser_hook(choice)
    def parser(index: int, text: str) -> State[list[F], S]:
        failures: list[F] = []
        for option in options:
            state = option(index, text)
            if isinstance(state, Failure):
                failures.append(state.error)
            else:
                return state
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
    """

    maximum = maximum or minimum

    @parser_hook(repeat)
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

        return Success(index, values)

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

    @parser_hook(sequence)
    def parser(index: int, text: str) -> State[F, list[S]]:
        current_index = index
        successes: list[S] = []

        for element in elements:
            state = element(current_index, text)
            if isinstance(state, Success):
                current_index = state.index
                successes.append(state.value)
            else:
                return state

        return Success(current_index, successes)

    return parser


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
    return parser_hook(many0)(repeat(element, 0, INFINITY))


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
    return parser_hook(many1)(repeat(element, 1, INFINITY))


def optional[F, S, S1](element: Parser[F, S], *, default: S) -> Parser[F, S]:
    """
    Repeat a parser zero or one time.

    Parameters:
        element: Parser to repeat.
        default: Value to use if the parser did not match.

    Examples:
        >>> # A parser to match an optional "A".
        >>> parser = p.optional(p.atom("A"), default="<default>")
        >>> parser(0, "BBBB")
        Success(index=0, value='<default>')
        >>> parser(0, "AAAA")
        Success(index=1, value='A')
    """

    return parser_hook(optional)(
        map_value(repeat(element, 0, 1), lambda v: v[0] if v else default)
    )


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

    @parser_hook(bind)
    def parser(index: int, text: str) -> State[F, S1]:
        state = element(index, text)
        if isinstance(state, Success):
            return fn(state.value)(state.index, text)
        return state

    return parser


def map[F, S, F1, S1](  # noqa: A001
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
        >>> number = p.map(p.regex(r"\\d+"), str2int)
        >>> number(0, "123")
        Success(index=3, value=123)
        >>> number(0, "abc")
        Failure(index=0, error='\\\\d+')
    """

    @parser_hook(map)
    def parser(index: int, text: str) -> State[F1, S1]:
        return fn(element(index, text))

    return parser


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
    return parser_hook(map_error)(
        map(
            element,
            lambda state: (
                Failure(state.index, mapper(state.error))
                if isinstance(state, Failure)
                else state
            ),
        )
    )


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
    return parser_hook(map_value)(
        map(
            element,
            lambda state: (
                Success(state.index, mapper(state.value))
                if isinstance(state, Success)
                else state
            ),
        )
    )


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

    return parser_hook(set)(map(element, mapper))


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
    return parser_hook(set_error)(map_error(element, lambda _: error))


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
    return parser_hook(set_value)(map_value(element, lambda _: value))


def skip1[F, S1, S2](one: Parser[F, S1], two: Parser[F, S2]) -> Parser[F, S2]:
    """
    Generates a parser combining two parsers sequentially. The value produced
    by the first is discarded.

    Parameters:
        one: First parser.
        two: Second parser.

    Examples:
        >>> # A parser to match letters followed by numbers and keeping only
        >>> # the letters.
        >>> letters = p.map_error(p.regex(r"[A-Z]+"), lambda e: "letters")
        >>> numbers = p.map_error(p.regex(r"[0-9]+"), lambda e: "numbers")
        >>> parser = p.skip1(letters, numbers)
        >>> parser(0, "1234")
        Failure(index=0, error='letters')
        >>> parser(0, "ABCD")
        Failure(index=4, error='numbers')
        >>> parser(0, "ABCD1234")
        Success(index=8, value='1234')
    """
    return parser_hook(skip1)(bind(one, lambda _: two))


def skip2[F, S1, S2](one: Parser[F, S1], two: Parser[F, S2]) -> Parser[F, S1]:
    """
    Generates a parser combining two parsers sequentially. The value produced
    by the second is discarded.

    Parameters:
        one: First parser.
        two: Second parser.

    Examples:
        >>> # A parser to match letters followed by numbers and keeping only
        >>> # the numbers.
        >>> letters = p.map_error(p.regex(r"[A-Z]+"), lambda e: "letters")
        >>> numbers = p.map_error(p.regex(r"[0-9]+"), lambda e: "numbers")
        >>> parser = p.skip2(letters, numbers)
        >>> parser(0, "1234")
        Failure(index=0, error='letters')
        >>> parser(0, "ABCD")
        Failure(index=4, error='numbers')
        >>> parser(0, "ABCD1234")
        Success(index=8, value='ABCD')
    """
    return parser_hook(skip2)(
        bind(one, lambda value1: map_value(two, lambda _: value1))
    )


class Reference[F, S]:
    """
    A reference is a parser that is not evaluated immediately. This is useful
    to create recursion.

    Examples:
        >>> type RecursiveSequence = str | list[RecursiveSequence]

        >>> # A parser to match "0" surrounded by one or more pair of brackets.
        >>> ref: p.Reference[RecursiveSequence, RecursiveSequence] = p.Reference()
        >>> recursive = p.choice(p.atom("0"), ref)
        >>> brackets = p.sequence(p.atom("{"), recursive, p.atom("}"))
        >>> # `ValueError` is raised if the reference is not set before use.
        >>> brackets(0, "{{0}}")
        Traceback (most recent call last):
        ...
        ValueError: reference not set
        >>> # Set the reference.
        >>> ref.set(brackets)
        >>> brackets(0, "{0}")
        Success(index=3, value=['{', '0', '}'])
        >>> brackets(0, "{{0}}")
        Success(index=5, value=['{', ['{', '0', '}'], '}'])
        >>> brackets(0, "{{{0}}}")
        Success(index=7, value=['{', ['{', ['{', '0', '}'], '}'], '}'])
    """

    def __init__(self) -> None:
        self._reference: Parser[F, S] | None = None

    def __call__(self, index: int, text: str) -> State[F, S]:
        if self._reference:
            return self._reference(index, text)
        message = "reference not set"
        raise ValueError(message)

    def set(self, parser: Parser[F, S]) -> None:
        self._reference = parser
