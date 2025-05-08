"""
Parser combinator library using a functional approach, sum types and generics.

The library revolves around two types. `Result` and `Parser`.

A `Parser` is a `Callable` that accepts two parameters; the first is an `int`
to indicate the starting position of the action, and the second a `str` in
which the action will be performed. This `Callable` returns a `Result`.

A `Result` is the sum type of a `Failure` and a `Success`. The first stores a
`position` and an `error`, the second stores a `position` and a `value`. The
difference names reflect semantic differences, in both cases the type of the
object is given by the user. While you can create `Result` instances directly,
the user of the library is expected to use `map_failure` and `map_success`
instead unless he is creating his own parser.
"""

from functools import lru_cache, wraps
from re import compile as recompile
from sys import maxsize
from typing import TYPE_CHECKING, NamedTuple, assert_never

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from re import Pattern

__all__ = (
    "INFINITY",
    "Failure",
    "Parser",
    "Reference",
    "Result",
    "Success",
    "atom",
    "cache",
    "compose_left",
    "compose_right",
    "eof",
    "many0",
    "many1",
    "map_failure",
    "map_success",
    "oneof",
    "optional",
    "regexp",
    "repeat",
    "sequence",
)


class Failure[F](NamedTuple):
    """
    Container for failed results, errors in this context.

    Parameters:
        * position: Position on the input where the error was found.
        * error: Error found.

    Examples:
        >>> import uparser as p

        # A string literal.
        >>> p.Failure(0, "some error")
        Failure(position=0, error='some error')

        # A list of strings.
        >>> p.Failure(1, ["some", "other", "error"])
        Failure(position=1, error=['some', 'other', 'error'])

        # A custom enumerated type.
        >>> from enum import Enum, auto
        >>> class MyError(Enum):
        ...     EOF = auto()
        ...     INVALID = auto()
        >>> p.Failure(2, MyError.EOF)
        Failure(position=2, error=<MyError.EOF: 1>)
    """

    position: int
    error: F


class Success[S](NamedTuple):
    """
    Container for successful results, values in this context.

    Parameters:
        * position: Position on the input where the value was found.
        * value: Value parsed.

    Examples:
        >>> import uparser as p

        # A string literal.
        >>> p.Success(0, "some value")
        Success(position=0, value='some value')

        # A list of strings.
        >>> p.Success(1, ["some", "other", "value"])
        Success(position=1, value=['some', 'other', 'value'])

        # A custom enumerated type.
        >>> from enum import Enum, auto
        >>> class Token(Enum):
        ...     IF = auto()
        ...     WHILE = auto()
        >>> p.Success(2, Token.WHILE)
        Success(position=2, value=<Token.WHILE: 2>)
    """

    position: int
    value: S


type Result[F, S] = Failure[F] | Success[S]
type Parser[F, S] = "Callable[[int, str], Result[F, S]]"

INFINITY = maxsize
"""Sentinel value for infinite repetitions."""

cache = lru_cache(maxsize=128, typed=True)
"""
Result caching strategy.

Caching is enabled by default using `functools.lru_cache`. The default
strategy can be change by setting `uparser.cache` to the strategy of your
choice.

```python
import uparser as p
from functools import cache as memoize


def disable_caching[F, S](parser: p.Parser[F, S]) -> p.Parser[F, S]:
    return parser


# Use an identity function to disable caching.
p.cache = disable_caching

# Replace the default strategy for `functools.cache`.
p.cache = memoize
```
"""


def atom(expected: str) -> Parser[str, str]:
    """
    A parser for a text literal.

    Parameters:
        * expected: The text literal to search for.

    Examples:
        >>> import uparser as p

        # A parser to match the keyword "if".
        >>> parser = p.atom("if")

        >>> parser(0, "while")
        Failure(position=0, error='if')

        >>> parser(0, "if")
        Success(position=2, value='if')
    """

    @wraps(atom)
    @cache
    def parser(position: int, text: str) -> Result[str, str]:
        if text.startswith(expected, position):
            return Success(position + len(expected), expected)
        return Failure(position, expected)

    return parser


def eof() -> Parser[str, str]:
    """
    A parser to check if all the available input has been consumed.

    Examples:
        >>> import uparser as p
        >>> parser = p.eof()

        # There is input remaining.
        >>> parser(0, "some input")
        Failure(position=0, error='')

        # No more input available.
        >>> parser(10, "some input")
        Success(position=10, value='')

        # An empty string is considered consumed.
        >>> parser(0, "")
        Success(position=0, value='')
    """

    @wraps(eof)
    @cache
    def parser(position: int, text: str) -> Result[str, str]:
        if position >= len(text):
            return Success(position, "")
        return Failure(position, "")

    return parser


def regexp(expression: "str | Pattern[str]") -> Parser[str, str]:
    r"""
    A parser for a regular expression.

    This parser uses `re.match` and it always returns the complete matched
    text (`re.match.group(0)`). Using precompiled regular expression is
    allowed to use custom flags, like `re.VERBOSE`.

    Parameters:
        * expression: A string to be compiled as a regular expression or an
          already compiled regular expression.

    Examples:
        >>> import re
        >>> import uparser as p

        # A parser to match digits using a string.
        >>> parser1 = p.regexp(r"\d+")
        >>> parser1(0, "letters")
        Failure(position=0, error='\\d+')
        >>> parser1(0, "1234")
        Success(position=4, value='1234')

        # A parser to match digits using a precompiled regular expression.
        >>> parser2 = p.regexp(re.compile(r"\d+ # comment", re.VERBOSE))
        >>> parser2(0, "letters")
        Failure(position=0, error='\\d+ # comment')
        >>> parser2(0, "1234")
        Success(position=4, value='1234')
    """
    pattern = recompile(expression)  # already compiled patterns are a noop

    @wraps(regexp)
    @cache
    def parser(position: int, text: str) -> Result[str, str]:
        if match := pattern.match(text, position):
            return Success(match.end(), match.group(0))
        return Failure(position, pattern.pattern)

    return parser


def oneof[F, S](*parsers: Parser[F, S]) -> "Parser[Sequence[F], S]":
    """
    A parser for an ordered choice.

    The first `Success` is returned immediately. Otherwise a `Failure` with
    the accumulated errors is returned.

    Parameters:
        * parsers: One or more parsers to evaluate in order.

    Examples:
        >>> import uparser as p

        # A parser to match "A" or "B".
        >>> parser = p.oneof(p.atom("A"), p.atom("B"))

        >>> parser(0, "CC")
        Failure(position=0, error=['A', 'B'])

        >>> parser(0, "BB")
        Success(position=1, value='B')

        >>> parser(0, "AA")
        Success(position=1, value='A')
    """

    @wraps(oneof)
    @cache
    def parser_(position: int, text: str) -> "Result[Sequence[F], S]":
        failures: list[F] = []
        for parser in parsers:
            match parser(position, text):
                case Success() as success:
                    return success
                case Failure(_, error):
                    failures.append(error)
                case _:
                    assert_never()
        return Failure(position, failures)

    return parser_


def repeat[F, S](
    parser: Parser[F, S], minimum: int, maximum: int | None = None
) -> "Parser[F, Sequence[S]]":
    """
    A parser to handle repetitions.

    Parameters:
        * parser: Parser to repeat.
        * minimum: Minimum number of iterations.
        * maximum: An optional maximum number of iterations.

    Examples:
        >>> import uparser as p

        # Zero or more times.
        >>> parser1 = p.repeat(p.atom("A"), 0, p.INFINITY)
        >>> parser1(0, "BBBB")
        Success(position=0, value=[])
        >>> parser1(0, "AAAA")
        Success(position=4, value=['A', 'A', 'A', 'A'])

        # One or more times.
        >>> parser2 = p.repeat(p.atom("A"), 1, p.INFINITY)
        >>> parser2(0, "BBBB")
        Failure(position=0, error='A')
        >>> parser2(0, "AAAA")
        Success(position=4, value=['A', 'A', 'A', 'A'])

        # Zero or one time.
        >>> parser3 = p.repeat(p.atom("A"), 0, 1)
        >>> parser3(0, "BBBB")
        Success(position=0, value=[])
        >>> parser3(0, "AAAA")
        Success(position=1, value=['A'])

        # Between 2 and 4 times.
        >>> parser4 = p.repeat(p.atom("A"), 2, 4)
        >>> parser4(0, "BBBB")
        Failure(position=0, error='A')
        >>> parser4(0, "AAAA")
        Success(position=4, value=['A', 'A', 'A', 'A'])
    """
    maximum = maximum or minimum

    @wraps(repeat)
    @cache
    def parser_(position: int, text: str) -> "Result[F, Sequence[S]]":
        current_position = position
        iterations = 0
        values: list[S] = []

        while iterations < maximum:
            match parser(current_position, text):
                case Success(position, value):
                    current_position = position
                    iterations += 1
                    values.append(value)
                case Failure() as failure:
                    if iterations >= minimum:
                        break
                    return failure
                case _:
                    assert_never()

        return Success(position, values)

    return parser_


def sequence[F, S](*parsers: Parser[F, S]) -> "Parser[F, Sequence[S]]":
    """
    A parser for sequencing actions.

    The first `Failure` is returned immediately. Otherwise a `Success` with
    the accumulated values is returned.

    Parameters:
        * parsers: One or more parsers to evaluate in order.

    Examples:
        >>> import uparser as p

        # A parser to match "A" followed by "B".
        >>> parser = p.sequence(p.atom("A"), p.atom("B"))

        >>> parser(0, "CCCC")
        Failure(position=0, error='A')

        >>> parser(0, "ACCC")
        Failure(position=1, error='B')

        >>> parser(0, "ABCD")
        Success(position=2, value=['A', 'B'])
    """

    @wraps(sequence)
    @cache
    def parser_(position: int, text: str) -> "Result[F, Sequence[S]]":
        last_position = position
        successes: list[S] = []

        for parser in parsers:
            match parser(last_position, text):
                case Success(position, value):
                    last_position = position
                    successes.append(value)
                case Failure() as failure:
                    return failure
                case _:
                    assert_never()

        return Success(last_position, successes)

    return parser_


def many0[F, S](parser: Parser[F, S]) -> "Parser[F, Sequence[S]]":
    """
    Repeat a parser zero or more times.

    Parameters:
        * parser: Parser to repeat.

    Examples:
        >>> import uparser as p

        # A parser to match an optional sequence of "A".
        >>> parser = p.many0(p.atom("A"))

        >>> parser(0, "BBBB")
        Success(position=0, value=[])

        >>> parser(0, "AAAA")
        Success(position=4, value=['A', 'A', 'A', 'A'])
    """
    return wraps(many0)(repeat(parser, 0, INFINITY))


def many1[F, S](parser: Parser[F, S]) -> "Parser[F, Sequence[S]]":
    """
    Repeat a parser one or more times.

    Parameters:
        * parser: Parser to repeat.

    Examples:
        >>> import uparser as p

        # A parser to match at least one "A".
        >>> parser = p.many1(p.atom("A"))

        >>> parser(0, "BBBB")
        Failure(position=0, error='A')

        >>> parser(0, "AAAA")
        Success(position=4, value=['A', 'A', 'A', 'A'])
    """
    return wraps(many1)(repeat(parser, 1, INFINITY))


def optional[F, S](parser: Parser[F, S]) -> Parser[F, S | None]:
    """
    Repeat a parser zero or one time.

    Parameters:
        * parser: Parser to repeat.

    Examples:
        >>> import uparser as p

        # A parser to match an optional "A".
        >>> parser = p.optional(p.atom("A"))

        >>> parser(0, "BBBB")
        Success(position=0, value=None)

        >>> parser(0, "AAAA")
        Success(position=1, value='A')
    """
    optional_ = repeat(parser, 0, 1)

    @wraps(optional)
    @cache
    def parser_(position: int, text: str) -> Result[F, S | None]:
        match optional_(position, text):
            case Failure() as failure:  # pragma: no cover
                return failure
            case Success() as success:
                return Success(
                    success.position, success.value[0] if success.value else None
                )
            case _:
                assert_never()

    return parser_


class Reference[F, S]:
    """
    A reference is a parser that is not evaluated immediately. This is usefull
    to create recursion.

    Examples:
        >>> import uparser as p
        >>> from collections.abc import Sequence
        >>> type RecursiveSequence = str | Sequence[RecursiveSequence]

        # A parser to match "0" surrounded by one or more pair of brackets.
        >>> ref: p.Reference[RecursiveSequence, RecursiveSequence] = p.Reference()
        >>> recursive = p.oneof(p.atom("0"), ref)
        >>> brackets = p.sequence(p.atom("{"), recursive, p.atom("}"))

        # `ValueError` is raised if the reference is not set before use.
        >>> brackets(0, "{{0}}")
        Traceback (most recent call last):
         ...
        ValueError: reference not set

        # Set the reference.
        >>> ref.set(brackets)

        >>> brackets(0, "{0}")
        Success(position=3, value=['{', '0', '}'])

        >>> brackets(0, "{{0}}")
        Success(position=5, value=['{', ['{', '0', '}'], '}'])

        >>> brackets(0, "{{{0}}}")
        Success(position=7, value=['{', ['{', ['{', '0', '}'], '}'], '}'])
    """

    def __init__(self) -> None:
        self._reference: Parser[F, S] | None = None

    def __call__(self, position: int, text: str) -> Result[F, S]:
        if self._reference:
            return self._reference(position, text)
        message = "reference not set"
        raise ValueError(message)

    def set(self, parser: Parser[F, S]) -> None:
        self._reference = parser


def compose_left[F, S1, S2](
    first: Parser[F, S1], second: Parser[F, S2]
) -> Parser[F, S1]:
    """
    Generates a parser combining two parsers sequentially. The value produced
    by the second is discarded.

    Parameters:
        * first: First parser.
        * second: Second parser.

    Examples:
        >>> import uparser as p

        # A parser to match letters followed by numbers and keeping only the
        # numbers.
        >>> letters = p.map_failure(p.regexp(r"[A-Z]+"), lambda e: "letters")
        >>> numbers = p.map_failure(p.regexp(r"[0-9]+"), lambda e: "numbers")
        >>> parser = p.compose_left(letters, numbers)

        >>> parser(0, "1234")
        Failure(position=0, error='letters')

        >>> parser(0, "ABCD")
        Failure(position=4, error='numbers')

        >>> parser(0, "ABCD1234")
        Success(position=8, value='ABCD')
    """

    @wraps(compose_left)
    @cache
    def parser_(position: int, text: str) -> Result[F, S1]:
        first_result = first(position, text)
        if isinstance(first_result, Failure):
            return first_result

        second_result = second(first_result.position, text)
        if isinstance(second_result, Failure):
            return second_result

        return Success(second_result.position, first_result.value)

    return parser_


def compose_right[F, S1, S2](
    first: Parser[F, S1], second: Parser[F, S2]
) -> Parser[F, S2]:
    """
    Generates a parser combining two parsers sequentially. The value produced
    by the first is discarded.

    Parameters:
        * first: First parser.
        * second: Second parser.

    Examples:
        >>> import uparser as p

        # A parser to match letters followed by numbers and keeping only the
        # letters.
        >>> letters = p.map_failure(p.regexp(r"[A-Z]+"), lambda e: "letters")
        >>> numbers = p.map_failure(p.regexp(r"[0-9]+"), lambda e: "numbers")
        >>> parser = p.compose_right(letters, numbers)

        >>> parser(0, "1234")
        Failure(position=0, error='letters')

        >>> parser(0, "ABCD")
        Failure(position=4, error='numbers')

        >>> parser(0, "ABCD1234")
        Success(position=8, value='1234')
    """

    @wraps(compose_right)
    @cache
    def parser_(position: int, text: str) -> Result[F, S2]:
        match first(position, text):
            case Failure() as failure:
                return failure
            case Success(pos, _):
                return second(pos, text)
            case _:
                assert_never()

    return parser_


def map_failure[F, F1, S](
    parser: Parser[F, S], fn: "Callable[[F], F1]"
) -> Parser[F1, S]:
    """
    Alters the error of a parser.

    Parameters:
        * parser: Parser to alter.
        * fn: Function (Callable) making the alteration.

    Examples:
        >>> import uparser as p

        # A parser to match "A" with a custom error.
        >>> parser = p.atom("A")
        >>> parser = p.map_failure(parser, lambda e: f"expected: {e!r}")

        >>> parser(0, "B")
        Failure(position=0, error="expected: 'A'")

        # Original `value` is not altered.
        >>> parser(0, "A")
        Success(position=1, value='A')
    """

    @wraps(map_failure)
    @cache
    def parser_(position: int, text: str) -> Result[F1, S]:
        match parser(position, text):
            case Failure(pos, err):
                return Failure(pos, fn(err))
            case Success() as success:
                return success
            case _:
                assert_never()

    return parser_


def map_success[F, S, S1](
    parser: Parser[F, S], fn: "Callable[[S], S1]"
) -> Parser[F, S1]:
    """
    Alters the value of a parser.

    Parameters:
        * parser: Parser to alter.
        * fn: Function (Callable) making the alteration.

    Examples:
        >>> import uparser as p

        # A parser to match "A" returning a custom value.
        >>> parser = p.atom("A")
        >>> parser = p.map_success(parser, lambda _: "got A")

        >>> parser(0, "A")
        Success(position=1, value='got A')

        # Original `error` is not altered.
        >>> parser(0, "B")
        Failure(position=0, error='A')
    """

    @wraps(map_success)
    @cache
    def parser_(position: int, text: str) -> Result[F, S1]:
        match parser(position, text):
            case Failure() as failure:
                return failure
            case Success(pos, val):
                return Success(pos, fn(val))
            case _:
                assert_never()

    return parser_
