from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass
from functools import update_wrapper
from typing import final


@final
@dataclass(frozen=True, slots=True)
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
@dataclass(frozen=True, slots=True)
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


def default_parser_hook[**P, F, S](
    caller: Callable[P, Parser[F, S]],
) -> Callable[[Parser[F, S]], Parser[F, S]]:
    """
    A decorator applied to all parsers created by the library. This is the
    default implementation for [parser_hook][] which uses
    [functools.update_wrapper][] to give parsers easier to debug names.

    Parameters:
        caller: A parser generator.
    """

    def decorator(parser: Parser[F, S]) -> Parser[F, S]:
        return update_wrapper(parser, caller)

    return decorator


parser_hook = ContextVar("parser_hook", default=default_parser_hook)
"""
A decorator applied to all parsers created by the library. The default
implementation is the identity function.


Examples:
    >>> import uparser as p

    >>> # Default behavior.
    >>> parser = p.atom("A")
    >>> parser.__name__
    'atom'

    >>> # Custom hook
    >>> from functools import lru_cache, wraps
    >>> def my_parser_hook(caller):
    >>>     return lambda parser: wraps(caller)(lru_cache(maxsize=32)(parser))
    >>> with p.parser_hook.set(my_parser_hook):
    >>>     parser = p.atom("A")
    >>>     parser.__name__
    'atom'
    >>>     hasattr(parser, "cache_info")
    True
"""
