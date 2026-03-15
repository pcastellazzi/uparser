from re import Pattern
from re import compile as re_compile

from uparser.core import Failure, Parser, State, Success, parser_hook


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

    @parser_hook.get()(atom)
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

    @parser_hook.get()(eof)
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

    @parser_hook.get()(regex)
    def parser(index: int, text: str) -> State[str, str]:
        if match := pattern.match(text, index):
            return Success(match.end(), match.group(0))
        return Failure(index, pattern.pattern)

    return parser
