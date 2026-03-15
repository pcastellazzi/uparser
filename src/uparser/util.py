from typing import assert_never

from uparser.core import Failure, Parser, State, Success


def run[F, S](element: Parser[F, S], text: str) -> S:
    """
    Run a parser on the given text, returning the value on success or raising
    `ValueError` on failure.

    Parameters:
        element: Parser to run.
        text: Input text to parse.

    Examples:
        >>> p.run(p.atom("A"), "A")
        'A'

        >>> p.run(p.atom("A"), "B")
        Traceback (most recent call last):
        ...
        ValueError: Error: A at position 0
    """
    match element(0, text):
        case Success(_, value):
            return value
        case Failure(index, error):
            message = f"Error: {error} at position {index}"
            raise ValueError(message)
        case _ as other:
            assert_never(other)


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
