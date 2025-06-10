"""
This is the "Hello World" of parsing, a calculator using polish notation
<https://en.wikipedia.org/wiki/Polish_notation>.

Basic math operations like addition, substraction, multiplication and division
are implemented over float numbers.
"""

from collections.abc import Callable, Sequence

import uparser as p

type Operator = Callable[[float, float], float]
type Token = None | float | Operator


def expression_to_list(result: Sequence[Token | list[Token]]) -> list[Token]:
    list_: list[Token] = []
    for expr in result:
        match expr:
            case float() | Callable():
                list_.append(expr)
            case Sequence():
                list_.extend(expression_to_list(expr))
            case _:
                pass
    return list_


def polish_notation_parser() -> p.Parser[str, list[Token]]:
    def operator_parser(op: str, fn: Operator) -> p.Parser[str, Operator]:
        fn.__name__ = fn.__qualname__ = f"OP: {op}"
        return p.map_value(p.atom(op), lambda _: fn)

    spaces = p.map_value(p.regex(r"[ ]+"), lambda _: None)
    number = p.map_value(p.regex(r"[+-]?\d+(\.\d+)?([eE][+-]?\d+)?"), float)

    operator = p.choice(
        operator_parser("+", lambda a, b: b + a),
        operator_parser("-", lambda a, b: b - a),
        operator_parser("*", lambda a, b: b * a),
        operator_parser("/", lambda a, b: b / a),
        operator_parser("^", lambda a, b: b**a),
    )

    expression: p.Reference[str, list[Token]] = p.Reference()

    number_or_expression = p.choice(number, expression)
    expression_decl = p.sequence(
        operator, spaces, number_or_expression, spaces, number_or_expression
    )
    expression_decl = p.map_error(expression_decl, lambda _: "expected an expression")
    expression_decl = p.map_value(expression_decl, expression_to_list)
    expression.set(expression_decl)

    return p.skip1(expression, p.eof())


def evaluate(list_: list[Token]) -> float:
    stack: list[float] = []
    for token in reversed(list_):
        match token:
            case float():
                stack.append(token)
            case Callable():
                stack.append(token(stack.pop(), stack.pop()))
            case _:
                message = f"unexpected token {token!r}"
                raise ValueError(message)
        print(repr(token), stack)
    return stack.pop()


if __name__ == "__main__":
    import sys

    def calculate(equation: str) -> float:
        match polish_notation_parser()(0, equation):
            case p.Failure(position, error):
                sys.stderr.write(f"ERROR: {error} at position {position}\n")
                sys.stderr.flush()
                sys.exit(1)
            case p.Success(position, list_):
                return evaluate(list_)

    print(f"RESULT: {calculate('+ / ^ ^ 3 2 - 5 1 * 2 4 3')}")
