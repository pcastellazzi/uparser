"""
This is the "Hello World" of parsing, a calculator using polish notation
<https://en.wikipedia.org/wiki/Polish_notation>.

Basic math operations like addition, substraction, multiplication and division
are implemented over floating point (as in python's float) numbers.
"""

import operator as op
from typing import assert_never

import uparser as p

OPERATORS = {
    "+": op.add,
    "-": op.sub,
    "*": op.mul,
    "/": op.truediv,
    "^": op.pow,
}

whitespace = p.regex("[\n\r\t ]*")
whitespace = p.set(whitespace, "whitespace", "whitespace")


def calc(operator: str, value1: float, value2: float) -> float:
    result = OPERATORS[operator](value1, value2)
    print(f"== {operator} ({value1!r}, {value2!r}) => {result!r}")
    return result


def token[S](parser: p.Parser[str, S]) -> p.Parser[str, S]:
    return p.skip1(whitespace, parser)


number = p.regex(r"[+-]?\d+(\.\d+)?([eE][+-]?\d+)?")
number = p.set_error(number, "expected a number")
number = p.map_value(number, float)
number = token(number)

operator = p.choice(*[p.atom(o) for o in OPERATORS])
operator = p.set_error(operator, "expected an operator")
operator = p.map_value(operator, lambda v: v[0])
operator = token(operator)

expression: p.Reference[str, float | str] = p.Reference()
operation = p.sequence(operator, expression, expression)
operation = p.map_value(operation, lambda v: calc(str(v[0]), float(v[1]), float(v[2])))

expression_decl = p.choice(number, operation)
expression_decl = p.set_error(expression_decl, "expected an expression")
expression.set(expression_decl)

parser = p.skip1(whitespace, expression)
parser = p.skip2(parser, whitespace)
parser = p.map_value(p.sequence(parser, p.eof()), lambda v: v[0])


def calculate(equation: str) -> float:
    match parser(0, equation):
        case p.Success(_, value):
            return float(value)
        case p.Failure(index, error):
            message = f"Error: {error} at position {index}"
            raise ValueError(message)
        case _ as other:
            assert_never(other)


if __name__ == "__main__":
    print(f"RESULT: {calculate('+ 3 / * 4 2 ^ - 1 5 ^ 2 3')}")
