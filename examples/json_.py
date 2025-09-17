import re
import typing

import uparser as p

type JSONValue = "None | bool | float | str | list[JSONValue] | dict[str, JSONValue]"

RE_NUMBER = re.compile(r"[-]?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?")
RE_STRING = re.compile(r'"(?:[^"\\\x00-\x1F]|\\["\\/bfnrt]|\\u[a-fA-F0-9]{4})*"')
RE_WHITESPACE = re.compile(r"[\n\r\t ]*")

whitespace = p.set(p.regex(RE_WHITESPACE), "whitespace", "whitespace")


def token[S](parser: p.Parser[str, S]) -> p.Parser[str, S]:
    return p.skip1(whitespace, parser)


def string2python(json_string: str) -> str:
    return bytes(json_string[1:-1].replace(r"\/", "/"), "ascii").decode(
        "unicode-escape"
    )


null = token(p.set_value(p.atom("null"), None))
false = token(p.set_value(p.atom("false"), False))  # noqa: FBT003
true = token(p.set_value(p.atom("true"), True))  # noqa: FBT003

number = p.regex(RE_NUMBER)
number = p.map_value(number, float)
number = p.set_error(number, "expected a number")
number = token(number)

string = p.regex(RE_STRING)
string = p.map_value(string, string2python)
string = p.set_error(string, "expected a string")
string = token(string)


def between[S](parser: p.Parser[str, S], *, left: str, right: str) -> p.Parser[str, S]:
    return p.skip1(token(p.atom(left)), p.skip2(parser, token(p.atom(right))))


def comma_separated_list[S](parser: p.Parser[str, S]) -> p.Parser[str, list[S]]:
    comma = token(p.atom(","))
    element = p.map_value(parser, lambda v: [v])
    csp = p.sequence(element, p.many0(p.skip1(comma, parser)))
    csp = p.optional(csp, default=[])
    return p.map_value(csp, lambda v: v[0] + v[1] if v else [])


json_value: p.Reference[str, JSONValue] = p.Reference()

array = between(comma_separated_list(json_value), left="[", right="]")
pair: p.Parser[str, tuple[str, JSONValue]] = p.map_value(
    p.sequence(p.skip2(string, token(p.atom(":"))), json_value),
    lambda v: (typing.cast("str", v[0]), v[1]),
)
object_ = between(comma_separated_list(pair), left="{", right="}")
object_ = p.map_value(object_, lambda v: dict(v))

json_value_def = p.choice(null, true, false, number, string, array, object_)
json_value_def = p.set_error(json_value_def, "expected a JSON object")
json_value.set(json_value_def)


def json_parser(json_string: str) -> JSONValue:
    match json_value(0, json_string):
        case p.Success(_, value):
            return value
        case p.Failure(index, error):
            message = f"Error: {error} at position {index}"
            raise ValueError(message)
        case _ as other:
            typing.assert_never(other)


if __name__ == "__main__":
    sample_json = r"""
    {
        "empty object": {},
        "empty array": [],
        "mixed array": [null, true, false, 123.45, "a string with \"quotes\""],
        "nested structure": {
            "description": "A deeply nested object inside an array",
            "data": [
                { "id": 1, "value": "first" },
                { "id": 2, "value": "second" }
            ]
        },
        "unicode support": "\u2728 \u1F680 \u26A1"
    }
    """
    parsed_result = json_parser(sample_json)
    print(f"RESULT: {parsed_result}")
