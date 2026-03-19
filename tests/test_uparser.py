from itertools import chain

import uparser as p

CONSTANTS = ("INFINITY",)
TYPES = "Failure", "Success", "State", "Parser"
PARSERS = "atom", "eof", "regex"
COMBINATORS = "bind", "transform", "choice", "repeat", "sequence"
SHORTCUTS = (
    "many0",
    "many1",
    "map_error",
    "map_value",
    "option",
    "set",
    "set_error",
    "set_value",
    "skip_left",
    "skip_right",
)
UTIL = "parser_hook", "Reference"


def assert_decorated[F, S](parser: p.Parser[F, S], *, name: str) -> None:
    # functools.wraps applied correctly
    assert getattr(parser, "__name__", "<unknown>") == name


def test_public_api_visibility() -> None:
    for export in chain(CONSTANTS, TYPES, PARSERS, COMBINATORS, SHORTCUTS, UTIL):
        assert export in p.__all__


def test_parsers_should_be_decorated() -> None:
    assert_decorated(p.bind(p.atom("atom"), lambda _: p.atom("atom2")), name="bind")
    assert_decorated(p.transform(p.atom("atom"), lambda s: s), name="transform")

    assert_decorated(p.atom("atom"), name="atom")
    assert_decorated(p.eof(), name="eof")
    assert_decorated(p.regex("regex"), name="regex")

    assert_decorated(p.choice(p.atom("atom")), name="choice")
    assert_decorated(p.repeat(p.atom("atom"), 1), name="repeat")
    assert_decorated(p.sequence(p.atom("atom")), name="sequence")

    assert_decorated(p.many0(p.atom("atom")), name="many0")
    assert_decorated(p.many1(p.atom("atom")), name="many1")
    assert_decorated(p.option(p.atom("atom"), default=""), name="option")
    assert_decorated(p.set(p.atom("atom"), "", ""), name="set")
    assert_decorated(p.set_error(p.atom("atom"), ""), name="set_error")
    assert_decorated(p.set_value(p.atom("atom"), ""), name="set_value")

    assert_decorated(p.map_error(p.atom("atom"), lambda _: "atom"), name="map_error")
    assert_decorated(p.map_value(p.atom("atom"), lambda _: "atom"), name="map_value")
    assert_decorated(p.skip_left(p.atom("atom"), p.atom("atom")), name="skip_left")
    assert_decorated(p.skip_right(p.atom("atom"), p.atom("atom")), name="skip_right")
