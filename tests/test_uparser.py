from collections.abc import Callable
from itertools import chain

import pytest

import uparser as p

TYPES = "Failure", "Success", "State", "Parser"
OTHER = "INFINITY", "parser_hook", "map", "bind"
PARSERS = "atom", "eof", "regex"
COMBINATORS = "choice", "repeat", "sequence"
SHORTCUTS = "many0", "many1", "optional", "set", "set_error", "set_value"
UTIL = "Reference", "map_error", "map_value", "skip1", "skip2"


def assert_decorated[F, S](parser: p.Parser[F, S], *, name: str) -> None:
    # functools.wraps applied correctly
    assert getattr(parser, "__name__", "<unknown>") == name
    # functools.cache applied correctly
    assert hasattr(parser, "cache_info")


def test_public_api_visibility() -> None:
    for export in chain(TYPES, OTHER, PARSERS, COMBINATORS, SHORTCUTS, UTIL):
        assert export in p.__all__


def test_parsers_should_be_decorated() -> None:
    assert_decorated(p.bind(p.atom("atom"), lambda _: p.atom("atom2")), name="bind")
    assert_decorated(p.map(p.atom("atom"), lambda s: s), name="map")

    assert_decorated(p.atom("atom"), name="atom")
    assert_decorated(p.eof(), name="eof")
    assert_decorated(p.regex("regex"), name="regex")

    assert_decorated(p.choice(p.atom("atom")), name="choice")
    assert_decorated(p.repeat(p.atom("atom"), 1), name="repeat")
    assert_decorated(p.sequence(p.atom("atom")), name="sequence")

    assert_decorated(p.many0(p.atom("atom")), name="many0")
    assert_decorated(p.many1(p.atom("atom")), name="many1")
    assert_decorated(p.optional(p.atom("atom"), default=""), name="optional")
    assert_decorated(p.set(p.atom("atom"), "", ""), name="set")
    assert_decorated(p.set_error(p.atom("atom"), ""), name="set_error")
    assert_decorated(p.set_value(p.atom("atom"), ""), name="set_value")

    assert_decorated(p.map_error(p.atom("atom"), lambda _: "atom"), name="map_error")
    assert_decorated(p.map_value(p.atom("atom"), lambda _: "atom"), name="map_value")
    assert_decorated(p.skip1(p.atom("atom"), p.atom("atom")), name="skip1")
    assert_decorated(p.skip2(p.atom("atom"), p.atom("atom")), name="skip2")


def test_caching_strategy(monkeypatch: pytest.MonkeyPatch) -> None:
    def disable_caching[F, S](
        _: Callable[..., p.Parser[F, S]],
    ) -> Callable[[p.Parser[F, S]], p.Parser[F, S]]:
        return lambda parser: parser

    # Setting `uparser.cache` is expected behavior. `pytest.MonkeyPatch` is
    # used to automatically undo the change when this test finishes.
    monkeypatch.setattr(p, "parser_hook", disable_caching)
    assert not hasattr(p.atom("atom"), "cache_info")
