from itertools import chain

import pytest

import uparser as p

TYPES = "Failure", "Success", "Result", "Parser"
CONSTANTS = ("INFINITY",)
PARSERS = "atom", "eof", "regexp"
COMBINATORS = "oneof", "repeat", "sequence"
SHORTCUTS = "many0", "many1", "optional"
UTIL = "Reference", "compose_left", "compose_left", "map_failure", "map_success"


def assert_decorated[F, S](parser: p.Parser[F, S], *, name: str) -> None:
    assert parser.__name__ == name  # functools.wraps applied correctly
    assert hasattr(parser, "cache_info")  # functools.cache applied correctly


def test_public_api_visibility() -> None:
    for export in chain(TYPES, CONSTANTS, PARSERS, COMBINATORS, SHORTCUTS, UTIL):
        assert export in p.__all__


def test_parsers_should_be_decorated() -> None:
    assert_decorated(p.atom("atom"), name="atom")
    assert_decorated(p.eof(), name="eof")
    assert_decorated(p.regexp("regexp"), name="regexp")

    assert_decorated(p.oneof(p.atom("atom")), name="oneof")
    assert_decorated(p.repeat(p.atom("atom"), 1), name="repeat")
    assert_decorated(p.sequence(p.atom("atom")), name="sequence")

    assert_decorated(p.many0(p.atom("atom")), name="many0")
    assert_decorated(p.many1(p.atom("atom")), name="many1")
    assert_decorated(p.optional(p.atom("atom")), name="optional")

    assert_decorated(
        p.compose_left(p.atom("atom"), p.atom("atom")), name="compose_left"
    )
    assert_decorated(
        p.compose_right(p.atom("atom"), p.atom("atom")), name="compose_right"
    )
    assert_decorated(
        p.map_failure(p.atom("atom"), lambda _: "atom"), name="map_failure"
    )
    assert_decorated(
        p.map_success(p.atom("atom"), lambda _: "atom"), name="map_success"
    )


def test_caching_strategy(monkeypatch: pytest.MonkeyPatch) -> None:
    def disable_caching[F, S](parser: p.Parser[F, S]) -> p.Parser[F, S]:
        return parser

    # Setting `uparser.cache` is expected behavior. `pytest.MonkeyPatch` is
    # used to automatically undo the change when this test finishes.
    monkeypatch.setattr(p, "cache", disable_caching)
    assert not hasattr(p.atom("atom"), "cache_info")
