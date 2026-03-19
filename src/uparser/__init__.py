"""
*uparser* is a minimalistic parser combinator library using a functional
approach, sum types and generics. My idea of a test case for the new type
hinting features introduced in Python 3.12.

The examples provided below assume the following prelude:

``` py
import re
import uparser as p
```
"""

from uparser.combinators import INFINITY, bind, choice, repeat, sequence, transform
from uparser.core import Failure, Parser, State, Success, parser_hook
from uparser.parsers import atom, eof, regex
from uparser.shortcuts import (
    EOF,
    many0,
    many1,
    map_error,
    map_value,
    option,
    set,  # noqa: A004
    set_error,
    set_value,
    skip_left,
    skip_right,
)
from uparser.util import Reference, run

__all__ = (
    "EOF",
    "INFINITY",
    "Failure",
    "Parser",
    "Reference",
    "State",
    "Success",
    "atom",
    "bind",
    "choice",
    "eof",
    "many0",
    "many1",
    "map_error",
    "map_value",
    "option",
    "parser_hook",
    "regex",
    "repeat",
    "run",
    "sequence",
    "set",
    "set_error",
    "set_value",
    "skip_left",
    "skip_right",
    "transform",
)
