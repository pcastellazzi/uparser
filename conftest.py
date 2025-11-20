import re
import types

import pytest

import uparser


@pytest.fixture(autouse=True)
def doctests_prelude(
    doctest_namespace: dict[str, types.ModuleType],
) -> None:
    doctest_namespace["p"] = uparser
    doctest_namespace["re"] = re
