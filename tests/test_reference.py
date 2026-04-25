from uparser.core import Success
from uparser.util import Reference


def test_reference_identity_check() -> None:
    class FalsyParser:
        def __call__(self, index: int, _: str) -> Success[str]:
            return Success(index + 1, "ok")

        def __bool__(self) -> bool:
            return False

    reference: Reference[None, str] = Reference()
    reference.set(FalsyParser())

    result = reference(0, "test")
    assert isinstance(result, Success)
    assert result.value == "ok"
