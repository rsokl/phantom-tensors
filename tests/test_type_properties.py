from __future__ import annotations

from typing import Tuple

from hypothesis import assume, given
from typing_extensions import Literal


def test_literal_singlet_tuple_same_as_scalar():
    assert Literal[(1,)] is Literal[1]  # type: ignore


@given(...)
def test_literals_can_check_by_identity(a: Tuple[int, ...], b: Tuple[int, ...]):
    assume(len(a))
    assume(len(b))
    assume(a != b)
    assert Literal[a] is Literal[a]  # type: ignore
    assert Literal[b] is Literal[b]  # type: ignore
    assert Literal[a] is not Literal[b]  # type: ignore