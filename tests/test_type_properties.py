from __future__ import annotations

import sys
from typing import Literal, Tuple

import pytest
from hypothesis import assume, given


@pytest.mark.xfail(
    sys.version_info < (3, 9), reason="Tuple normalization introduced in 3.9"
)
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


def test_literal_hashes_consistently():
    assert {Literal[1]: 1}[Literal[1]] == 1
    assert hash(Literal[1, 2]) == hash(Literal[1, 2])


@pytest.mark.parametrize("content", [1, 2, (1, 2)])
def test_literal_args_always_tuples(content):
    assert Literal[content].__args__ == (content,) if not isinstance(content, tuple) else content  # type: ignore
