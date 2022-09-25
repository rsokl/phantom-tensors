import re
from typing import Any, NewType, TypeVar

import pytest
from typing_extensions import Literal as L, TypeVarTuple, Unpack as U

from phantom_tensors import dim_binding_scope, parse
from phantom_tensors.array import SupportsArray as Array
from phantom_tensors.errors import ParseError
from tests.arrlike import arr

T = TypeVar("T")
Ts = TypeVarTuple("Ts")
A = NewType("A", int)
B = NewType("B", int)
C = NewType("C", int)


def test_parse_error_msg():
    with pytest.raises(
        ParseError,
        match=re.escape("shape-(2, 1) doesn't match shape-type ([...], A=2, A=2)"),
    ):
        parse(arr(2, 1), Array[U[Ts], A, A])


@pytest.mark.parametrize(
    "tensor_type_pairs",
    [
        # (arr(), Array[()]),
        (arr(), Array[U[Ts]]),
        (arr(2), Array[A]),
        (arr(2), Array[int]),
        (arr(2), Array[Any]),
        (arr(2), Array[U[Ts]]),
        (arr(2), Array[U[Ts], A]),
        (arr(2), Array[A, U[Ts]]),
        (arr(2, 2), Array[A, A]),
        (arr(2, 2), Array[U[Ts], A, A]),
        (arr(2, 2), Array[A, U[Ts], A]),
        (arr(2, 2), Array[A, A, U[Ts]]),
        (arr(1, 3, 2, 2), Array[U[Ts], A, A]),
        (arr(2, 1, 3, 2), Array[A, U[Ts], A]),
        (arr(2, 2, 1, 3), Array[A, A, U[Ts]]),
        (arr(1, 2, 1, 3, 2), Array[A, B, U[Ts], B]),
        (arr(1, 2, 3), Array[Any, Any, Any]),
        (arr(1, 2, 3), Array[int, int, int]),
        (arr(2, 1, 2), Array[A, B, A]),
        (arr(2, 1, 3), Array[A, B, C]),
        (
            (arr(5), Array[A]),
            (arr(5, 2), Array[A, B]),
        ),
        (arr(1), Array[L[1]]),
        (arr(3), Array[L[1, 2, 3]]),
        (arr(1, 2), Array[L[1], L[2]]),
        (arr(1, 2, 1), Array[L[1], L[2], L[1]]),
    ],
)
def test_parse_consistent_types(tensor_type_pairs):
    parse(*tensor_type_pairs)


@pytest.mark.parametrize(
    "tensor_type_pairs",
    [
        (arr(), Array[int]),
        (arr(), Array[int, U[Ts]]),
        (arr(2), Array[int, int]),
        (arr(2, 4), Array[A, A]),
        (arr(2, 1, 1), Array[A, B, A]),
        (arr(1, 1, 2), Array[A, A, A]),
        (arr(2, 1, 1), Array[A, U[Ts], A]),
        (arr(1), Array[A, B, C]),
        ((arr(2, 4), Array[A, A]),),
        (
            (arr(4), Array[A]),
            (arr(5), Array[A]),
        ),
        (arr(3), Array[L[1]]),
        (arr(2, 2), Array[L[1], L[2]]),
        (arr(1, 1), Array[L[1], L[2]]),
        (arr(1, 1, 1), Array[L[1], L[2], L[1]]),
    ],
)
def test_parse_inconsistent_types(tensor_type_pairs):
    with pytest.raises(ParseError):
        parse(*tensor_type_pairs)


def test_parse_in_and_out_of_binding_scope():
    with dim_binding_scope:

        parse(arr(2), Array[A])  # binds A=2

        with pytest.raises(ParseError):
            parse(arr(3), Array[A])

        parse(arr(2), Array[A])

        parse(arr(2, 4), Array[A, B])  # binds B=4
        parse(arr(2, 9), Array[A, int])

        with pytest.raises(ParseError):
            parse(arr(2), Array[B])

    # no dims bound
    parse(arr(1, 3, 3, 1), Array[B, A, A, B])  # no dims bound
    parse(arr(1, 4, 4, 1), Array[B, A, A, B])

    parse(
        (arr(9), Array[B]),
        (arr(9, 2, 2, 9), Array[B, A, A, B]),
    )


def test_parse_bind_multiple():
    with dim_binding_scope:  # enter dimension-binding scope
        parse(
            (arr(2), Array[A]),  # <-binds A=2
            (arr(9), Array[B]),  # <-binds B=9
            (arr(9, 2, 9), Array[B, A, B]),  # <-checks A & B
        )

        with pytest.raises(
            ParseError,
            match=re.escape("shape-(78,) doesn't match shape-type (A=2,)"),
        ):
            # can't re-bind A within scope
            parse(arr(78), Array[A])

        with pytest.raises(
            ParseError,
            match=re.escape("shape-(22,) doesn't match shape-type (B=9,)"),
        ):
            # can't re-bind B within scope
            parse(arr(22), Array[B])

        parse(arr(2), Array[A])
        parse(arr(9), Array[B])

        # exit dimension-binding scope

    parse(arr(78, 22), Array[A, B])  # now ok


AStr = NewType("AStr", str)


@pytest.mark.parametrize(
    "bad_type",
    [
        Array[AStr],
        Array[str],
        Array[int, str],
        Array[U[Ts], str],
    ],
)
def test_bad_type_validation(bad_type):
    with pytest.raises(TypeError):
        parse(arr(1), bad_type)
