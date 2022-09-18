import re
from typing import NewType, TypeVar, cast

import numpy as np
import pytest
import torch as tr
from beartype import beartype
from beartype.roar import (
    BeartypeCallHintParamViolation,
    BeartypeCallHintReturnViolation,
)
from typing_extensions import Literal as L, TypeVarTuple, Unpack as U

from phantom import Phantom
from phantom_tensors import dim_binding_scope, parse
from phantom_tensors.errors import ParseError
from phantom_tensors.numpy import NDArray
from phantom_tensors.torch import Tensor

T = TypeVar("T")
Ts = TypeVarTuple("Ts")
A = NewType("A", int)
B = NewType("B", int)
C = NewType("C", int)


class One_to_Three(int, Phantom, predicate=lambda x: 0 < x < 4):
    ...


class Ten_or_Eleven(int, Phantom, predicate=lambda x: 10 <= x <= 11):
    ...


NewOneToThree = NewType("NewOneToThree", One_to_Three)


def test_NDArray():
    assert issubclass(NDArray, np.ndarray)
    assert issubclass(NDArray[A], np.ndarray)

    parse(np.ones((2,)), NDArray[A])
    with pytest.raises(ParseError):
        parse(np.ones((2, 3)), NDArray[A, A])


def test_Tensor():
    assert issubclass(Tensor, tr.Tensor)
    assert issubclass(Tensor[A], tr.Tensor)

    parse(tr.ones((2,)), Tensor[A])
    with pytest.raises(ParseError):
        parse(tr.ones((2, 3)), Tensor[A, A])


def test_parse_error_msg():
    with pytest.raises(
        ParseError,
        match=re.escape("shape-(2, 1) doesn't match shape-type ([...], A=2, A=2)"),
    ):
        parse(np.ones((2, 1)), NDArray[U[Ts], A, A])


@pytest.mark.parametrize(
    "tensor_type_pairs",
    [
        (tr.ones(()), Tensor[()]),
        (tr.ones(()), Tensor[U[Ts]]),
        (tr.ones(2), Tensor[A]),
        (tr.ones(2), Tensor[U[Ts]]),
        (tr.ones(2), Tensor[U[Ts], A]),
        (tr.ones(2), Tensor[A, U[Ts]]),
        (tr.ones(2, 2), Tensor[A, A]),
        (tr.ones(2, 2), Tensor[U[Ts], A, A]),
        (tr.ones(2, 2), Tensor[A, U[Ts], A]),
        (tr.ones(2, 2), Tensor[A, A, U[Ts]]),
        (tr.ones(1, 3, 2, 2), Tensor[U[Ts], A, A]),
        (tr.ones(2, 1, 3, 2), Tensor[A, U[Ts], A]),
        (tr.ones(2, 2, 1, 3), Tensor[A, A, U[Ts]]),
        (tr.ones(1, 2, 1, 3, 2), Tensor[A, B, U[Ts], B]),
        (tr.ones(1, 2, 3), Tensor[int, int, int]),
        (tr.ones(2, 1, 2), Tensor[A, B, A]),
        (tr.ones(2, 1, 3), Tensor[A, B, C]),
        (
            (tr.ones(5), Tensor[A]),
            (tr.ones(5, 2), Tensor[A, B]),
        ),
        (tr.ones(1), Tensor[L[1]]),
        (tr.ones(3), Tensor[L[1, 2, 3]]),
        (tr.ones(1, 2), Tensor[L[1], L[2]]),
        (tr.ones(1, 2, 1), Tensor[L[1], L[2], L[1]]),
        (tr.ones(2, 2, 10), Tensor[One_to_Three, int, Ten_or_Eleven]),
        (tr.ones(2, 10), Tensor[One_to_Three, U[Ts], Ten_or_Eleven]),
        (tr.ones(2, 2, 10), Tensor[One_to_Three, U[Ts], Ten_or_Eleven]),
        (tr.ones(2, 0, 0, 10), Tensor[One_to_Three, U[Ts], Ten_or_Eleven]),
        (tr.ones(2, 2, 10), Tensor[NewOneToThree, int, Ten_or_Eleven]),
        (tr.ones(0, 0, 2, 11), Tensor[U[Ts], One_to_Three, Ten_or_Eleven]),
    ],
)
def test_parse_consistent_types(tensor_type_pairs):
    parse(*tensor_type_pairs)


@pytest.mark.parametrize(
    "tensor_type_pairs",
    [
        (tr.ones(2), NDArray[int]),  # type mismatch
        (np.ones((2,)), Tensor[int]),  # type mismatch
        (tr.ones(()), Tensor[int]),
        (tr.ones(()), Tensor[int, U[Ts]]),
        (tr.ones(2), Tensor[int, int]),
        (tr.ones(2, 4), Tensor[A, A]),
        (tr.ones(2, 1, 1), Tensor[A, B, A]),
        (tr.ones(2, 1, 1), Tensor[A, U[Ts], A]),
        (tr.ones(1), Tensor[A, B, C]),
        ((tr.ones(2, 4), Tensor[A, A]),),
        (
            (tr.ones(4), Tensor[A]),
            (tr.ones(5), Tensor[A]),
        ),
        (tr.ones(3), Tensor[L[1]]),
        (tr.ones(2, 2), Tensor[L[1], L[2]]),
        (tr.ones(1, 1), Tensor[L[1], L[2]]),
        (tr.ones(1, 1, 1), Tensor[L[1], L[2], L[1]]),
        (tr.ones(10, 2, 10), Tensor[One_to_Three, int, Ten_or_Eleven]),
        (tr.ones(10, 2, 10), Tensor[NewOneToThree, int, Ten_or_Eleven]),
        (tr.ones(2, 2, 8), Tensor[NewOneToThree, int, Ten_or_Eleven]),
        (tr.ones(0, 10), Tensor[One_to_Three, U[Ts], Ten_or_Eleven]),
        (tr.ones(2, 2, 0), Tensor[One_to_Three, U[Ts], Ten_or_Eleven]),
        (tr.ones(2, 0, 0, 0), Tensor[One_to_Three, U[Ts], Ten_or_Eleven]),
        (tr.ones(0, 0, 2, 0), Tensor[U[Ts], One_to_Three, Ten_or_Eleven]),
    ],
)
def test_parse_inconsistent_types(tensor_type_pairs):
    with pytest.raises(ParseError):
        parse(*tensor_type_pairs)


def test_type_var():
    @dim_binding_scope
    @beartype
    def diag(sqr: Tensor[T, T]) -> Tensor[T]:
        return cast(Tensor[T], tr.diag(sqr))

    non_sqr = parse(tr.ones(2, 3), Tensor[A, B])
    with pytest.raises(BeartypeCallHintParamViolation):
        diag(non_sqr)  # type: ignore


def test_catches_wrong_instance():
    with pytest.raises(
        ParseError,
        match=re.escape(
            "Expected <class 'numpy.ndarray'>, got: <class 'torch.Tensor'>"
        ),
    ):
        parse(tr.tensor(1), NDArray[A, B])

    with pytest.raises(
        ParseError,
        match=re.escape(
            "Expected <class 'torch.Tensor'>, got: <class 'numpy.ndarray'>"
        ),
    ):
        parse(np.array(1), Tensor[A])


def test_isinstance_works():
    with dim_binding_scope:

        assert isinstance(tr.ones(2), Tensor[A])  # type: ignore
        assert not isinstance(tr.ones(3), Tensor[A])  # type: ignore
        assert isinstance(tr.ones(2), Tensor[A])  # type: ignore

        assert isinstance(tr.ones(2, 4), Tensor[A, B])  # type: ignore
        assert not isinstance(tr.ones(2), Tensor[B])  # type: ignore
        assert isinstance(tr.ones(4), Tensor[B])  # type: ignore
        assert isinstance(tr.ones(4, 2, 2, 4), Tensor[B, A, A, B])  # type: ignore

    assert isinstance(tr.ones(1, 3, 3, 1), Tensor[B, A, A, B])  # type: ignore
    assert isinstance(tr.ones(1, 4, 4, 1), Tensor[B, A, A, B])  # type: ignore


def test_parse_in_and_out_of_binding_scope():
    with dim_binding_scope:

        parse(tr.ones(2), Tensor[A])  # binds A=2

        with pytest.raises(ParseError):
            parse(tr.ones(3), Tensor[A])

        parse(tr.ones(2), Tensor[A])

        parse(tr.ones(2, 4), Tensor[A, B])  # binds B=4
        parse(tr.ones(2, 9), Tensor[A, int])

        with pytest.raises(ParseError):
            parse(tr.ones(2), Tensor[B])

    # no dims bound
    parse(tr.ones(1, 3, 3, 1), Tensor[B, A, A, B])  # no dims bound
    parse(tr.ones(1, 4, 4, 1), Tensor[B, A, A, B])

    parse(
        (tr.ones(9), Tensor[B]),
        (tr.ones(9, 2, 2, 9), Tensor[B, A, A, B]),
    )


def test_parse_bind_multiple():
    with dim_binding_scope:  # enter dimension-binding scope
        parse(
            (tr.ones(2), Tensor[A]),  # <-binds A=2
            (tr.ones(9), Tensor[B]),  # <-binds B=9
            (tr.ones(9, 2, 9), Tensor[B, A, B]),  # <-checks A & B
        )

        with pytest.raises(
            ParseError,
            match=re.escape("shape-(78,) doesn't match shape-type (A=2,)"),
        ):
            # can't re-bind A within scope
            parse(tr.ones(78), Tensor[A])

        with pytest.raises(
            ParseError,
            match=re.escape("shape-(22,) doesn't match shape-type (B=9,)"),
        ):
            # can't re-bind B within scope
            parse(tr.ones(22), Tensor[B])

        parse(tr.ones(2), Tensor[A])
        parse(tr.ones(9), Tensor[B])

        # exit dimension-binding scope

    parse(tr.ones(78, 22), Tensor[A, B])  # now ok


def test_matmul_example():
    @dim_binding_scope
    @beartype
    def matrix_multiply(x: Tensor[A, B], y: Tensor[B, C]) -> Tensor[A, C]:
        out = x @ x.T
        return cast(Tensor[A, C], out)

    x, y = parse(
        (tr.ones(3, 4), Tensor[A, B]),
        (tr.ones(4, 5), Tensor[B, C]),
    )
    x  # type revealed: Tensor[A, B]
    y  # type revealed: Tensor[B, C]

    with pytest.raises(BeartypeCallHintReturnViolation):
        matrix_multiply(x, y)


def test_runtime_checking_with_beartype():
    @dim_binding_scope
    # ^ ensures A, B, C consistent across all input/output tensor shapes
    #   within scope of function
    @beartype
    def matrix_multiply(x: Tensor[A, B], y: Tensor[B, C]) -> Tensor[A, C]:
        a, b = x.shape
        b, c = y.shape
        return cast(Tensor[A, C], tr.ones(a, c))

    @beartype
    def needs_vector(x: Tensor[int]):
        ...

    x, y = parse(
        (tr.ones(3, 4), Tensor[A, B]),
        (tr.ones(4, 5), Tensor[B, C]),
    )
    x  # type revealed: Tensor[A, B]
    y  # type revealed: Tensor[B, C]

    z = matrix_multiply(x, y)
    z  # type revealed: Tensor[A, C]

    with pytest.raises(Exception):
        needs_vector(z)  # type: ignore

    with pytest.raises(Exception):
        matrix_multiply(x, x)  # type: ignore


AStr = NewType("AStr", str)


@pytest.mark.parametrize(
    "bad_type",
    [
        Tensor[AStr],
        Tensor[str],
        Tensor[int, str],
        Tensor[U[Ts], str],
    ],
)
def test_bad_type_validation(bad_type):
    with pytest.raises(TypeError):
        parse(tr.ones(1), bad_type)
