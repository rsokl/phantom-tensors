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
from typing_extensions import TypeVarTuple, Unpack as U

from phantom import Phantom
from phantom_tensors import dim_binding_scope, parse
from phantom_tensors.alphabet import A, B, C
from phantom_tensors.array import SupportsArray as Array
from phantom_tensors.errors import ParseError
from phantom_tensors.numpy import NDArray
from phantom_tensors.torch import Tensor
from tests.arrlike import arr

T = TypeVar("T")
Ts = TypeVarTuple("Ts")


class One_to_Three(int, Phantom, predicate=lambda x: 0 < x < 4): ...


class Ten_or_Eleven(int, Phantom, predicate=lambda x: 10 <= x <= 11): ...


class EvenOnly(int, Phantom, predicate=lambda x: x % 2 == 0): ...


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


@pytest.mark.parametrize(
    "tensor_type_pairs",
    [
        (tr.ones(2), NDArray[int]),  # type mismatch
        (np.ones((2,)), Tensor[int]),  # type mismatch
        (arr(10, 2, 10), Array[One_to_Three, int, Ten_or_Eleven]),
        (arr(10, 2, 10), Array[NewOneToThree, int, Ten_or_Eleven]),
        (arr(2, 2, 8), Array[NewOneToThree, int, Ten_or_Eleven]),
        (arr(0, 10), Array[One_to_Three, U[Ts], Ten_or_Eleven]),  # type: ignore
        (arr(2, 2, 0), Array[One_to_Three, U[Ts], Ten_or_Eleven]),  # type: ignore
        (arr(2, 0, 0, 0), Array[One_to_Three, U[Ts], Ten_or_Eleven]),  # type: ignore
        (arr(0, 0, 2, 0), Array[U[Ts], One_to_Three, Ten_or_Eleven]),  # type: ignore
    ],
)
def test_parse_inconsistent_types(tensor_type_pairs):
    with pytest.raises(ParseError):
        parse(*tensor_type_pairs)


def test_phantom_checks():
    assert not isinstance(np.ones((2,)), Tensor[int])  # type: ignore
    assert not isinstance(tr.ones((2,)), NDArray[int])  # type: ignore

    assert not isinstance(tr.ones((2, 2)), Tensor[int])  # type: ignore
    assert not isinstance(np.ones((2, 2)), NDArray[int])  # type: ignore


def test_type_var_with_beartype():
    @dim_binding_scope
    @beartype
    def diag(sqr: Tensor[T, T]) -> Tensor[T]:
        return cast(Tensor[T], tr.diag(sqr))

    non_sqr = parse(tr.ones(2, 3), Tensor[A, B])
    with pytest.raises(BeartypeCallHintParamViolation):
        diag(non_sqr)  # type: ignore


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
    # x  # type revealed: Tensor[A, B]
    # y  # type revealed: Tensor[B, C]

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
    def needs_vector(x: Tensor[int]): ...

    x, y = parse(
        (tr.ones(3, 4), Tensor[A, B]),
        (tr.ones(4, 5), Tensor[B, C]),
    )
    # x  # type revealed: Tensor[A, B]
    # y  # type revealed: Tensor[B, C]

    z = matrix_multiply(x, y)
    # z  # type revealed: Tensor[A, C]

    with pytest.raises(Exception):
        needs_vector(z)  # type: ignore

    with pytest.raises(Exception):
        matrix_multiply(x, x)  # type: ignore


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


@pytest.mark.parametrize(
    "tensor_type_pairs",
    [
        (arr(2, 2, 10), Array[One_to_Three, int, Ten_or_Eleven]),
        (arr(2, 10), Array[One_to_Three, U[Ts], Ten_or_Eleven]),  # type: ignore
        (arr(2, 2, 10), Array[One_to_Three, U[Ts], Ten_or_Eleven]),  # type: ignore
        (arr(2, 0, 0, 10), Array[One_to_Three, U[Ts], Ten_or_Eleven]),  # type: ignore
        (arr(2, 2, 10), Array[NewOneToThree, int, Ten_or_Eleven]),
        (arr(0, 0, 2, 11), Array[U[Ts], One_to_Three, Ten_or_Eleven]),  # type: ignore
    ],
)
def test_parse_consistent_types(tensor_type_pairs):
    parse(*tensor_type_pairs)


def test_non_binding_subint_dims_pass():
    parse(arr(2, 4, 6), Array[EvenOnly, EvenOnly, EvenOnly])
    parse(
        (arr(2, 4), Array[EvenOnly, EvenOnly]),
        (arr(6, 8), Array[EvenOnly, EvenOnly]),
    )


def test_non_binding_subint_dims_validates():

    with pytest.raises(
        ParseError,
        match=re.escape(
            r"shape-(2, 1) doesn't match shape-type (EvenOnly=?, EvenOnly=?)"
        ),
    ):
        parse(arr(2, 1), Array[EvenOnly, EvenOnly])

    with pytest.raises(
        ParseError,
        match=re.escape(
            r"shape-(6, 3) doesn't match shape-type (EvenOnly=?, EvenOnly=?)"
        ),
    ):
        parse(
            (arr(2, 4), Array[EvenOnly, EvenOnly]),
            (arr(6, 3), Array[EvenOnly, EvenOnly]),
        )


def test_binding_validated_dims_validates():

    with pytest.raises(
        ParseError,
        match=re.escape(
            r"shape-(1, 2, 3) doesn't match shape-type (NewOneToThree=1, NewOneToThree=1, NewOneToThree=1)"
        ),
    ):
        parse(arr(1, 2, 3), Array[NewOneToThree, NewOneToThree, NewOneToThree])

    with pytest.raises(
        ParseError,
        match=re.escape(
            r"shape-(1, 2) doesn't match shape-type (NewOneToThree=1, NewOneToThree=1)"
        ),
    ):
        parse(
            (arr(1, 1), Array[NewOneToThree, NewOneToThree]),
            (arr(1, 2), Array[NewOneToThree, NewOneToThree]),
        )


@pytest.mark.parametrize("good_shape", [(1, 1), (2, 2), (3, 3)])
def test_binding_validated_dims_passes(good_shape):
    parse(arr(*good_shape), Array[NewOneToThree, NewOneToThree])
