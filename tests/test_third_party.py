import re
from typing import TypeVar, cast

import numpy as np
import pytest
import torch as tr
from beartype import beartype
from beartype.roar import (
    BeartypeCallHintParamViolation,
    BeartypeCallHintReturnViolation,
)

from phantom_tensors import dim_binding_scope, parse
from phantom_tensors.alphabet import A, B, C
from phantom_tensors.errors import ParseError
from phantom_tensors.numpy import NDArray
from phantom_tensors.torch import Tensor

T = TypeVar("T")


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
    ],
)
def test_parse_inconsistent_types(tensor_type_pairs):
    with pytest.raises(ParseError):
        parse(*tensor_type_pairs)


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
