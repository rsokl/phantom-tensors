from beartype import beartype
import re
import torch as tr
import numpy as np
from phantom_tensors.torch import Tensor
from phantom_tensors.numpy import NDArray
from phantom_tensors import dim_binding_scope, parse
from phantom_tensors.errors import ParseError

from typing import NewType, cast
import pytest

A = NewType("A", int)
B = NewType("B", int)
C = NewType("C", int)


def test_NDArray():
    x = parse(np.ones((2,)), NDArray[A])
    with pytest.raises(ParseError):
        parse(np.ones((2, 3)), NDArray[A, A])


def test_Tensor():
    x = parse(tr.ones((2,)), Tensor[A])
    with pytest.raises(ParseError):
        parse(tr.ones((2, 3)), Tensor[A, A])


@pytest.mark.parametrize(
    "tensor_type_pairs",
    [
        (tr.ones(2), Tensor[int, int]),
        (tr.ones(2, 4), Tensor[A, A]),
        (tr.ones(2, 1, 1), Tensor[A, B, A]),
        (tr.ones(1), Tensor[A, B, C]),
        ((tr.ones(2, 4), Tensor[A, A]),),
        (
            (tr.ones(4), Tensor[A]),
            (tr.ones(5), Tensor[A]),
        ),
    ],
)
def test_parse_consistency(tensor_type_pairs):
    with pytest.raises(ParseError):
        parse(*tensor_type_pairs)


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

from beartype.roar import BeartypeCallHintReturnViolation
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
