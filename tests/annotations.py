# pyright: strict
from __future__ import annotations

import sys
from typing import Tuple as tuple

import torch as tr
from typing_extensions import assert_type

from phantom_tensors import parse
from phantom_tensors.alphabet import A, B
from phantom_tensors.torch import Tensor


def parse_tensors(x: tr.Tensor):
    assert_type(parse(x, Tensor[A]), Tensor[A])
    assert_type(parse((x, Tensor[A])), Tensor[A])
    assert_type(
        parse(
            (x, Tensor[A]),
            (x, Tensor[A, B]),
        ),
        tuple[Tensor[A], Tensor[A, B]],
    )
    assert_type(
        parse(
            (x, Tensor[A]),
            (x, Tensor[A, B]),
            (x, Tensor[B, A]),
        ),
        tuple[Tensor[A], Tensor[A, B], Tensor[B, A]],
    )

    assert_type(
        parse(
            (x, Tensor[A]),
            (x, Tensor[A, B]),
            (x, Tensor[B, A]),
            (x, Tensor[B, B]),
        ),
        tuple[Tensor[A], Tensor[A, B], Tensor[B, A], Tensor[B, B]],
    )

    assert_type(
        parse(
            (x, Tensor[A]),
            (x, Tensor[A, B]),
            (x, Tensor[B, A]),
            (x, Tensor[B, B]),
            (x, Tensor[B]),
        ),
        tuple[Tensor[A], Tensor[A, B], Tensor[B, A], Tensor[B, B], Tensor[B]],
    )

    assert_type(
        parse(
            (x, Tensor[A]),
            (x, Tensor[A, B]),
            (x, Tensor[B, A]),
            (x, Tensor[B, B]),
            (x, Tensor[B]),
            (x, Tensor[A]),
        ),
        tuple[
            Tensor[A], Tensor[A, B], Tensor[B, A], Tensor[B, B], Tensor[B], Tensor[A]
        ],
    )


def check_bad_tensor_parse(x: tr.Tensor):
    parse(1, tuple[A])  # type: ignore
    parse((1,), A)  # type: ignore

    parse(x, A)  # type: ignore
    parse((x, A))  # type: ignore
    parse((x, A), (x, tuple[A]))  # type: ignore


def check_readme_blurb_one():
    import numpy as np

    from phantom_tensors import parse
    from phantom_tensors.numpy import NDArray

    # runtime: checks that shapes (2, 3) and (3, 2)
    #          match (A, B) and (B, A) pattern
    if sys.version_info < (3, 8):
        return

    x, y = parse(
        (np.ones((2, 3)), NDArray[A, B]),
        (np.ones((3, 2)), NDArray[B, A]),
    )

    assert_type(x, NDArray[A, B])
    assert_type(y, NDArray[B, A])


def check_readme2():
    from typing import Any

    import numpy as np

    from phantom_tensors import parse
    from phantom_tensors.alphabet import A, B  # these are just NewType(..., int) types
    from phantom_tensors.numpy import NDArray

    def func_on_2d(x: NDArray[Any, Any]): ...

    def func_on_3d(x: NDArray[Any, Any, Any]): ...

    def func_on_any_arr(x: np.ndarray[Any, Any]): ...

    if sys.version_info < (3, 8):
        return

    # runtime: ensures shape of arr_3d matches (A, B, A) patterns
    arr_3d = parse(np.ones((3, 5, 3)), NDArray[A, B, A])

    func_on_2d(arr_3d)  # type: ignore
    func_on_3d(arr_3d)  # static type checker: OK
    func_on_any_arr(arr_3d)  # static type checker: OK


def check_readme3():
    from typing import TypeVar, cast

    import torch as tr
    from beartype import beartype

    from phantom_tensors import dim_binding_scope
    from phantom_tensors.alphabet import A, B, C
    from phantom_tensors.torch import Tensor

    T1 = TypeVar("T1")
    T2 = TypeVar("T2")
    T3 = TypeVar("T3")

    @dim_binding_scope
    @beartype  # <- adds runtime type checking to function's interfaces
    def buggy_matmul(x: Tensor[T1, T2], y: Tensor[T2, T3]) -> Tensor[T1, T3]:
        out = x @ x.T  # <- wrong operation!
        # Will return shape-(A, A) tensor, not (A, C)
        # (and we lie to the static type checker to try to get away with it)
        return cast(Tensor[T1, T3], out)

    x, y = parse(
        (tr.ones(3, 4), Tensor[A, B]),
        (tr.ones(4, 5), Tensor[B, C]),
    )
    assert_type(x, Tensor[A, B])
    assert_type(y, Tensor[B, C])

    # At runtime:
    # beartype raises and catches shape-mismatch of output.
    # Function should return shape-(A, C) but, at runtime, returns
    # shape-(A, A)
    z = buggy_matmul(x, y)  # beartype roars!

    assert_type(z, Tensor[A, C])


def check_readme4():
    from typing import Any

    import numpy as np
    import torch as tr

    from phantom_tensors import parse
    from phantom_tensors.alphabet import A, B
    from phantom_tensors.numpy import NDArray
    from phantom_tensors.torch import Tensor

    if sys.version_info < (3, 8):
        return

    t1, arr, t2 = parse(
        # <- Runtime: enter dimension-binding context
        (tr.rand(9, 2, 9), Tensor[B, A, B]),  # <-binds A=2 & B=9
        (np.ones((2,)), NDArray[A]),  # <- checks A==2
        (tr.rand(9), Tensor[B]),  # <- checks B==9
    )  # <- Runtime: exit dimension-binding scope
    #    Statically: casts t1, arr, t2 to shape-typed Tensors

    # static type checkers now see
    # t1: Tensor[B, A, B]
    # arr: NDArray[A]
    # t2: Tensor[B]

    w = parse(tr.rand(78), Tensor[A])
    # <- binds A=78 within this context

    assert_type(t1, Tensor[B, A, B])
    assert_type(arr, NDArray[A])
    assert_type(t2, Tensor[B])
    assert_type(w, Tensor[A])

    def vanilla_numpy(x: np.ndarray[Any, Any]): ...

    def vanilla_torch(x: tr.Tensor): ...

    vanilla_numpy(arr)  # type checker: OK
    vanilla_torch(t1)  # type checker: OK
    vanilla_torch(arr)  # type: ignore [type checker: Error!]


def check_phantom_example():
    from typing import Any

    import torch as tr

    from phantom import Phantom
    from phantom_tensors import parse
    from phantom_tensors.torch import Tensor

    class EvenOnly(int, Phantom[Any], predicate=lambda x: x % 2 == 0): ...

    assert_type(parse(tr.ones(1, 0), Tensor[int, EvenOnly]), Tensor[int, EvenOnly])
    assert_type(parse(tr.ones(1, 2), Tensor[int, EvenOnly]), Tensor[int, EvenOnly])
    assert_type(parse(tr.ones(1, 4), Tensor[int, EvenOnly]), Tensor[int, EvenOnly])

    parse(tr.ones(1, 3), Tensor[int, EvenOnly])  # runtime: ParseError


def check_beartype_example():
    from typing import Any

    import pytest
    from beartype import beartype
    from typing_extensions import assert_type

    from phantom_tensors import dim_binding_scope, parse
    from phantom_tensors.alphabet import A, B, C
    from phantom_tensors.torch import Tensor

    # @dim_binding_scope:
    #   ensures A, B, C consistent across all input/output tensor shapes
    #   within scope of function
    @dim_binding_scope
    @beartype  # <-- adds isinstance checks on inputs & outputs
    def matrix_multiply(x: Tensor[A, B], y: Tensor[B, C]) -> Tensor[A, C]:
        a, _ = x.shape
        _, c = y.shape
        return parse(tr.rand(a, c), Tensor[A, C])

    @beartype
    def needs_vector(x: Tensor[Any]): ...

    x, y = parse(
        (tr.rand(3, 4), Tensor[A, B]),
        (tr.rand(4, 5), Tensor[B, C]),
    )

    z = matrix_multiply(x, y)
    assert_type(z, Tensor[A, C])

    with pytest.raises(Exception):
        # beartype raises error: input Tensor[A, C] doesn't match Tensor[A]
        needs_vector(z)  # type: ignore

    with pytest.raises(Exception):
        # beartype raises error: inputs Tensor[A, B], Tensor[A, B] don't match signature
        matrix_multiply(x, x)  # type: ignore


# def check_parse_ints(x: int | Literal[1]):
#     assert_type(
#         parse_ints(x, A),
#         A,
#     )
#     assert_type(
#         parse_ints((x, A)),
#         A,
#     )
#     assert_type(
#         parse_ints(
#             (1, A),
#             (2, B),
#         ),
#         tuple[A, B],
#     )
#     assert_type(
#         parse_ints(
#             (1, A),
#             (2, B),
#             (2, B),
#         ),
#         tuple[A, B, B],
#     )
#     assert_type(
#         parse_ints(
#             (1, A),
#             (2, B),
#             (2, B),
#             (1, A),
#         ),
#         tuple[A, B, B, A],
#     )
#     assert_type(
#         parse_ints(
#             (1, A),
#             (2, B),
#             (2, B),
#             (1, A),
#             (2, B),
#         ),
#         tuple[A, B, B, A, B],
#     )
#     assert_type(
#         parse_ints((1, A), (2, B), (2, B), (1, A), (2, B), (1, A)),
#         tuple[A, B, B, A, B, A],
#     )


# def check_parse_tuples(x: int):

#     assert_type(
#         parse_tuples((x,), tuple[A]),
#         tuple[A],
#     )

#     assert_type(
#         parse_tuples(((x,), tuple[A])),
#         tuple[A],
#     )
#     assert_type(
#         parse_tuples(
#             ((x,), tuple[A]),
#             ((x, x), tuple[A, A]),
#         ),
#         tuple[
#             tuple[A],
#             tuple[A, A],
#         ],
#     )
#     assert_type(
#         parse_tuples(
#             ((x,), tuple[A]),
#             ((x, x), tuple[A, A]),
#             ((x, x), tuple[A, A, B]),
#         ),
#         tuple[
#             tuple[A],
#             tuple[A, A],
#             tuple[A, A, B],
#         ],
#     )
#     assert_type(
#         parse_tuples(
#             ((x,), tuple[A]),
#             ((x, x), tuple[A, A]),
#             ((x, x, x), tuple[A, A, B]),
#             ((x, x, x), tuple[A, A, B]),
#         ),
#         tuple[
#             tuple[A],
#             tuple[A, A],
#             tuple[A, A, B],
#             tuple[A, A, B],
#         ],
#     )
#     assert_type(
#         parse_tuples(
#             ((x,), tuple[A]),
#             ((x, x), tuple[A, A]),
#             ((x, x, x), tuple[A, A, B]),
#             ((x, x, x), tuple[A, A, B]),
#             ((x,), tuple[A]),
#         ),
#         tuple[tuple[A], tuple[A, A], tuple[A, A, B], tuple[A, A, B], tuple[A]],
#     )
#     assert_type(
#         parse_tuples(
#             ((x,), tuple[A]),
#             ((x, x), tuple[A, A]),
#             ((x, x, x), tuple[A, A, B]),
#             ((x, x, x), tuple[A, A, B]),
#             ((x,), tuple[A]),
#             ((x,), tuple[B]),
#         ),
#         tuple[
#             tuple[A], tuple[A, A], tuple[A, A, B], tuple[A, A, B], tuple[A], tuple[B]
#         ],
#     )


# def check_bad_int_parse(x: tr.Tensor, y: tuple[int, ...], z: int):
#     parse_ints(x, A)
#     parse_ints(y, A)
#     parse_ints((x, A))
#     parse_ints((y, A))

#     parse_ints(z, tuple[A])
#     parse_ints(z, Tensor[A])

# def check_bad_tuple_parse(x: tr.Tensor, y: tuple[int, ...], z: int):
#     parse_tuples(x, tuple[A])
#     parse_tuples(z, tuple[A])

#     parse_tuples((x, tuple[A]))
#     parse_tuples((z, tuple[A]))

#     parse_ints(y, A)
#     parse_ints(y, Tensor[A])
