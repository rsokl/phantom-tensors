from __future__ import annotations

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
#     parse_ints(x, A)  # type: ignore
#     parse_ints(y, A)  # type: ignore
#     parse_ints((x, A))  # type: ignore
#     parse_ints((y, A))  # type: ignore

#     parse_ints(z, tuple[A])  # type: ignore
#     parse_ints(z, Tensor[A])  # type: ignore

# def check_bad_tuple_parse(x: tr.Tensor, y: tuple[int, ...], z: int):
#     parse_tuples(x, tuple[A])  # type: ignore
#     parse_tuples(z, tuple[A])  # type: ignore

#     parse_tuples((x, tuple[A]))  # type: ignore
#     parse_tuples((z, tuple[A]))  # type: ignore

#     parse_ints(y, A)  # type: ignore
#     parse_ints(y, Tensor[A])  # type: ignore
