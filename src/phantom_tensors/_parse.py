from __future__ import annotations

from typing import Any, TypeVar, cast, overload

from typing_extensions import Protocol, TypeAlias, TypeVarTuple, Unpack as U

import phantom_tensors._utils as _utils

from ._internals import DimBinder, check, dim_binding_scope
from .errors import ParseError

__all__ = ["parse"]

Ta = TypeVar("Ta", bound=tuple[Any, Any])
Tb = TypeVar("Tb")
Ts = TypeVarTuple("Ts")


class HasShape(Protocol[U[Ts]]):
    @property
    def shape(self) -> tuple[U[Ts]]:
        ...


TupleInt: TypeAlias = tuple[int, ...]

S1 = TypeVar("S1", bound=HasShape)
S2 = TypeVar("S2", bound=HasShape)
S3 = TypeVar("S3", bound=HasShape)
S4 = TypeVar("S4", bound=HasShape)
S5 = TypeVar("S5", bound=HasShape)
S6 = TypeVar("S6", bound=HasShape)

Ts1 = TypeVarTuple("Ts1")
Ts2 = TypeVarTuple("Ts2")
Ts3 = TypeVarTuple("Ts3")
Ts4 = TypeVarTuple("Ts4")
Ts5 = TypeVarTuple("Ts5")
Ts6 = TypeVarTuple("Ts6")

T1 = TypeVar("T1", bound=tuple)
T2 = TypeVar("T2", bound=tuple)
T3 = TypeVar("T3", bound=tuple)
T4 = TypeVar("T4", bound=tuple)
T5 = TypeVar("T5", bound=tuple)
T6 = TypeVar("T6", bound=tuple)


I1 = TypeVar("I1", bound=int)
I2 = TypeVar("I2", bound=int)
I3 = TypeVar("I3", bound=int)
I4 = TypeVar("I4", bound=int)
I5 = TypeVar("I5", bound=int)
I6 = TypeVar("I6", bound=int)


def _to_tuple(x: Ta | tuple[Ta, ...]) -> tuple[Ta, ...]:
    if len(x) == 2 and not isinstance(x[0], tuple):
        return (x,)  # type: ignore
    return x


@overload
def parse(
    __a: tuple[HasShape, type[S1]],
    __b: tuple[HasShape, type[S2]],
    __c: tuple[HasShape, type[S3]],
    __d: tuple[HasShape, type[S4]],
    __e: tuple[HasShape, type[S5]],
    __f: tuple[HasShape, type[S6]],
) -> tuple[S1, S2, S3, S4, S5, S6]:
    ...


@overload
def parse(
    __a: tuple[HasShape, type[S1]],
    __b: tuple[HasShape, type[S2]],
    __c: tuple[HasShape, type[S3]],
    __d: tuple[HasShape, type[S4]],
    __e: tuple[HasShape, type[S5]],
) -> tuple[S1, S2, S3, S4, S5]:
    ...


@overload
def parse(
    __a: tuple[HasShape, type[S1]],
    __b: tuple[HasShape, type[S2]],
    __c: tuple[HasShape, type[S3]],
    __d: tuple[HasShape, type[S4]],
) -> tuple[S1, S2, S3, S4]:
    ...


@overload
def parse(
    __a: tuple[HasShape, type[S1]],
    __b: tuple[HasShape, type[S2]],
    __c: tuple[HasShape, type[S3]],
) -> tuple[S1, S2, S3]:
    ...


@overload
def parse(
    __a: HasShape,
    __b: type[S1],
) -> S1:
    ...


@overload
def parse(
    __a: tuple[HasShape, type[S1]],
    __b: tuple[HasShape, type[S2]],
) -> tuple[S1, S2]:
    ...


@overload
def parse(__a: tuple[HasShape, type[S1]]) -> S1:
    ...


@overload
def parse(
    *tensor_type_pairs: tuple[HasShape, type[HasShape]] | HasShape | type[HasShape]
) -> HasShape | tuple[HasShape, ...]:
    ...


@dim_binding_scope
def parse(
    *tensor_type_pairs: tuple[HasShape, type[HasShape]] | HasShape | type[HasShape]
) -> HasShape | tuple[HasShape, ...]:
    out = []
    if len(tensor_type_pairs) == 0:
        raise ValueError("")
    if len(tensor_type_pairs) == 2 and not isinstance(tensor_type_pairs[0], tuple):
        tensor_type_pairs = (tensor_type_pairs,)  # type: ignore

    pairs = cast(
        tuple[tuple[HasShape, type[HasShape]], ...], _to_tuple(tensor_type_pairs)
    )

    del tensor_type_pairs

    for tensor, type_ in pairs:
        # Todo: remove phantom type logic
        if not isinstance(tensor, type_.__bound__):  # type: ignore
            if isinstance(type_.__bound__, tuple):
                tp, *_ = type_.__bound__
            else:
                tp = type_.__bound__
            raise ParseError(f"Expected {tp}, got: {type(tensor)}")

        type_shape = type_._shape  # type: ignore
        if not check(type_shape, tensor.shape):
            assert DimBinder.bindings is not None
            type_str = ", ".join(
                (
                    f"{getattr(p, '__name__', repr(p))}={DimBinder.bindings.get(p, '?')}"
                    if not _utils.is_typevar_unpack(p)
                    else "[...]"
                )
                for p in type_shape
            )
            if len(type_shape) == 1:
                # (A) -> (A,)
                type_str += ","
            raise ParseError(
                f"shape-{tuple(tensor.shape)} doesn't match shape-type ({type_str})"
            )
        out.append(tensor)
    if len(out) == 1:
        return out[0]
    return tuple(out)


# @overload
# def parse_ints(
#     __a: tuple[int, type[I1]],
#     __b: tuple[int, type[I2]],
#     __c: tuple[int, type[I3]],
#     __d: tuple[int, type[I4]],
#     __e: tuple[int, type[I5]],
#     __f: tuple[int, type[I6]],
# ) -> tuple[I1, I2, I3, I4, I5, I6]:
#     ...


# @overload
# def parse_ints(
#     __a: tuple[int, type[I1]],
#     __b: tuple[int, type[I2]],
#     __c: tuple[int, type[I3]],
#     __d: tuple[int, type[I4]],
#     __e: tuple[int, type[I5]],
# ) -> tuple[I1, I2, I3, I4, I5]:
#     ...


# @overload
# def parse_ints(
#     __a: tuple[int, type[I1]],
#     __b: tuple[int, type[I2]],
#     __c: tuple[int, type[I3]],
#     __d: tuple[int, type[I4]],
# ) -> tuple[I1, I2, I3, I4]:
#     ...


# @overload
# def parse_ints(
#     __a: tuple[int, type[I1]],
#     __b: tuple[int, type[I2]],
#     __c: tuple[int, type[I3]],
# ) -> tuple[I1, I2, I3]:
#     ...


# @overload
# def parse_ints(
#     __a: int,
#     __b: type[I1],
# ) -> I1:
#     ...


# @overload
# def parse_ints(
#     __a: tuple[int, type[I1]],
#     __b: tuple[int, type[I2]],
# ) -> tuple[I1, I2]:
#     ...


# @overload
# def parse_ints(__a: tuple[int, type[I1]]) -> I1:
#     ...


# @overload
# def parse_ints(
#     *tensor_type_pairs: tuple[int, type[int]] | int | type[int]
# ) -> int | tuple[int, ...]:
#     ...


# @dim_binding_scope
# def parse_ints(
#     *tensor_type_pairs: tuple[int, type[int]] | int | type[int]
# ) -> int | tuple[int, ...]:
#     ...

# @overload
# def parse_tuples(
#     __a: tuple[TupleInt, type[tuple[U[Ts1]]]],
#     __b: tuple[TupleInt, type[tuple[U[Ts2]]]],
#     __c: tuple[TupleInt, type[tuple[U[Ts3]]]],
#     __d: tuple[TupleInt, type[tuple[U[Ts4]]]],
#     __e: tuple[TupleInt, type[tuple[U[Ts5]]]],
#     __f: tuple[TupleInt, type[tuple[U[Ts6]]]],
# ) -> tuple[
#     tuple[U[Ts1]],
#     tuple[U[Ts2]],
#     tuple[U[Ts3]],
#     tuple[U[Ts4]],
#     tuple[U[Ts5]],
#     tuple[U[Ts6]],
# ]:
#     ...


# @overload
# def parse_tuples(
#     __a: tuple[TupleInt, type[tuple[U[Ts1]]]],
#     __b: tuple[TupleInt, type[tuple[U[Ts2]]]],
#     __c: tuple[TupleInt, type[tuple[U[Ts3]]]],
#     __d: tuple[TupleInt, type[tuple[U[Ts4]]]],
#     __e: tuple[TupleInt, type[tuple[U[Ts5]]]],
# ) -> tuple[tuple[U[Ts1]], tuple[U[Ts2]], tuple[U[Ts3]], tuple[U[Ts4]], tuple[U[Ts5]]]:
#     ...


# @overload
# def parse_tuples(
#     __a: tuple[TupleInt, type[tuple[U[Ts1]]]],
#     __b: tuple[TupleInt, type[tuple[U[Ts2]]]],
#     __c: tuple[TupleInt, type[tuple[U[Ts3]]]],
#     __d: tuple[TupleInt, type[tuple[U[Ts4]]]],
# ) -> tuple[tuple[U[Ts1]], tuple[U[Ts2]], tuple[U[Ts3]], tuple[U[Ts4]]]:
#     ...


# @overload
# def parse_tuples(
#     __a: tuple[TupleInt, type[tuple[U[Ts1]]]],
#     __b: tuple[TupleInt, type[tuple[U[Ts2]]]],
#     __c: tuple[TupleInt, type[tuple[U[Ts3]]]],
# ) -> tuple[tuple[U[Ts1]], tuple[U[Ts2]], tuple[U[Ts3]]]:
#     ...


# @overload
# def parse_tuples(
#     __a: TupleInt,
#     __b: T1,
# ) -> T1:
#     ...


# @overload
# def parse_tuples(
#     __a: TupleInt,
#     __b: type[tuple[U[Ts1]]],
# ) -> tuple[U[Ts1]]:
#     ...


# @overload
# def parse_tuples(
#     __a: tuple[TupleInt, type[tuple[U[Ts1]]]],
#     __b: tuple[TupleInt, type[tuple[U[Ts2]]]],
# ) -> tuple[tuple[U[Ts1]], tuple[U[Ts2]]]:
#     ...


# @overload
# def parse_tuples(__a: tuple[TupleInt, type[tuple[U[Ts1]]]]) -> tuple[U[Ts1]]:
#     ...


# @overload
# def parse_tuples(
#     *tensor_type_pairs: tuple[TupleInt, type[tuple[Any, ...]]]
#     | TupleInt
#     | type[tuple[Any, ...]]
# ) -> TupleInt | tuple[Any, ...]:
#     ...


# @dim_binding_scope
# def parse_tuples(
#     *tensor_type_pairs: tuple[TupleInt, type[tuple[Any, ...]]]
#     | TupleInt
#     | type[tuple[Any, ...]]
# ) -> TupleInt | tuple[Any, ...]:
#     ...
