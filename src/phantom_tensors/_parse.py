# pyright: strict
from __future__ import annotations

from typing import Any, List, Tuple, Type, TypeVar, Union, cast, overload

from typing_extensions import ClassVar, Protocol, TypeAlias, TypeVarTuple

import phantom_tensors._utils as _utils

from ._internals import DimBinder, ShapeDimType, check, dim_binding_scope
from .errors import ParseError

__all__ = ["parse"]

Ta = TypeVar("Ta", bound=Tuple[Any, Any])
Tb = TypeVar("Tb")
Ts = TypeVarTuple("Ts")


class _Generic(Protocol):
    __origin__: Type[Any]
    __args__: Tuple[ShapeDimType, ...]


class _Phantom(Protocol):
    __bound__: ClassVar[Union[Type[Any], Tuple[Type[Any], ...]]]
    __args__: Tuple[ShapeDimType, ...]


class HasShape(Protocol):
    @property
    def shape(self) -> Any:
        ...


TupleInt: TypeAlias = Tuple[int, ...]

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

T1 = TypeVar("T1", bound=Tuple[Any, ...])
T2 = TypeVar("T2", bound=Tuple[Any, ...])
T3 = TypeVar("T3", bound=Tuple[Any, ...])
T4 = TypeVar("T4", bound=Tuple[Any, ...])
T5 = TypeVar("T5", bound=Tuple[Any, ...])
T6 = TypeVar("T6", bound=Tuple[Any, ...])


I1 = TypeVar("I1", bound=int)
I2 = TypeVar("I2", bound=int)
I3 = TypeVar("I3", bound=int)
I4 = TypeVar("I4", bound=int)
I5 = TypeVar("I5", bound=int)
I6 = TypeVar("I6", bound=int)


def _to_tuple(x: Ta | Tuple[Ta, ...]) -> Tuple[Ta, ...]:
    if len(x) == 2 and not isinstance(x[0], tuple):
        return (x,)  # type: ignore
    return x


@overload
def parse(
    __a: Tuple[HasShape, Type[S1]],
    __b: Tuple[HasShape, Type[S2]],
    __c: Tuple[HasShape, Type[S3]],
    __d: Tuple[HasShape, Type[S4]],
    __e: Tuple[HasShape, Type[S5]],
    __f: Tuple[HasShape, Type[S6]],
) -> Tuple[S1, S2, S3, S4, S5, S6]:
    ...


@overload
def parse(
    __a: Tuple[HasShape, Type[S1]],
    __b: Tuple[HasShape, Type[S2]],
    __c: Tuple[HasShape, Type[S3]],
    __d: Tuple[HasShape, Type[S4]],
    __e: Tuple[HasShape, Type[S5]],
) -> Tuple[S1, S2, S3, S4, S5]:
    ...


@overload
def parse(
    __a: Tuple[HasShape, Type[S1]],
    __b: Tuple[HasShape, Type[S2]],
    __c: Tuple[HasShape, Type[S3]],
    __d: Tuple[HasShape, Type[S4]],
) -> Tuple[S1, S2, S3, S4]:
    ...


@overload
def parse(
    __a: Tuple[HasShape, Type[S1]],
    __b: Tuple[HasShape, Type[S2]],
    __c: Tuple[HasShape, Type[S3]],
) -> Tuple[S1, S2, S3]:
    ...


@overload
def parse(
    __a: HasShape,
    __b: Type[S1],
) -> S1:
    ...


@overload
def parse(
    __a: Tuple[HasShape, Type[S1]],
    __b: Tuple[HasShape, Type[S2]],
) -> Tuple[S1, S2]:
    ...


@overload
def parse(__a: Tuple[HasShape, Type[S1]]) -> S1:
    ...


@overload
def parse(
    *tensor_type_pairs: Tuple[HasShape, Type[HasShape]] | HasShape | Type[HasShape]
) -> HasShape | Tuple[HasShape, ...]:
    ...


@dim_binding_scope
def parse(
    *tensor_type_pairs: Tuple[HasShape, Type[HasShape]] | HasShape | Type[HasShape]
) -> HasShape | Tuple[HasShape, ...]:
    if len(tensor_type_pairs) == 0:
        raise ValueError("")
    if len(tensor_type_pairs) == 2 and not isinstance(tensor_type_pairs[0], tuple):
        tensor_type_pairs = (tensor_type_pairs,)  # type: ignore

    pairs = cast(
        Tuple[Tuple[HasShape, Type[HasShape]], ...], _to_tuple(tensor_type_pairs)
    )

    out: List[HasShape] = []

    del tensor_type_pairs

    for tensor, type_ in pairs:
        if hasattr(type_, "__origin__"):
            type_ = cast(_Generic, type_)
            type_shape = type_.__args__
            if not isinstance(tensor, type_.__origin__):
                raise ParseError(f"Expected {type_.__origin__}, got: {type(tensor)}")

        elif hasattr(type_, "__bound__"):
            # Todo: remove phantom type logic
            type_ = cast(_Phantom, type_)

            if not isinstance(tensor, type_.__bound__):
                if isinstance(type_.__bound__, tuple):
                    tp, *_ = type_.__bound__
                else:
                    tp = type_.__bound__
                raise ParseError(f"Expected {tp}, got: {type(tensor)}")

            type_shape = type_.__args__
        else:
            assert False

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
#     __a: Tuple[int, Type[I1]],
#     __b: Tuple[int, Type[I2]],
#     __c: Tuple[int, Type[I3]],
#     __d: Tuple[int, Type[I4]],
#     __e: Tuple[int, Type[I5]],
#     __f: Tuple[int, Type[I6]],
# ) -> Tuple[I1, I2, I3, I4, I5, I6]:
#     ...


# @overload
# def parse_ints(
#     __a: Tuple[int, Type[I1]],
#     __b: Tuple[int, Type[I2]],
#     __c: Tuple[int, Type[I3]],
#     __d: Tuple[int, Type[I4]],
#     __e: Tuple[int, Type[I5]],
# ) -> Tuple[I1, I2, I3, I4, I5]:
#     ...


# @overload
# def parse_ints(
#     __a: Tuple[int, Type[I1]],
#     __b: Tuple[int, Type[I2]],
#     __c: Tuple[int, Type[I3]],
#     __d: Tuple[int, Type[I4]],
# ) -> Tuple[I1, I2, I3, I4]:
#     ...


# @overload
# def parse_ints(
#     __a: Tuple[int, Type[I1]],
#     __b: Tuple[int, Type[I2]],
#     __c: Tuple[int, Type[I3]],
# ) -> Tuple[I1, I2, I3]:
#     ...


# @overload
# def parse_ints(
#     __a: int,
#     __b: Type[I1],
# ) -> I1:
#     ...


# @overload
# def parse_ints(
#     __a: Tuple[int, Type[I1]],
#     __b: Tuple[int, Type[I2]],
# ) -> Tuple[I1, I2]:
#     ...


# @overload
# def parse_ints(__a: Tuple[int, Type[I1]]) -> I1:
#     ...


# @overload
# def parse_ints(
#     *tensor_type_pairs: Tuple[int, Type[int]] | int | Type[int]
# ) -> int | Tuple[int, ...]:
#     ...


# @dim_binding_scope
# def parse_ints(
#     *tensor_type_pairs: Tuple[int, Type[int]] | int | Type[int]
# ) -> int | Tuple[int, ...]:
#     ...

# @overload
# def parse_tuples(
#     __a: Tuple[TupleInt, Type[Tuple[U[Ts1]]]],
#     __b: Tuple[TupleInt, Type[Tuple[U[Ts2]]]],
#     __c: Tuple[TupleInt, Type[Tuple[U[Ts3]]]],
#     __d: Tuple[TupleInt, Type[Tuple[U[Ts4]]]],
#     __e: Tuple[TupleInt, Type[Tuple[U[Ts5]]]],
#     __f: Tuple[TupleInt, Type[Tuple[U[Ts6]]]],
# ) -> Tuple[
#     Tuple[U[Ts1]],
#     Tuple[U[Ts2]],
#     Tuple[U[Ts3]],
#     Tuple[U[Ts4]],
#     Tuple[U[Ts5]],
#     Tuple[U[Ts6]],
# ]:
#     ...


# @overload
# def parse_tuples(
#     __a: Tuple[TupleInt, Type[Tuple[U[Ts1]]]],
#     __b: Tuple[TupleInt, Type[Tuple[U[Ts2]]]],
#     __c: Tuple[TupleInt, Type[Tuple[U[Ts3]]]],
#     __d: Tuple[TupleInt, Type[Tuple[U[Ts4]]]],
#     __e: Tuple[TupleInt, Type[Tuple[U[Ts5]]]],
# ) -> Tuple[Tuple[U[Ts1]], Tuple[U[Ts2]], Tuple[U[Ts3]], Tuple[U[Ts4]], Tuple[U[Ts5]]]:
#     ...


# @overload
# def parse_tuples(
#     __a: Tuple[TupleInt, Type[Tuple[U[Ts1]]]],
#     __b: Tuple[TupleInt, Type[Tuple[U[Ts2]]]],
#     __c: Tuple[TupleInt, Type[Tuple[U[Ts3]]]],
#     __d: Tuple[TupleInt, Type[Tuple[U[Ts4]]]],
# ) -> Tuple[Tuple[U[Ts1]], Tuple[U[Ts2]], Tuple[U[Ts3]], Tuple[U[Ts4]]]:
#     ...


# @overload
# def parse_tuples(
#     __a: Tuple[TupleInt, Type[Tuple[U[Ts1]]]],
#     __b: Tuple[TupleInt, Type[Tuple[U[Ts2]]]],
#     __c: Tuple[TupleInt, Type[Tuple[U[Ts3]]]],
# ) -> Tuple[Tuple[U[Ts1]], Tuple[U[Ts2]], Tuple[U[Ts3]]]:
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
#     __b: Type[Tuple[U[Ts1]]],
# ) -> Tuple[U[Ts1]]:
#     ...


# @overload
# def parse_tuples(
#     __a: Tuple[TupleInt, Type[Tuple[U[Ts1]]]],
#     __b: Tuple[TupleInt, Type[Tuple[U[Ts2]]]],
# ) -> Tuple[Tuple[U[Ts1]], Tuple[U[Ts2]]]:
#     ...


# @overload
# def parse_tuples(__a: Tuple[TupleInt, Type[Tuple[U[Ts1]]]]) -> Tuple[U[Ts1]]:
#     ...


# @overload
# def parse_tuples(
#     *tensor_type_pairs: Tuple[TupleInt, Type[Tuple[Any, ...]]]
#     | TupleInt
#     | Type[Tuple[Any, ...]]
# ) -> TupleInt | Tuple[Any, ...]:
#     ...


# @dim_binding_scope
# def parse_tuples(
#     *tensor_type_pairs: Tuple[TupleInt, Type[Tuple[Any, ...]]]
#     | TupleInt
#     | Type[Tuple[Any, ...]]
# ) -> TupleInt | Tuple[Any, ...]:
#     ...
