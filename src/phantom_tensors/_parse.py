from __future__ import annotations

from typing import Type, TypeVar, overload

from typing_extensions import Protocol

from ._internals import DimBinder, check, dim_binding_scope
from .errors import ParseError

class HasShape(Protocol):
    @property
    def shape(self) -> tuple[int, ...]:
        ...

T1 = TypeVar("T1", bound=HasShape)
T2 = TypeVar("T2", bound=HasShape)
T3 = TypeVar("T3", bound=HasShape)
T4 = TypeVar("T4", bound=HasShape)
T5 = TypeVar("T5", bound=HasShape)
T6 = TypeVar("T6", bound=HasShape)


@overload
def parse(
    __a: tuple[HasShape, Type[T1]],
    __b: tuple[HasShape, Type[T2]],
    __c: tuple[HasShape, Type[T3]],
    __d: tuple[HasShape, Type[T4]],
    __e: tuple[HasShape, Type[T5]],
    __f: tuple[HasShape, Type[T6]],
) -> tuple[T1, T2, T3, T4, T5, T6]:
    ...

@overload
def parse(
    __a: tuple[HasShape, Type[T1]],
    __b: tuple[HasShape, Type[T2]],
    __c: tuple[HasShape, Type[T3]],
    __d: tuple[HasShape, Type[T4]],
    __e: tuple[HasShape, Type[T5]],
) -> tuple[T1, T2, T3, T4, T5]:
    ...

@overload
def parse(
    __a: tuple[HasShape, Type[T1]],
    __b: tuple[HasShape, Type[T2]],
    __c: tuple[HasShape, Type[T3]],
    __d: tuple[HasShape, Type[T4]],
) -> tuple[T1, T2, T3, T4]:
    ...


@overload
def parse(
    __a: tuple[HasShape, Type[T1]],
    __b: tuple[HasShape, Type[T2]],
    __c: tuple[HasShape, Type[T3]],
) -> tuple[T1, T2, T3]:
    ...


@overload
def parse(
    __a: HasShape,
    __b: Type[T1],
) -> T1:
    ...


@overload
def parse(
    __a: tuple[HasShape, Type[T1]],
    __b: tuple[HasShape, Type[T2]],
) -> tuple[T1, T2]:
    ...


@overload
def parse(__a: tuple[HasShape, Type[T1]]) -> T1:
    ...


@overload
def parse(
    *tensor_type_pairs: tuple[HasShape, Type[HasShape]] | HasShape | Type[HasShape]
) -> HasShape | tuple[HasShape, ...]:
    ...


@dim_binding_scope
def parse(
    *tensor_type_pairs: tuple[HasShape, Type[HasShape]] | HasShape | Type[HasShape]
) -> HasShape | tuple[HasShape, ...]:
    out = []
    if len(tensor_type_pairs) == 0:
        raise ValueError("")
    if len(tensor_type_pairs) == 2 and not isinstance(tensor_type_pairs[0], tuple):
        tensor_type_pairs = (tensor_type_pairs,)  # type: ignore

    for tensor, type_ in tensor_type_pairs:  # type: ignore
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
                f"{p.__name__}={DimBinder.bindings.get(p, '?')}" for p in type_shape
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
