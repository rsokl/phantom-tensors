# pyright: strict

import abc
from typing import Any, Tuple, Type

from typing_extensions import Literal, Protocol, TypeGuard, TypeVarTuple, Unpack

_Ts = TypeVarTuple("_Ts")

UnpackType = type(Unpack[_Ts])  # type: ignore
LiteralType = type(Literal[1])


class CustomInstanceCheck(abc.ABCMeta):
    def __instancecheck__(self, instance: object) -> bool:
        return self.__instancecheck__(instance)


class NewTypeLike(Protocol):
    __name__: str
    __supertype__: Type[Any]

    def __call__(self, x: Any) -> int:
        ...


class NewTypeInt(Protocol):
    __name__: str
    __supertype__: Type[int]

    def __call__(self, x: Any) -> int:
        ...


class UnpackLike(Protocol):
    _inst: Literal[True]
    _name: Literal[None]
    __origin__: Type[Any] = Unpack
    __args__: Tuple[TypeVarTuple]
    __parameters__: Tuple[TypeVarTuple]
    __module__: str


class LiteralLike(Protocol):
    _inst: Literal[True]
    _name: Literal[None]
    __origin__: Type[Any] = Literal
    __args__: Tuple[Any, ...]
    __parameters__: Tuple[()]
    __module__: str


class TupleGeneric(Protocol):
    __origin__: Type[Tuple[Any, ...]]
    __args__: Tuple[Type[Any], ...]


def is_newtype(x: Any) -> TypeGuard[NewTypeLike]:
    return hasattr(x, "__supertype__")


def is_newtype_int(x: Any) -> TypeGuard[NewTypeInt]:
    supertype = getattr(x, "__supertype__", None)
    if supertype is None:
        return False
    return issubclass(supertype, int)


def is_typevar_unpack(x: Any) -> TypeGuard[UnpackLike]:
    return isinstance(x, UnpackType)


def is_tuple_generic(x: Any) -> TypeGuard[TupleGeneric]:
    return getattr(x, "__origin__", None) is tuple


def is_literal(x: Any) -> TypeGuard[LiteralLike]:
    return isinstance(x, LiteralType)
