from typing import Any

from typing_extensions import Literal, Protocol, TypeGuard, TypeVarTuple, Unpack


class NewTypeInt(Protocol):
    __name__: str
    __supertype__: type[int]

    def __call__(self, x: Any) -> int:
        ...


class UnpackLike(Protocol):
    _inst: Literal[True]
    _name: Literal[None]
    __origin__: type[Any] = Unpack
    __args__: tuple[TypeVarTuple]
    __parameters__: tuple[TypeVarTuple]
    __module__: str


class LiteralLike(Protocol):
    _inst: Literal[True]
    _name: Literal[None]
    __origin__: type[Any] = Literal
    __args__: tuple[Any, ...]
    __parameters__: tuple[()]
    __module__: str


class TupleGeneric(Protocol):
    __origin__: type[tuple]
    __args__: tuple[type[Any], ...]


def is_newtype_int(x: Any) -> TypeGuard[NewTypeInt]:
    supertype = getattr(x, "__supertype__", None)
    if supertype is None:
        return False
    return issubclass(supertype, int)


def is_typevar_unpack(x: Any) -> TypeGuard[UnpackLike]:
    return getattr(x, "__origin__", None) is Unpack


def is_tuple_generic(x: Any) -> TypeGuard[TupleGeneric]:
    return getattr(x, "__origin__", None) is tuple


def is_literal(x: Any) -> TypeGuard[LiteralLike]:
    return getattr(x, "__origin__", None) is Literal
