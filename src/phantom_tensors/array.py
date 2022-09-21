# pyright: strict
from typing import Any, Tuple

import typing_extensions as _te
from typing_extensions import Protocol, runtime_checkable

__all__ = ["SupportsArray"]


Shape = _te.TypeVarTuple("Shape")


@runtime_checkable
class SupportsArray(Protocol[_te.Unpack[Shape]]):
    def __array__(self) -> Any:
        ...

    @property
    def shape(self) -> Tuple[_te.Unpack[Shape]]:
        ...
