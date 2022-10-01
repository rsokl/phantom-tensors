# pyright: strict
try:
    from numpy import ndarray as _ndarray
    from numpy.typing import NDArray as _NDArray
except ImportError:
    raise ImportError("You must install numpy in order to user `phantom_tensors.numpy`")

from typing import TYPE_CHECKING, Any, Generic, Sequence, SupportsIndex, Tuple, Union

import typing_extensions as _te

from ._internals import check
from ._utils import CustomInstanceCheck

__all__ = ["NDArray"]


Shape = _te.TypeVarTuple("Shape")


class NDArray(Generic[_te.Unpack[Shape]], _NDArray[Any]):

    if not TYPE_CHECKING:
        _cache = {}

        @classmethod
        def __class_getitem__(cls, key):
            if not isinstance(key, tuple):
                key = (key,)

            class PhantomNDArray(
                _ndarray,
                metaclass=CustomInstanceCheck,
            ):
                __origin__ = _ndarray
                # TODO: conform with ndarray[shape, dtype]
                __args__ = key

                @classmethod
                def __instancecheck__(cls, __instance: object) -> bool:
                    if not isinstance(__instance, _ndarray):
                        return False
                    return check(key, __instance.shape)

            return PhantomNDArray

    @property
    def shape(self) -> Tuple[_te.Unpack[Shape]]:  # type: ignore
        ...

    @shape.setter
    def shape(self, value: Union[SupportsIndex, Sequence[SupportsIndex]]) -> None:
        ...
