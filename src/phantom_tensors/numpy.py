try:
    from numpy import ndarray as _ndarray
    from numpy.typing import NDArray as _NDArray
except ImportError:
    raise ImportError("You must install numpy in order to user `phantom_tensors.numpy`")

from typing import TYPE_CHECKING, Any, Generic, TypeVar

import typing_extensions as _te

from phantom import Phantom as _Phantom

from ._internals import check

__all__ = ["NDArray"]


Shape = _te.TypeVarTuple("Shape")


class NDArray(Generic[_te.Unpack[Shape]], _NDArray[Any]):

    if not TYPE_CHECKING:
        _cache = {}

        @classmethod
        def __class_getitem__(cls, key):
            if not isinstance(key, tuple):
                key = (key,)
            try:
                kk = tuple(k.__name__ for k in key)
                if kk in cls._cache:
                    return cls._cache[kk]
            except AttributeError:
                kk = None

            class PhantomTensor(
                _ndarray,
                _Phantom,
                predicate=lambda x: check(key, x.shape),
            ):
                _shape = key

            if kk is not None:
                cls._cache[kk] = PhantomTensor

            return PhantomTensor

    @property
    def shape(self) -> tuple[_te.Unpack[Shape]]:
        ...
