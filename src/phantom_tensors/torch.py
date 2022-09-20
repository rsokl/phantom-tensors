# pyright: strict
try:
    from torch import Tensor as _Tensor
except ImportError:
    raise ImportError(
        "You must install pytorch in order to user `phantom_tensors.torch`"
    )

from typing import TYPE_CHECKING, Generic, Tuple

import typing_extensions as _te

from phantom import Phantom as _Phantom, PhantomMeta as _PhantomMeta

from ._internals import check

__all__ = ["Tensor"]


Shape = _te.TypeVarTuple("Shape")


class _NewMeta(_PhantomMeta, type(_Tensor)):
    ...


class Tensor(Generic[_te.Unpack[Shape]], _Tensor):
    if not TYPE_CHECKING:
        _cache = {}

        @classmethod
        def __class_getitem__(cls, key):
            if not isinstance(key, tuple):
                key = (key,)

            # try:
            #     kk = tuple(k.__name__ for k in key)
            #     if kk in cls._cache:
            #         return cls._cache[kk]
            # except AttributeError:
            #     kk = None

            class PhantomTensor(
                _Tensor,
                _Phantom,
                metaclass=_NewMeta,
                predicate=lambda x: check(key, x.shape),
            ):
                _shape = key

            # if kk is not None:
            #     cls._cache[kk] = PhantomTensor
            return PhantomTensor

    @property
    def shape(self) -> Tuple[_te.Unpack[Shape]]:  # type: ignore
        ...
