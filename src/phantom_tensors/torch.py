# pyright: strict
try:
    from torch import Tensor as _Tensor
except ImportError:
    raise ImportError(
        "You must install pytorch in order to user `phantom_tensors.torch`"
    )

from typing import TYPE_CHECKING, Generic, Tuple

import typing_extensions as _te

from ._internals import check
from ._utils import CustomInstanceCheck

__all__ = ["Tensor"]


Shape = _te.TypeVarTuple("Shape")


class _NewMeta(CustomInstanceCheck, type(_Tensor)):
    ...


class Tensor(Generic[_te.Unpack[Shape]], _Tensor):
    if not TYPE_CHECKING:
        # TODO: add caching

        @classmethod
        def __class_getitem__(cls, key):
            if not isinstance(key, tuple):
                key = (key,)

            class PhantomTensor(
                _Tensor,
                metaclass=_NewMeta,
            ):
                __origin__ = _Tensor
                __args__ = key

                @classmethod
                def __instancecheck__(cls, __instance: object) -> bool:
                    if not isinstance(__instance, _Tensor):
                        return False
                    return check(key, __instance.shape)

            return PhantomTensor

    @property
    def shape(self) -> Tuple[_te.Unpack[Shape]]:  # type: ignore
        ...
