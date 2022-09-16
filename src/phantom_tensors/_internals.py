from __future__ import annotations

from collections import defaultdict
from functools import wraps
from typing import Any, Callable, Hashable, Optional, TypeVar, cast

from typing_extensions import Unpack

F = TypeVar("F", bound=Callable[..., Any])


class DimBinder:
    bindings: Optional[dict[Any, int]] = None


class DimBindContext:
    _depth: int = 0

    def __enter__(self):
        self._depth += 1
        if self._depth == 1:
            DimBinder.bindings = {}

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._depth -= 1
        if self._depth == 0:
            DimBinder.bindings = None

    def __call__(self, func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return cast(F, wrapper)


dim_binding_scope = DimBindContext()


def check(shape_type: tuple[Hashable, ...], shape: tuple[int, ...]) -> bool:

    # E.g. Tensor[A, B, B, C] :: matches == {A: [0], B: [1, 2], C: [3]}
    matches: defaultdict[Any, list[int]] = defaultdict(list)
    var_field_ind: Optional[int] = None  # contains *Ts

    for n, a in enumerate(shape_type):
        if a is int:
            # E.g. Tensor[int, int]: no constraints on shape
            continue
        elif getattr(a, "__origin__", None) is Unpack:
            if var_field_ind is not None:
                assert False  # duplicate vartuple not allowed!
            var_field_ind = n
        elif getattr(a, "__supertype__", None) is int or isinstance(a, TypeVar):
            # note TypeVarTuple is instance of TypeVar, need to check it first
            matches[a].append(n if var_field_ind is None else n - len(shape_type))
        else:
            assert False

    if var_field_ind is None and len(shape_type) != len(shape):
        return False

    if var_field_ind is not None:
        if len(shape) < len(shape_type) - 1:
            return False

    for symbol, indices in matches.items():
        if len(indices) == 1 and DimBinder.bindings is None:
            continue

        first, *rest = indices
        if DimBinder.bindings is None or symbol is int:
            a = shape[first]
        else:
            _a = shape[first]
            a = DimBinder.bindings.setdefault(symbol, _a)
            if a != _a:
                return False
        if not all(a == shape[b] for b in rest):
            return False
    return True
