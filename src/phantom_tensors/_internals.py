from __future__ import annotations
from collections import defaultdict

from typing import (
    Any,
    cast,
    Hashable,
    Optional,
    Callable,
    TypeVar,
)

from functools import wraps

F = TypeVar('F', bound=Callable[..., Any])

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

    if len(shape_type) != len(shape):  # TODO: permit arbitrary len-shape
        return False

    # E.g. Tensor[A, B, B, C] :: matches == {A: [0], B: [1, 2], C: [3]}
    matches: defaultdict[Any, list[int]] = defaultdict(list)
    for n, a in enumerate(shape_type):
        matches[a].append(n)

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
        
        