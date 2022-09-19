from __future__ import annotations

from collections import defaultdict
from functools import wraps
from typing import Any, Callable, Optional, Tuple, Type, TypeVar, Union, cast

from typing_extensions import TypeAlias

import phantom_tensors._utils as _utils
from phantom_tensors._utils import LiteralLike, NewTypeLike, UnpackLike

ShapeDimType: TypeAlias = Union[
    Type[int],
    Type[UnpackLike],
    # the following all bind as dimension symbols by-reference
    Type[TypeVar],
    NewTypeLike,
    LiteralLike,
]

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


def setdefault_binding(symbol, value):
    # to do: speed up by short-cutting access to .bindings
    if DimBinder.bindings is None or symbol is int:
        return value
    else:
        return DimBinder.bindings.setdefault(symbol, value)


def check(shape_type: Tuple[ShapeDimType, ...], shape: Tuple[int, ...]) -> bool:
    # Don't need to check types / values of `shape` -- assumed
    # to be donwstream of `ndarray.shape`  call and thus already
    # validatated
    # Maybe enable extra strict mode where we do that checking

    # E.g. Tensor[A, B, B, C] -> matches == {A: [0], B: [1, 2], C: [3]}
    matches: defaultdict[ShapeDimType, list[int]] = defaultdict(list)

    # These can be cached globally -- are independent of match pattern
    # E.g. Tensor[Literal[1]] -> validators {Literal[1]: lambda x: x == 1}
    validators: dict[ShapeDimType, Callable[[Any], bool]] = {}

    var_field_ind: Optional[int] = None  # contains *Ts

    # TODO: add caching
    for n, dim_symbol in enumerate(shape_type):
        if dim_symbol is int:
            # E.g. Tensor[int, int]: no constraints on shape
            continue
        if _utils.is_typevar_unpack(dim_symbol):
            if var_field_ind is not None:
                assert False  # duplicate vartuple not allowed!
            var_field_ind = n
            continue

        # The following symbols bind to dimensions (by symbol-reference)
        # Some of them may also carry with them additional validation checks,
        # which need only be checked when the symbol is first bound

        if _utils.is_newtype(dim_symbol):
            _supertype = dim_symbol.__supertype__
            if _supertype is not int:
                if not issubclass(_supertype, int):
                    raise TypeError(
                        f"Dimensions expressed by NewTypes must be associated with an int or subclass of int. Got NewType of supertype {_supertype}"
                    )
                validators[dim_symbol] = lambda x, sp=_supertype: isinstance(x, sp)
            del _supertype
        elif isinstance(dim_symbol, TypeVar):
            pass
        elif _utils.is_literal(dim_symbol):
            _expected_literals = dim_symbol.__args__
            validators[dim_symbol] = lambda x, y=_expected_literals: any(
                x == val for val in y
            )
            del _expected_literals

        elif isinstance(dim_symbol, type) and issubclass(dim_symbol, int):
            validators[dim_symbol] = lambda x, type_=dim_symbol: isinstance(x, type_)
        else:
            raise TypeError(
                f"Got shape-type {shape_type} with dim {dim_symbol}. Valid dimensions are `type[int] | Unpack | TypeVar | NewType | Literal`"
            )

        matches[dim_symbol].append(n if var_field_ind is None else n - len(shape_type))

    if var_field_ind is None and len(shape_type) != len(shape):
        return False

    if var_field_ind is not None:
        if len(shape) < len(shape_type) - 1:
            return False

    _bindings = DimBinder.bindings

    for symbol, indices in matches.items():
        validation_fn = validators.get(symbol, None)

        if len(indices) == 1 and _bindings is None and validation_fn is None:
            continue

        first_index, *rest = indices
        actual_val = shape[first_index]
        if _bindings is None or symbol is int:
            expected_val = actual_val
        else:
            if symbol in _bindings:
                expected_val = _bindings[symbol]
                if actual_val != expected_val:
                    return False
            else:
                if validation_fn is not None and not validation_fn(actual_val):
                    return False
                _bindings[symbol] = actual_val
                expected_val = actual_val

        if not all(expected_val == shape[val_b] for val_b in rest):
            return False
    return True
