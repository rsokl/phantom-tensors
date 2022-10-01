# pyright: strict
from __future__ import annotations

from collections import defaultdict
from functools import wraps
from typing import Any, Callable, Iterable, Optional, Tuple, Type, TypeVar, Union, cast

from typing_extensions import Final, TypeAlias

import phantom_tensors._utils as _utils
from phantom_tensors._utils import LiteralLike, NewTypeLike, UnpackLike

ShapeDimType: TypeAlias = Union[
    Type[int],
    Type[UnpackLike],
    # Literal[Type[Any]]  -- can't actually express this
    # the following all bind as dimension symbols by-reference
    Type[TypeVar],
    NewTypeLike,
    LiteralLike,
]

LiteralCheck: TypeAlias = Callable[[Any, Iterable[Any]], bool]

F = TypeVar("F", bound=Callable[..., Any])


class DimBinder:
    bindings: Optional[dict[Any, int]] = None


class DimBindContext:
    _depth: int = 0

    def __enter__(self):
        self._depth += 1
        if self._depth == 1:
            DimBinder.bindings = {}

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any):
        self._depth -= 1
        if self._depth == 0:
            DimBinder.bindings = None

    def __call__(self, func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any):
            with self:
                return func(*args, **kwargs)

        return cast(F, wrapper)


dim_binding_scope = DimBindContext()


def check(shape_type: Tuple[ShapeDimType, ...], shape: Tuple[int, ...]) -> bool:
    # Don't need to check types / values of `shape` -- assumed
    # to be donwstream of `ndarray.shape`  call and thus already
    # validatated
    # Maybe enable extra strict mode where we do that checking

    # E.g. Tensor[A, B, B, C] -> matches == {A: [0], B: [1, 2], C: [3]}
    bound_symbols: defaultdict[ShapeDimType, list[int]] = defaultdict(list)
    validated_symbols: defaultdict[ShapeDimType, list[int]] = defaultdict(list)

    # These can be cached globally -- are independent of match pattern
    # E.g. Tensor[Literal[1]] -> validators {Literal[1]: lambda x: x == 1}
    validators: dict[ShapeDimType, Callable[[Any], bool]] = {}

    var_field_ind: Optional[int] = None  # contains *Ts

    # TODO: Add caching to validation process.
    #       - Should this use weakrefs?
    for n, dim_symbol in enumerate(shape_type):
        if dim_symbol is Any or dim_symbol is int:
            # E.g. Tensor[int, int] or Tensor[Any]
            # --> no constraints on shape
            continue
        if _utils.is_typevar_unpack(dim_symbol):
            if var_field_ind is not None:
                raise TypeError(
                    f"Type-shape {shape_type} specifies more than one TypeVarTuple"
                )
            var_field_ind = n
            continue

        # if variadic tuple is present, need to use negative indexing to reference
        # location from the end of the tuple
        CURRENT_INDEX: Final = n if var_field_ind is None else n - len(shape_type)

        # The following symbols bind to dimensions (by symbol-reference)
        # Some of them may also carry with them additional validation checks,
        # which need only be checked when the symbol is first bound
        _match_list = bound_symbols[dim_symbol]

        if _match_list:
            # We have already encountered symbol; do not need to validate or
            # extract validation function
            _match_list.append(CURRENT_INDEX)
            continue

        _validate_list = validated_symbols[dim_symbol]

        if _validate_list or dim_symbol in validators:
            del bound_symbols[dim_symbol]
            _validate_list.append(CURRENT_INDEX)
            continue

        if _utils.is_newtype(dim_symbol):
            _supertype = dim_symbol.__supertype__
            if _supertype is not int:
                if not issubclass(_supertype, int):
                    raise TypeError(
                        f"Dimensions expressed by NewTypes must be associated with an "
                        f"int or subclass of int. shape-type {shape_type} contains a "
                        f"NewType of supertype {_supertype}"
                    )
                validators[dim_symbol] = lambda x, sp=_supertype: isinstance(x, sp)  # type: ignore
            _match_list.append(CURRENT_INDEX)
            del _supertype
        elif isinstance(dim_symbol, TypeVar):
            _match_list.append(CURRENT_INDEX)
        elif _utils.is_literal(dim_symbol) and dim_symbol:
            _expected_literals = dim_symbol.__args__
            literal_check: LiteralCheck = lambda x, y=_expected_literals: any(
                x == val for val in y
            )
            validators[dim_symbol] = literal_check
            _validate_list.append(CURRENT_INDEX)
            del _expected_literals

        elif isinstance(dim_symbol, type) and issubclass(dim_symbol, int):
            validators[dim_symbol] = lambda x, type_=dim_symbol: isinstance(x, type_)  # type: ignore
            _validate_list.append(CURRENT_INDEX)
        else:
            raise TypeError(
                f"Got shape-type {shape_type} with dim {dim_symbol}. Valid dimensions "
                f"are `type[int] | Unpack | TypeVar | NewType | Literal`"
            )
        if not _match_list:
            del bound_symbols[dim_symbol]

        if not _validate_list:
            del validated_symbols[dim_symbol]

    if var_field_ind is None and len(shape_type) != len(shape):
        # E.g.      type: Tensor[A, B, C]
        #      vs. shape: (1, 1)  or (1, 1, 1, 1)  # must have exactly 3 dim
        return False

    if var_field_ind is not None and len(shape) < len(shape_type) - 1:
        # E.g.      type: Tensor[A, *Ts, C]
        #      vs. shape: (1,)  # should have at least 2 dim
        return False

    _bindings = DimBinder.bindings

    for symbol, indices in bound_symbols.items():
        validation_fn = validators.get(symbol, None)

        if len(indices) == 1 and _bindings is None and validation_fn is None:
            continue

        actual_val = shape[indices[0]]

        if _bindings is None or symbol is Any or symbol is int:
            expected_val = actual_val
        else:
            if symbol in _bindings:
                expected_val = _bindings[symbol]
            else:
                if validation_fn is not None and not validation_fn(actual_val):
                    return False
                _bindings[symbol] = actual_val
                expected_val = actual_val

        if not all(expected_val == shape[index] for index in indices):
            return False

    for symbol, indices in validated_symbols.items():
        validation_fn = validators[symbol]
        if not all(validation_fn(shape[index]) for index in indices):
            return False

    return True
