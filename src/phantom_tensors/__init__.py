# pyright: strict

from typing import TYPE_CHECKING

from ._internals import dim_binding_scope
from ._parse import parse

__all__ = ["parse", "dim_binding_scope"]


if not TYPE_CHECKING:
    try:
        from ._version import version as __version__
    except ImportError:
        __version__ = "unknown version"
else:  # pragma: no cover
    __version__: str
