# pyright: strict

from ._internals import dim_binding_scope
from ._parse import parse

__version__ = "0.0.0"

__all__ = ["parse", "dim_binding_scope"]

from . import _version

__version__ = _version.get_versions()["version"]
