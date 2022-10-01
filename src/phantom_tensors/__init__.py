# pyright: strict

from . import _version
from ._internals import dim_binding_scope
from ._parse import parse

__all__ = ["parse", "dim_binding_scope"]


__version__: str = _version.get_versions()["version"]
