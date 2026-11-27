"""
Backwards-compatibility shim aliasing ``src.*`` imports to ``ghost.*``.

This directory exists only in the source repository so existing commands
(``python -m src.server``), Docker images, examples, and tests keep working.
It is excluded from packaged distributions built from ``pyproject.toml`` so
installing ``ghost`` never collides with a host project's own ``src`` package.
"""
from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import sys


class _SrcAliasLoader(importlib.abc.Loader):
    """Loader that aliases a ``src.*`` module directly to its ``ghost.*`` module object."""

    def __init__(self, target_name: str, target_spec: importlib.machinery.ModuleSpec):
        self.target_name = target_name
        self.target_spec = target_spec

    def create_module(self, spec: importlib.machinery.ModuleSpec):
        module = importlib.import_module(self.target_name)
        sys.modules[spec.name] = module
        return module

    def exec_module(self, module) -> None:
        # Target module was already executed by importlib.import_module.
        return None

    def get_code(self, fullname: str):
        loader = self.target_spec.loader
        if loader is not None and hasattr(loader, "get_code"):
            return loader.get_code(self.target_name)
        return None

    def get_filename(self, fullname: str):
        loader = self.target_spec.loader
        if loader is not None and hasattr(loader, "get_filename"):
            return loader.get_filename(self.target_name)
        return self.target_spec.origin


class _SrcAliasFinder(importlib.abc.MetaPathFinder):
    """Redirect ``src.<subpath>`` lookups to ``ghost.<subpath>``."""

    def find_spec(self, fullname: str, path=None, target=None):
        if not fullname.startswith("src."):
            return None
        target_name = "ghost." + fullname[4:]
        target_spec = importlib.util.find_spec(target_name)
        if target_spec is None:
            return None
        is_pkg = target_spec.submodule_search_locations is not None
        spec = importlib.machinery.ModuleSpec(
            name=fullname,
            loader=_SrcAliasLoader(target_name, target_spec),
            origin=target_spec.origin,
            is_package=is_pkg,
        )
        if is_pkg:
            spec.submodule_search_locations = list(target_spec.submodule_search_locations)
        return spec


if not any(isinstance(f, _SrcAliasFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, _SrcAliasFinder())
