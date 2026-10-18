"""pixel3dmm

Restores the numpy scalar aliases that numpy >= 1.24 removed, because ``chumpy``
(imported while unpickling the FLAME ``.pkl`` model) and a few vendored
preprocessing modules still use them. Runs on ``import pixel3dmm``, before
anything touches chumpy; only missing attributes are added -- nothing is
overwritten.
"""

import numpy as _np

for _name, _py in (
    ("bool", bool),
    ("int", int),
    ("float", float),
    ("complex", complex),
    ("object", object),
    ("unicode", str),
    ("str", str),
    ("long", int),
    # numpy 2.0 also dropped the capitalised inf/nan spellings (no-ops under the
    # pinned numpy 1.26).
    ("Inf", float("inf")),
    ("Infinity", float("inf")),
    ("infty", float("inf")),
    ("PINF", float("inf")),
    ("NINF", float("-inf")),
    ("NaN", float("nan")),
    ("NAN", float("nan")),
):
    if not hasattr(_np, _name):
        setattr(_np, _name, _py)

del _np

# Python >= 3.11 removed inspect.getargspec, which chumpy 0.70 still calls at import time
# (chumpy is imported while unpickling the FLAME .pkl models).
import inspect as _inspect
if not hasattr(_inspect, "getargspec"):
    _inspect.getargspec = _inspect.getfullargspec
del _inspect

# scipy >= 1.14 removed scipy.integrate.simps (PIPNet's lib/functions.py imports it; only its AUC
# evaluation helper uses it). simpson is the same rule under its new name.
try:
    import scipy.integrate as _si
    if not hasattr(_si, "simps"):
        _si.simps = _si.simpson
    del _si
except ImportError:
    pass


def _register_nvdiffrast_jit_finder():
    """Python >= 3.12: let nvdiffrast find its JIT-built plugin.

    nvdiffrast 0.3.x builds its plugin with torch.utils.cpp_extension.load() and then calls
    importlib.import_module("nvdiffrast_plugin"). Up to Python 3.11, loading a single-phase-init
    extension also put it into sys.modules, so that import succeeded; on 3.12 it does not, and the
    import fails with "No module named 'nvdiffrast_plugin'". This finder resolves exactly those two
    names to the .so in torch's extension build directory (it is lazy: nothing is imported here).
    """
    import sys
    if sys.version_info < (3, 12):
        return
    import importlib.abc
    import importlib.util
    import os

    class _NvdiffrastPluginFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name not in ("nvdiffrast_plugin", "nvdiffrast_plugin_gl"):
                return None
            import torch.utils.cpp_extension as ce
            so = os.path.join(ce._get_build_directory(name, False), name + ".so")
            return importlib.util.spec_from_file_location(name, so) if os.path.exists(so) else None

    if not any(type(f).__name__ == "_NvdiffrastPluginFinder" for f in sys.meta_path):
        sys.meta_path.append(_NvdiffrastPluginFinder())


_register_nvdiffrast_jit_finder()
