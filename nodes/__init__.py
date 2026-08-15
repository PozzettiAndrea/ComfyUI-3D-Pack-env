"""Node package for ComfyUI-3D-Pack-env -- imported INSIDE the isolated env.

The presence of `comfy-env.toml` next to this file is the isolation switch:
comfy-env materialises one pixi env for this directory, runs a metadata scan
inside it, and forwards node execution to a persistent worker. Nothing here is
imported by the host ComfyUI process.

Everything the nodes import now lives under this directory: nodes.py, the
shared helpers (mesh_processer/, shared_utils/) and both vendored module trees
(Gen_3D_Modules/, MVs_Algorithms/). The pack root is NOT on sys.path, so
nothing sitting beside the repo can shadow a stdlib or site-packages module.

Three roots still have to be registered, and the reason is upstream's vendored
code, not this packaging: ~200 files inside the model families import each
other by BARE top-level name -- `from craftsman.models import ...`,
`from TRELLIS.trellis import ...`, `from mesh_processer.mesh import Mesh`.
Rewriting those into package-relative imports would touch hundreds of vendored
files and make every future upstream merge a conflict, so they are left alone
and the three roots they expect are added here instead:

    nodes/                   -> Gen_3D_Modules, MVs_Algorithms, mesh_processer,
                                shared_utils
    nodes/Gen_3D_Modules/    -> CRM, TRELLIS, craftsman, LGM, ... (21 names)
    nodes/MVs_Algorithms/    -> DiffRastMesh, FlexiCubes, GaussianSplatting, NeRF

This is contained: it happens inside the isolated environment's interpreter,
so it cannot affect the host ComfyUI process or any other pack.
"""

import inspect
import os
import sys

NODES_PATH = os.path.dirname(os.path.realpath(__file__))
MODULE_PATH = os.path.join(NODES_PATH, "Gen_3D_Modules")
MV_ALGO_PATH = os.path.join(NODES_PATH, "MVs_Algorithms")

for _p in (NODES_PATH, MODULE_PATH, MV_ALGO_PATH):
    if _p not in sys.path:
        sys.path.append(_p)

from . import nodes as _nodes_module

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

for _name, _cls in inspect.getmembers(_nodes_module, inspect.isclass):
    # Only classes DEFINED in nodes.py are nodes; imported ones are not.
    if _cls.__module__ != _nodes_module.__name__:
        continue
    _display = _name.replace("_", " ")
    NODE_CLASS_MAPPINGS[f"[Comfy3D] {_display}"] = _cls
    NODE_DISPLAY_NAME_MAPPINGS[f"[Comfy3D] {_display}"] = _display

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
