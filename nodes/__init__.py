"""Node package for ComfyUI-3D-Pack-env -- imported INSIDE the isolated env.

The presence of `comfy-env.toml` next to this file is the isolation switch:
comfy-env materialises one pixi env for this directory, runs a metadata scan
inside it, and forwards node execution to a persistent worker. Nothing here is
imported by the host ComfyUI process.

Upstream keeps the model families as top-level packages reachable via sys.path
(`from TRELLIS...`, `from CRM...`) and also as `Gen_3D_Modules.X`; the shared
helpers are imported absolutely (`from mesh_processer.mesh import Mesh`) from
inside those families. Both styles are preserved by putting the pack root, the
two module roots, and this directory on sys.path before importing nodes.py.
"""

import inspect
import os
import sys

NODES_PATH = os.path.dirname(os.path.realpath(__file__))
ROOT_PATH = os.path.dirname(NODES_PATH)
MODULE_PATH = os.path.join(ROOT_PATH, "Gen_3D_Modules")
MV_ALGO_PATH = os.path.join(ROOT_PATH, "MVs_Algorithms")

for _p in (ROOT_PATH, MODULE_PATH, MV_ALGO_PATH, NODES_PATH):
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
