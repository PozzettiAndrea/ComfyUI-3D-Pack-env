"""Node package for ComfyUI-3D-Pack-enved -- imported INSIDE the isolated env.

The presence of `comfy-env.toml` next to this file is the isolation switch:
comfy-env materialises one pixi env for this directory, runs a metadata scan
inside it, and forwards node execution to a persistent worker. Nothing here is
imported by the host ComfyUI process.

Everything the nodes import lives under this directory: nodes.py, the shared
helpers (mesh_processer/, shared_utils/) and both vendored module trees
(Gen_3D_Modules/, MVs_Algorithms/).

NOTHING is added to sys.path. Upstream made its families importable by
scattering sys.path entries -- three here plus ~11 more inserted at runtime
from inside the vendored code -- which promoted 71 directory names, including
`src`, `utils`, `dinov2` and `custom_rasterizer`, to TOP-LEVEL module names
that could shadow or be shadowed by real pip packages in the same env. That is
gone:

  * every import statement is package-relative, resolved through __package__
    (tools/relativize_imports.py)
  * every dotted module STRING resolved at runtime -- config `target:` values,
    registry lookups -- goes through _vendor_paths.py, which maps the legacy
    bare root onto its real location here (tools/anchor_dynamic_imports.py)

Both scripts are idempotent: re-run them after `git merge upstream/main` to
re-apply the transformation to freshly pulled code.

The vendored families' own sys.path.insert calls are gone too -- all 11 of
them. There is no sys.path manipulation anywhere in this pack.
"""

import inspect

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
