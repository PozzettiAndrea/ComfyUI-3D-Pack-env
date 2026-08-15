"""Resolve upstream's legacy top-level module STRINGS to this package.

The import statements in the vendored families were rewritten to package-
relative form by tools/relativize_imports.py. But some module references are
not import statements at all -- they are dotted strings resolved at runtime:

  * config-driven instantiation, LDM style. Configs/CRM_configs/*.yaml carry
        target: CRM.imagedream.ldm.interface.LatentDiffusionInterface
    which get_obj_from_str() feeds to importlib.
  * registry lookups, e.g. craftsman.find("michelangelo-autoencoder") falling
    back to importing a dotted path.

Those strings live in data files and user-copied configs, so rewriting them
would break every config anyone already has. Instead the call sites now route
through import_module() below, which maps the legacy bare root
("CRM", "craftsman", "hy3dshape", ...) onto its real location inside this
package before importing. Unknown heads pass through untouched, so genuine
third-party targets (torch.nn.SiLU, webdataset, ...) still work.

The alias table is DISCOVERED from the directory tree at first use, not
hard-coded, so it stays correct after `git merge upstream/main` adds or
renames a family. Root order mirrors sys.path precedence: the nested roots
were originally added with sys.path.insert(0, ...) and therefore outrank the
top-level ones -- deepest first.
"""

from __future__ import annotations

import functools
import importlib
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
# This module is <pack>.nodes._vendor_paths; the package everything lives in
# is <pack>.nodes.
_PKG = __name__.rsplit(".", 1)[0]

_ROOTS = (
    "Gen_3D_Modules/CharacterGen/Stage_3D",
    "Gen_3D_Modules/Hunyuan3D_2_1/hy3dpaint",
    "Gen_3D_Modules/Hunyuan3D_2_1/hy3dshape",
    "Gen_3D_Modules/Stable3DGen/trellis_fork",
    "Gen_3D_Modules/CharacterGen",
    "Gen_3D_Modules/Hunyuan3D_2_1",
    "Gen_3D_Modules/PartCrafter",
    "Gen_3D_Modules/Stable3DGen",
    "Gen_3D_Modules/Unique3D",
    "Gen_3D_Modules",
    "MVs_Algorithms",
    "",
)

_NEVER = set(getattr(sys, "stdlib_module_names", ())) | {
    "torch", "numpy", "cv2", "PIL", "trimesh", "einops", "diffusers",
    "transformers", "scipy", "kornia", "imageio", "matplotlib", "omegaconf",
    "pytorch_lightning", "safetensors", "huggingface_hub", "tqdm", "yaml",
    "webdataset", "torchsparse", "spconv",
}


@functools.lru_cache(maxsize=1)
def _aliases() -> dict[str, str]:
    """bare top-level name -> dotted path relative to this package."""
    out: dict[str, str] = {}
    for root in _ROOTS:
        d = os.path.join(_HERE, *root.split("/")) if root else _HERE
        if not os.path.isdir(d):
            continue
        prefix = [p for p in root.split("/") if p]
        try:
            entries = sorted(os.listdir(d))
        except OSError:
            continue
        for entry in entries:
            name = entry[:-3] if entry.endswith(".py") else entry
            if name.startswith((".", "_")) or name in _NEVER or name in out:
                continue
            full = os.path.join(d, entry)
            if os.path.isdir(full) or entry.endswith(".py"):
                out[name] = ".".join(prefix + [name])
    return out


def resolve(dotted: str) -> str:
    """Map a legacy dotted module string into this package, if it is one."""
    if not dotted or dotted.startswith("."):
        return dotted
    head, _, rest = dotted.partition(".")
    target = _aliases().get(head)
    if target is None:
        return dotted
    return f"{_PKG}.{target}" + (f".{rest}" if rest else "")


def import_module(dotted: str):
    """importlib.import_module, with legacy top-level names resolved."""
    return importlib.import_module(resolve(dotted))
