#!/usr/bin/env python
"""Rewrite upstream's bare top-level imports into package-relative imports.

Upstream ComfyUI-3D-Pack makes its vendored model families importable by
scattering sys.path entries: three at the top (nodes/, Gen_3D_Modules/,
MVs_Algorithms/) plus ~11 more inserted at runtime from inside the vendored
code itself (Hunyuan3D_2_1/hy3dshape, PartCrafter, Stable3DGen, ...). That
turns 71 directory names -- including `src`, `utils`, `dinov2` and
`custom_rasterizer` -- into TOP-LEVEL module names inside the environment,
where any one of them can shadow, or be shadowed by, a real pip package.

This script converts every such import into a package-relative one, which
resolves through __package__ and never consults sys.path:

    from craftsman.utils.config import ExperimentConfig
        -> from ...utils.config import ExperimentConfig     (inside craftsman)
    from mesh_processer.mesh import Mesh
        -> from ....mesh_processer.mesh import Mesh         (from a family)
    import craftsman
        -> from .. import craftsman

It is IDEMPOTENT: relative imports are left alone, so re-running after a
`git merge upstream/main` re-applies the transformation to freshly pulled
code. That is the intended workflow -- do not hand-edit what this generates.

Usage:
    python tools/relativize_imports.py            # apply
    python tools/relativize_imports.py --check    # report only, exit 1 if work remains
"""

from __future__ import annotations

import argparse
import ast
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
PACK = os.path.dirname(HERE)
NODES = os.path.join(PACK, "nodes")

# Every directory the vendored code treats as a sys.path root. "" is nodes/.
# The nested ones come from sys.path.insert/append calls inside the families.
#
# ORDER MATTERS and models sys.path precedence: the nested roots are added with
# sys.path.insert(0, ...), so they OUTRANK the top-level ones. Sorted
# deepest-first below for exactly that reason -- e.g. `hy3dshape` must resolve
# to Hunyuan3D_2_1/hy3dshape/hy3dshape (the inner package, put on the path by
# hy3dshape/hy3dshape/__init__.py) and NOT to Hunyuan3D_2_1/hy3dshape.
ROOTS = [
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
]

# Never rewrite these even if a directory happens to share the name.
STDLIB = set(getattr(sys, "stdlib_module_names", ()))
NEVER = STDLIB | {
    "torch", "numpy", "cv2", "PIL", "trimesh", "einops", "diffusers",
    "transformers", "scipy", "kornia", "imageio", "matplotlib", "omegaconf",
    "pytorch_lightning", "safetensors", "huggingface_hub", "tqdm", "yaml",
}


def discover_names() -> dict[str, tuple[str, ...]]:
    """Map bare top-level name -> its canonical dotted path under nodes/.

    First root wins, mirroring sys.path shadowing order.
    """
    out: dict[str, tuple[str, ...]] = {}
    for root in ROOTS:
        d = os.path.join(NODES, root)
        if not os.path.isdir(d):
            continue
        prefix = tuple(p for p in root.split("/") if p)
        for entry in sorted(os.listdir(d)):
            name = entry[:-3] if entry.endswith(".py") else entry
            if name.startswith((".", "_")) or name in NEVER or name in out:
                continue
            full = os.path.join(d, entry)
            if os.path.isdir(full) or entry.endswith(".py"):
                out[name] = prefix + (name,)
    return out


def file_package(path: str) -> tuple[str, ...]:
    """Dotted package of a .py file, relative to nodes/."""
    rel = os.path.relpath(path, NODES).replace("\\", "/")
    parts = rel.split("/")
    parts = parts[:-1] if parts[-1] == "__init__.py" else parts[:-1]
    return tuple(parts)


def relative_for(pkg: tuple[str, ...], target: tuple[str, ...]) -> tuple[int, tuple[str, ...]]:
    """Level (number of dots) and remainder to reach `target` from `pkg`."""
    common = 0
    for a, b in zip(pkg, target):
        if a != b:
            break
        common += 1
    return len(pkg) - common + 1, target[common:]


def render_names(aliases) -> str:
    return ", ".join(a.name + (f" as {a.asname}" if a.asname else "") for a in aliases)


def rewrite_file(path: str, names: dict[str, tuple[str, ...]], apply: bool) -> int:
    src = open(path, encoding="utf-8", errors="replace").read()
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return 0
    pkg = file_package(path)
    lines = src.splitlines(keepends=True)
    # byte offsets per line so we can splice by (lineno, col_offset)
    starts, off = [], 0
    for ln in lines:
        starts.append(off)
        off += len(ln)

    edits: list[tuple[int, int, str]] = []

    for node in ast.walk(tree):
        indent = " " * node.col_offset if hasattr(node, "col_offset") else ""
        if isinstance(node, ast.ImportFrom):
            if node.level != 0 or not node.module:
                continue
            head = node.module.split(".")[0]
            if head not in names:
                continue
            target = names[head] + tuple(node.module.split(".")[1:])
            level, rest = relative_for(pkg, target)
            new = f"from {'.' * level}{'.'.join(rest)} import {render_names(node.names)}"
        elif isinstance(node, ast.Import):
            if not any(a.name.split(".")[0] in names for a in node.names):
                continue
            stmts = []
            for a in node.names:
                head = a.name.split(".")[0]
                if head not in names:
                    stmts.append(f"import {a.name}" + (f" as {a.asname}" if a.asname else ""))
                    continue
                target = names[head] + tuple(a.name.split(".")[1:])
                # `import a.b.c` binds `a` unless aliased; `import a.b.c as x` binds x.
                if a.asname:
                    level, rest = relative_for(pkg, target[:-1])
                    stmts.append(
                        f"from {'.' * level}{'.'.join(rest)} import {target[-1]} as {a.asname}"
                    )
                else:
                    # bind the head name -> import the head package itself
                    level, rest = relative_for(pkg, names[head][:-1])
                    dotted = "." * level + ".".join(rest)
                    stmts.append(f"from {dotted} import {head}")
            new = ("\n" + indent).join(stmts)
        else:
            continue

        s = starts[node.lineno - 1] + node.col_offset
        e = starts[node.end_lineno - 1] + node.end_col_offset
        edits.append((s, e, new))

    if not edits:
        return 0
    if apply:
        for s, e, new in sorted(edits, reverse=True):
            src = src[:s] + new + src[e:]
        open(path, "w", encoding="utf-8", newline="").write(src)
    return len(edits)


def ensure_packages(names: dict[str, tuple[str, ...]], apply: bool) -> list[str]:
    """Relative imports need real packages: add __init__.py where missing."""
    created = []
    wanted = {os.path.join(NODES, *t) for t in names.values()}
    wanted |= {os.path.join(NODES, *[p for p in r.split("/") if p]) for r in ROOTS if r}
    for d in sorted(wanted):
        if not os.path.isdir(d):
            continue
        init = os.path.join(d, "__init__.py")
        if not os.path.exists(init):
            created.append(os.path.relpath(init, PACK))
            if apply:
                open(init, "w", encoding="utf-8").close()
    return created


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="report only")
    args = ap.parse_args()
    apply = not args.check

    names = discover_names()
    created = ensure_packages(names, apply)

    total, touched = 0, Counter()
    for d, dirs, fs in os.walk(NODES):
        dirs[:] = [x for x in dirs if x != "__pycache__"]
        for f in fs:
            if not f.endswith(".py"):
                continue
            p = os.path.join(d, f)
            n = rewrite_file(p, names, apply)
            if n:
                total += n
                touched[os.path.relpath(p, PACK)] = n

    verb = "would rewrite" if args.check else "rewrote"
    print(f"magic top-level names : {len(names)}")
    print(f"__init__.py {'missing' if args.check else 'created'}   : {len(created)}")
    print(f"files {verb:<14}: {len(touched)}")
    print(f"imports {verb:<12}: {total}")
    if args.check and (total or created):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
