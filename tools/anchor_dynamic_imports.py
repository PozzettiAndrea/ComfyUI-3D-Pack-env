#!/usr/bin/env python
"""Point the vendored config-string resolvers at nodes/_vendor_paths.py.

tools/relativize_imports.py fixes import STATEMENTS. This fixes the other
half: dotted module STRINGS resolved at runtime, which no static rewrite can
reach because they live in YAML (`target: CRM.imagedream.ldm.interface.X`)
and in registry lookups.

Every vendored family ships the same LDM-derived idiom:

    return getattr(importlib.import_module(module, package=None), cls)

`module` there is a legacy bare-rooted string. This script rewrites those
calls to go through the shared resolver, which maps the legacy root onto its
real location inside the package and leaves genuine third-party targets alone:

    return getattr(_vendor_import(module), cls)

and inserts the matching package-relative import of the helper.

IDEMPOTENT: already-anchored call sites are skipped, so re-run after
`git merge upstream/main`.

Usage:  python tools/anchor_dynamic_imports.py [--check]
"""

from __future__ import annotations

import argparse
import ast
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
PACK = os.path.dirname(HERE)
NODES = os.path.join(PACK, "nodes")

HELPER = "_vendor_paths"
ALIAS = "_vendor_import"

# `importlib.import_module(X, package=None)` and the bare `importlib.import_module(X)`
# that precedes it in the reload branch. Anything anchored to __name__, a literal
# third-party target, or an f".{...}" relative name is left alone.
CALL = re.compile(
    r"importlib\.import_module\(\s*(?P<arg>[A-Za-z_][A-Za-z_0-9]*)\s*"
    r"(?:,\s*package\s*=\s*(?:None|__name__)\s*)?\)"
)


def rel_import(path: str) -> str:
    """Package-relative import line for the helper, from this file's location."""
    rel = os.path.relpath(path, NODES).replace("\\", "/").split("/")
    depth = len(rel) - 1  # directories between nodes/ and the file
    return f"from {'.' * (depth + 1)}{HELPER} import import_module as {ALIAS}"


def patch(path: str, apply: bool) -> int:
    src = open(path, encoding="utf-8", errors="replace").read()
    if f"import_module as {ALIAS}" in src:
        return 0  # already anchored
    new, n = CALL.subn(lambda m: f"{ALIAS}({m.group('arg')})", src)
    if not n:
        return 0
    # insert the helper import after the module docstring / first import block
    try:
        tree = ast.parse(new)
    except SyntaxError:
        return 0
    line = 0
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)) or (
            isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
        ):
            line = node.end_lineno
        else:
            break
    lines = new.splitlines(keepends=True)
    lines.insert(line, rel_import(path) + "\n")
    new = "".join(lines)
    if apply:
        open(path, "w", encoding="utf-8", newline="").write(new)
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    apply = not args.check
    files, total = 0, 0
    for d, dirs, fs in os.walk(NODES):
        dirs[:] = [x for x in dirs if x != "__pycache__"]
        for f in fs:
            if not f.endswith(".py") or f == f"{HELPER}.py":
                continue
            n = patch(os.path.join(d, f), apply)
            if n:
                files += 1
                total += n
    verb = "would anchor" if args.check else "anchored"
    print(f"files {verb}  : {files}")
    print(f"calls {verb}  : {total}")
    return 1 if (args.check and total) else 0


if __name__ == "__main__":
    raise SystemExit(main())
