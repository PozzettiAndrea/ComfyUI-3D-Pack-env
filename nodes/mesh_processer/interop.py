"""The single boundary between trimesh (the wire) and torch (the compute).

The `TRIMESH` socket carries a plain `trimesh.Trimesh`: CPU, float64 vertices,
int64 faces -- not a choice, trimesh's setters force both and there is no
opt-out. Everything the rasterizers need that a Trimesh cannot hold rides
beside it on `MESH_EXTRAS`, a plain dict of torch tensors:

    vt, ft, vn, fn, vc, albedo, metallicRoughness

`ft`/`fn` are the reason the split exists at all. A Trimesh has ONE index
buffer; this pack needs three, because a vertex at a UV seam has one position
and several UVs. Keeping them out here means `align_v_to_vt` is never forced,
so no vertex is duplicated and an OBJ still round-trips.

Two invariants, both load-bearing:

  * Nothing outside this module calls `torch.from_numpy` on a Trimesh array or
    `.cpu().numpy()` on mesh geometry. Grep-enforceable, and the only thing
    that will stop this rotting.
  * Never alias a Trimesh array into torch. `TrackedArray` invalidates its
    cache from Python-level `__setitem__`; a torch write through a shared
    buffer bypasses that, and `vertex_normals`/`bounds`/`area` then return
    stale values forever with no error. Always copy.

The V-axis flip lives here and nowhere else. This pack stores UV with V
inverted relative to the OBJ/glTF convention (mesh.py:210 on load, :405 on
trimesh load, :855 flipping back on write); `trimesh.visual.uv` is unflipped.
Miss the flip and textures render vertically mirrored -- and they render, so
nothing raises.
"""

import numpy as np
import torch
import trimesh

_EXTRA_FIELDS = ("vt", "ft", "vn", "fn", "vc", "albedo", "metallicRoughness")


def _flip_v(uv):
    """Swap between this pack's UV convention and trimesh's. Self-inverse.

    Always returns a fresh array -- writing `uv[:, 1] = 1 - uv[:, 1]` in place
    is what mesh.py:405 does today, and it mutates the caller's live Trimesh.
    """
    uv = np.asarray(uv, dtype=np.float64).copy()
    uv[:, 1] = 1.0 - uv[:, 1]
    return uv


def to_trimesh(mesh):
    """Mesh -> (Trimesh, extras dict or None).

    UV is mirrored onto `visual.uv` only when it is vertex-aligned, i.e. when
    there is no separate `ft`. That is the only case trimesh can represent, and
    it is what makes `tm.export()` carry a texture for free. When `ft` differs
    the UV stays in extras alone -- putting a mismatched uv on the Trimesh
    would be silently wrong.
    """
    v = mesh.v.detach().cpu().numpy().astype(np.float64)
    f = mesh.f.detach().cpu().numpy().astype(np.int64)

    tm = trimesh.Trimesh(vertices=v, faces=f, process=False)

    aligned = mesh.vt is not None and (
        mesh.ft is None or torch.equal(mesh.ft.to(mesh.f.dtype), mesh.f)
    )
    if aligned and mesh.vt.shape[0] == mesh.v.shape[0]:
        tm.visual = trimesh.visual.TextureVisuals(
            uv=_flip_v(mesh.vt.detach().cpu().numpy())
        )

    extras = {}
    for name in _EXTRA_FIELDS:
        val = getattr(mesh, name, None)
        if val is not None:
            extras[name] = val.detach()
    return tm, (extras or None)


def from_trimesh(tm, extras=None, device=None):
    """(Trimesh, extras) -> Mesh, for code that still wants the old container.

    Transitional: it exists so consumers can migrate one at a time instead of
    in a single merge. Every use of it is a site not yet converted.
    """
    from .mesh import Mesh

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    v = torch.as_tensor(np.asarray(tm.vertices), dtype=torch.float32, device=device)
    f = torch.as_tensor(np.asarray(tm.faces), dtype=torch.int32, device=device)

    kwargs = {name: None for name in _EXTRA_FIELDS}
    kwargs.update({k: val.to(device) for k, val in (extras or {}).items()})

    # Fall back to the Trimesh's own UV when extras carry none. Flip back into
    # this pack's convention on the way in.
    if kwargs.get("vt") is None:
        uv = getattr(getattr(tm, "visual", None), "uv", None)
        if uv is not None and len(uv):
            kwargs["vt"] = torch.as_tensor(
                _flip_v(uv), dtype=torch.float32, device=device
            )
            kwargs["ft"] = f

    return Mesh(v=v, f=f, device=device, **kwargs)


_EXTRAS_KEY = "comfy3d_extras"


def wire_out(mesh):
    """Whatever a node returns -> the `TRIMESH` socket payload.

    Extras ride in `metadata` rather than on a second output slot. A second
    slot would shift every downstream link index in all 40 shipped workflows
    and every user's saved graph; metadata costs nothing and changes no slot.

    That is only safe because a wire mesh is built fresh here at every return.
    `trimesh.util.concatenate`, `submesh` and `simplify_quadric_decimation` all
    drop metadata -- the pack calls the last of those at nodes.py:4772, but on
    a node-local mesh that is rebuilt through here before it reaches the wire.
    Returning a mesh that has been through one of those without rebuilding it
    would silently lose the texture.
    """
    if isinstance(mesh, trimesh.Trimesh) or mesh is None:
        return mesh
    tm, extras = to_trimesh(mesh)
    if extras:
        tm.metadata[_EXTRAS_KEY] = extras
    return tm


def wire_in(payload, device=None):
    """The `TRIMESH` socket payload -> the container the node body expects.

    Tolerates a Mesh so a half-migrated graph, or a node reached from code that
    has not gone through wire_out, still works instead of failing obscurely.
    """
    if payload is None:
        return None
    if not isinstance(payload, trimesh.Trimesh):
        return payload
    extras = (payload.metadata or {}).get(_EXTRAS_KEY)
    return from_trimesh(payload, extras, device=device)


def raster_tensors(tm, extras=None, device=None):
    """(Trimesh, extras) -> the tensors a rasterizer wants.

    nvdiffrast documents float32 positions and int32 indices, on GPU,
    contiguous. This is the one place that narrowing happens.

    Returns copies, never views onto the Trimesh's arrays -- see the aliasing
    note in the module docstring.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    v = torch.as_tensor(np.asarray(tm.vertices), dtype=torch.float32, device=device).contiguous()
    f = torch.as_tensor(np.asarray(tm.faces), dtype=torch.int32, device=device).contiguous()

    out = {"v": v, "f": f}
    for name in _EXTRA_FIELDS:
        val = (extras or {}).get(name)
        if val is None:
            out[name] = None
            continue
        val = val.to(device)
        if name in ("ft", "fn"):
            val = val.to(torch.int32)
        elif val.is_floating_point():
            val = val.to(torch.float32)
        out[name] = val.contiguous()

    # A rasterizer indexing UVs with the position buffer is the common
    # degenerate case; make it explicit rather than letting it read None.
    if out["ft"] is None and out["vt"] is not None:
        out["ft"] = f
    if out["fn"] is None and out["vn"] is not None:
        out["fn"] = f
    return out
