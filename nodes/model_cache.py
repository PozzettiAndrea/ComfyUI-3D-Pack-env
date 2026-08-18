"""Module-level model cache backing the config-dict loader sockets.

Loader nodes do not emit live models. They emit a JSON-safe *recipe* dict
describing how to build one; the consuming node calls `resolve()` to
materialize it. Two things fall out of that:

  * The socket payload is plain data (str/int/bool/list/dict), so it crosses
    comfy-env's isolation boundary without pickling a multi-gigabyte pipeline.
    See TRELLIS2's nodes_loader.py for the same idea -- note how it stores
    dtype as "bf16" rather than torch.bfloat16 for exactly this reason.
  * Identical recipes share one object, so re-running a workflow does not
    re-load weights. Loading happens on first use inside the consumer, not in
    the loader node.

Recipes are ordered: a mutator node (set scheduler, load a state dict) appends
to `ops` rather than mutating a live object, and `resolve()` replays the ops
after building the base model. Two recipes that differ only in op order are
different cache entries, which is correct -- the operations are not
commutative.
"""

import json
import logging
import threading

log = logging.getLogger("comfy3d")

_CACHE = {}
_PATCHERS = {}
_POST_LOADED = set()
_LOCK = threading.Lock()


def freeze(config):
    """Stable hashable key for a JSON-safe config dict.

    sort_keys makes the key independent of insertion order; default=str keeps
    a stray non-serializable value from raising instead of just missing a
    cache hit.
    """
    return json.dumps(config, sort_keys=True, default=str)


def resolve(kind, config, builder):
    """Return the model for `config`, building it via `builder` on a miss.

    `builder` runs OUTSIDE the lock -- loading a pipeline takes minutes and
    holding the lock would serialize every other node in the worker. The
    setdefault on re-acquire means a concurrent duplicate build loses the race
    harmlessly rather than replacing a model another node already handed out.
    """
    key = (kind, freeze(config))
    with _LOCK:
        hit = _CACHE.get(key)
    if hit is not None:
        log.debug("[Comfy3D] model cache hit: %s", kind)
        return hit

    log.info("[Comfy3D] model cache miss, building: %s", kind)
    obj = builder(config)
    with _LOCK:
        return _CACHE.setdefault(key, obj)


def clear(kind=None):
    """Drop cached models -- all of them, or just one kind.

    Managed models are dropped from _PATCHERS too; ComfyUI still holds its own
    reference until its next eviction pass, which is its call to make.
    """
    with _LOCK:
        for store in (_CACHE, _PATCHERS):
            if kind is None:
                store.clear()
            else:
                for k in [k for k in store if k[0] == kind]:
                    del store[k]
        for k in [k for k in _POST_LOADED if kind is None or k[0] == kind]:
            _POST_LOADED.discard(k)


def recipe(kind, **fields):
    """Build a fresh recipe dict. `ops` is always present so mutator nodes
    never have to special-case a missing key."""
    cfg = {"kind": kind, "ops": []}
    cfg.update(fields)
    return cfg


def with_op(config, op, **fields):
    """Return a copy of `config` with one operation appended.

    Copying rather than mutating matters: a loader output can feed two
    different mutator chains in the same graph, and ComfyUI hands both
    branches the same object.
    """
    out = dict(config)
    out["ops"] = list(config.get("ops", [])) + [{"op": op, **fields}]
    return out


# ---------------------------------------------------------------------------
# ComfyUI-managed models
# ---------------------------------------------------------------------------
#
# Everything above caches a built object in `_CACHE` and keeps it forever. That
# is fine for a recipe dict and wrong for a model: a plain dict is invisible to
# ComfyUI's memory manager, so the weights sit in VRAM for the life of the
# process and the "Free memory" button -- which calls unload_all_models() --
# cannot reach them. Load two generators in one session and you hold both.
#
# The fix is the pattern ComfyUI-TRELLIS2 uses in stages.py: build the model on
# CPU, wrap it in a ModelPatcher, cache the *patcher*, and call load_models_gpu()
# on every access. ComfyUI then owns the placement decision and can evict this
# model to make room for another one, exactly as it does for a checkpoint.
#
# Node sockets are unaffected: `managed()` returns the live model, same as
# before. Only the ownership of its VRAM changes.


def _patch_target(module):
    """Adapt a module so ComfyUI's ModelPatcher can own it.

    ModelPatcher assigns `model.device` while loading and unpatching
    (model_patcher.py:1103, :1157). diffusers' ModelMixin exposes `device` as a
    read-only property reporting the first parameter's device, so the
    assignment raises "property 'device' ... has no setter" and the load dies
    half-done. Wrapping puts a plain settable attribute in the way; `.to()`,
    `.state_dict()` and the parameters still belong to the real module, which
    is all ModelPatcher touches otherwise.

    Modules that already accept the assignment are returned untouched, so this
    costs nothing for the pack's own nn.Modules.
    """
    import torch

    try:
        module.device = getattr(module, "device", None)
        return module
    except AttributeError:
        pass

    class _PatchTarget(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner
            self.device = None

        def forward(self, *a, **k):
            return self.inner(*a, **k)

    return _PatchTarget(module)


def _wrap(model, load_device, offload_device):
    """Wrap a module in a ModelPatcher.

    Raises if the object cannot be wrapped -- ModelPatcher sizes a model through
    state_dict(), so a diffusers pipeline, a tuple of models or a plain wrapper
    object has to keep the old unmanaged behaviour. `managed()` catches that.
    """
    import comfy.model_patcher

    return comfy.model_patcher.ModelPatcher(
        model, load_device=load_device, offload_device=offload_device
    )



def _module_components(obj):
    """The nn.Modules inside `obj`, by name.

    A diffusers pipeline is not itself an nn.Module and has no state_dict, so
    ModelPatcher cannot size it -- but it holds several that can be wrapped
    individually. `.components` is diffusers' own accessor and also returns the
    tokenizer, scheduler and feature_extractor, which are not modules; those are
    skipped. Wrapping per component is better than one patcher would be anyway:
    ComfyUI can evict a text encoder while the unet stays resident.
    """
    import torch

    comps = getattr(obj, "components", None)
    if isinstance(comps, dict):
        found = {k: v for k, v in comps.items() if isinstance(v, torch.nn.Module)}
        if found:
            return found
    # Not a diffusers pipeline: take whatever modules it holds as attributes.
    try:
        attrs = vars(obj)
    except TypeError:
        return {}
    return {k: v for k, v in attrs.items() if isinstance(v, torch.nn.Module)}


def managed(kind, config, builder, post_load=None):
    """Build once, hand ComfyUI the VRAM, return the live object.

    Caches ModelPatchers, not models. A plain dict of models is invisible to
    ComfyUI: unload_all_models() -- what the Free memory button calls -- cannot
    reach it, so the weights sit in VRAM for the life of the process. Wrapped,
    ComfyUI can evict them to make room, exactly as it does for a checkpoint.

    Handles three shapes:
      * an nn.Module          -> one patcher
      * a pipeline of modules -> one patcher per component, all loaded together
      * anything else         -> plain cache, logged, VRAM off the books

    Callers and node sockets are unchanged: this returns whatever the builder
    returned. The builder SHOULD leave weights on CPU -- one that ends in
    `.to(DEVICE)` still works, but it has already spent the VRAM before ComfyUI
    was asked whether there was room.

    `post_load(obj, device)` runs once after the first GPU load, for setup that
    must allocate on the target device -- InstantMesh's FlexiCubes geometry is
    the case it exists for; it cannot go in a builder that runs on CPU.
    """
    import torch
    import comfy.model_management

    key = (kind, freeze(config))

    with _LOCK:
        entry = _PATCHERS.get(key)

    if entry is None:
        obj = builder(config)

        if isinstance(obj, torch.nn.Module):
            targets = {"": obj}
        else:
            targets = _module_components(obj)

        if not targets:
            log.info("[Comfy3D] %s holds no nn.Module; caching unmanaged", kind)
            return resolve(kind, config, lambda _cfg: obj)

        load_device = comfy.model_management.get_torch_device()
        offload_device = comfy.model_management.unet_offload_device()

        patchers = []
        for name, module in targets.items():
            try:
                patchers.append(_wrap(_patch_target(module),
                                     load_device, offload_device))
            except Exception as exc:
                log.warning("[Comfy3D] %s: component %r could not be wrapped "
                            "(%s); its VRAM stays off the books",
                            kind, name or kind, exc)

        if not patchers:
            return resolve(kind, config, lambda _cfg: obj)

        with _LOCK:
            entry = _PATCHERS.setdefault(key, (obj, patchers))
        log.info("[Comfy3D] %s is ComfyUI-managed (%d component(s))",
                 kind, len(entry[1]))

    obj, patchers = entry
    # Every call, not just on a miss: another node may have offloaded us since.
    comfy.model_management.load_models_gpu(patchers)

    if post_load is not None and key not in _POST_LOADED:
        post_load(obj, patchers[0].load_device)
        _POST_LOADED.add(key)

    return obj


def offload(kind=None):
    """Return managed models to CPU so ComfyUI can reuse the VRAM.

    Needed because load_models_gpu() fast-exits for a model already in
    current_loaded_models -- without an explicit unpatch, models accumulate on
    the GPU across a workflow instead of taking turns.
    """
    import comfy.model_management

    with _LOCK:
        targets = [pat
                   for k, (_obj, patchers) in _PATCHERS.items()
                   if kind is None or k[0] == kind
                   for pat in patchers]

    for patcher in targets:
        patcher.unpatch_model(device_to=patcher.offload_device)
    if targets:
        comfy.model_management.soft_empty_cache()
