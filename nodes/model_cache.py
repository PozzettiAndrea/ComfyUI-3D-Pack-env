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



def managed(kind, config, builder, post_load=None):
    """Build once on CPU, cache the ModelPatcher, hand ComfyUI the VRAM decision.

    Returns the live model, so callers and node sockets are unchanged. The
    builder MUST leave the model on CPU -- a builder that ends in `.to(DEVICE)`
    defeats the whole point, because the weights are already in VRAM before
    ComfyUI is asked whether there is room.

    `post_load(model, device)` runs once, after the first GPU load, for setup
    that needs to allocate on the target device -- InstantMesh's FlexiCubes
    geometry is the case this exists for. It cannot go in the builder because
    the builder runs on CPU.
    """
    import torch
    import comfy.model_management

    key = (kind, freeze(config))

    with _LOCK:
        patcher = _PATCHERS.get(key)

    if patcher is None:
        model = builder(config)

        if not isinstance(model, torch.nn.Module) or not hasattr(model, "state_dict"):
            # Not wrappable. Fall back to the plain cache so behaviour is
            # unchanged, but say so -- this model's VRAM stays off the books.
            log.info("[Comfy3D] %s is not an nn.Module; caching unmanaged", kind)
            return resolve(kind, config, lambda _cfg: model)

        load_device = comfy.model_management.get_torch_device()
        offload_device = comfy.model_management.unet_offload_device()

        try:
            patcher = _wrap(model, load_device, offload_device)
        except Exception as exc:
            log.warning("[Comfy3D] %s could not be wrapped in a ModelPatcher (%s); "
                        "caching unmanaged", kind, exc)
            return resolve(kind, config, lambda _cfg: model)

        with _LOCK:
            patcher = _PATCHERS.setdefault(key, patcher)
        log.info("[Comfy3D] built %s on CPU, now ComfyUI-managed", kind)

    # Every call, not just on a miss: another node may have offloaded us since.
    comfy.model_management.load_models_gpu([patcher])

    if post_load is not None and key not in _POST_LOADED:
        post_load(patcher.model, patcher.load_device)
        _POST_LOADED.add(key)

    return patcher.model


def offload(kind=None):
    """Return managed models to CPU so ComfyUI can reuse the VRAM.

    Needed because load_models_gpu() fast-exits for a model already in
    current_loaded_models -- without an explicit unpatch, models accumulate on
    the GPU across a workflow instead of taking turns.
    """
    import comfy.model_management

    with _LOCK:
        targets = [p for k, p in _PATCHERS.items() if kind is None or k[0] == kind]

    for patcher in targets:
        patcher.unpatch_model(device_to=patcher.offload_device)
    if targets:
        comfy.model_management.soft_empty_cache()
