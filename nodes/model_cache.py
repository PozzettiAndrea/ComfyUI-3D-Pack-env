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


def _say(msg):
    """Print where it will actually be seen.

    logging.getLogger("comfy3d") has no handler in the isolated worker, so every
    log.info() here went nowhere -- which made "are these models actually under
    ComfyUI's control?" unanswerable from a run log. Whether a model is managed
    or silently off the books is exactly the thing worth being able to read.
    """
    print(f"[Comfy3D][model_cache] {msg}", flush=True)

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
    import comfy.model_management
    import comfy.model_patcher

    # size=: ModelPatcher otherwise lazily derives the size from state_dict(), and
    # everything that plans around VRAM -- free_memory(), the eviction ordering in
    # load_models_gpu() -- reads it. Measuring up front with module_size() is what
    # SAM3 does (sam3_model_patcher.py:37) and it costs one pass over the params.
    return comfy.model_patcher.ModelPatcher(
        model,
        load_device=load_device,
        offload_device=offload_device,
        size=comfy.model_management.module_size(model),
    )



# How far _module_components will recurse through wrapper objects. Two is
# enough for every shape in this pack (pipeline -> .models dict -> module, or
# pipeline -> wrapper -> .pipeline -> module) and stops a cyclic or absurdly
# nested object from walking forever.
_MAX_COMPONENT_DEPTH = 3

# Types that are never worth descending into, either because they cannot hold a
# model or because walking them is expensive or unsafe.
_SKIP_CONTAINER_TYPES = (str, bytes, bytearray, int, float, bool, complex)


def _is_container(value):
    """Whether to look inside `value` for models.

    Deliberately conservative: only plain objects with a __dict__, and never a
    tensor (walking a tensor's attributes is pointless and touching .data can
    trigger device work) or a torch device/dtype.
    """
    import torch

    if value is None or isinstance(value, _SKIP_CONTAINER_TYPES):
        return False
    if isinstance(value, (torch.Tensor, torch.device, torch.dtype)):
        return False
    if isinstance(value, type):          # a class object, not an instance
        return False
    if value.__class__.__module__ == "builtins":
        return False
    return hasattr(value, "__dict__")


def _module_components(obj, _depth=0, _seen=None):
    """Every nn.Module inside `obj`, by name.

    A diffusers pipeline is not itself an nn.Module and has no state_dict, so
    ModelPatcher cannot size it -- but it holds several that can be wrapped
    individually. `.components` is diffusers' own accessor and also returns the
    tokenizer, scheduler and feature_extractor, which are not modules; those are
    skipped. Wrapping per component is better than one patcher would be anyway:
    ComfyUI can evict a text encoder while the unet stays resident.

    Attributes are searched RECURSIVELY, and dicts/lists are searched too,
    because several pipelines here keep their sub-models in a plain dict:

        Stable3DGen/trellis_fork/pipelines/base.py:17   self.models = models
        Hunyuan3D_2_1/hy3dpaint/textureGenPipeline.py:90 self.models = {}
        Hunyuan3D_V2/hy3dgen/texgen/pipelines.py:71      self.models = {}

    A flat vars() scan returns the dict, not the modules in it, so those
    pipelines yielded NOTHING, fell through to the unmanaged branch in
    managed(), and their multi-GB weights stayed resident for the life of the
    worker -- invisible to every eviction path. That was the single biggest
    reason this pack could fill a 24 GB card.

    Recursion is bounded and deduped by id(): a wrapper (e.g.
    Multiview_Diffusion_Net.pipeline) can reach the same module by two routes,
    and wrapping one module in two ModelPatchers would double-count its size and
    let one patcher offload weights the other believes are resident.
    """
    import torch

    if _seen is None:
        _seen = set()

    found = {}

    def take(name, value):
        if isinstance(value, torch.nn.Module):
            if id(value) not in _seen:
                _seen.add(id(value))
                found[name] = value
            return True
        return False

    def walk(name, value):
        if take(name, value):
            return
        if _depth >= _MAX_COMPONENT_DEPTH:
            return
        if isinstance(value, dict):
            for k, v in value.items():
                walk(f"{name}.{k}", v)
        elif isinstance(value, (list, tuple)):
            for i, v in enumerate(value):
                walk(f"{name}[{i}]", v)
        elif _is_container(value):
            # A plain wrapper object holding a pipeline/model. Recurse one level
            # deeper rather than losing everything it owns.
            found.update(_module_components(value, _depth + 1, _seen))

    comps = getattr(obj, "components", None)
    if isinstance(comps, dict):
        for k, v in comps.items():
            walk(k, v)
        if found:
            return found

    try:
        attrs = vars(obj)
    except TypeError:
        return found

    for k, v in attrs.items():
        if k.startswith("__"):
            continue
        walk(k, v)
    return found


def _offload_others(current_key):
    """Return every OTHER reloadable model's weights to the offload device.

    This is the "models take turns" step. Without it, load_models_gpu() fast-exits
    for anything already in current_loaded_models, so a graph that runs three
    families in sequence keeps all three resident and the third one OOMs.

    Only entries flagged `reloadable` are touched. A reloadable entry is one whose
    consumers call managed() again on every execution (the recipe families), so
    ComfyUI reloads it on next use. Entries created by a loader that hands the
    live object straight to a consumer are NOT reloadable: unpatching those would
    silently leave a model on CPU with nothing to bring it back.
    """
    import comfy.model_management

    with _LOCK:
        targets = [pat
                   for k, (_obj, patchers, reloadable) in _PATCHERS.items()
                   if reloadable and k != current_key
                   for pat in patchers]

    if not targets:
        return

    for patcher in targets:
        patcher.unpatch_model(device_to=patcher.offload_device)
    comfy.model_management.soft_empty_cache()
    _say(f"offloaded {len(targets)} patcher(s) from other models to make room")


def managed(kind, config, builder, post_load=None, reloadable=False):
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
        # Make room BEFORE building. Some builders still place weights on the
        # device themselves, so their peak lands inside builder() -- before any
        # ModelPatcher exists for this model, and therefore before the
        # load_models_gpu() below could evict anything.
        #
        # This used to be unload_all_models(), which was WRONG here: it unpatches
        # every registered patcher including models whose consumers never call
        # managed() again (TripoSR, StableFast3D, InstantMesh, the CRM family...).
        # Those hold the live object directly, so nothing reloads them and their
        # next forward runs against CPU weights. _offload_others() touches only
        # entries flagged reloadable -- the recipe families, whose consumers
        # re-resolve on every execution and so get reloaded automatically.
        _offload_others(key)

        obj = builder(config)

        if isinstance(obj, torch.nn.Module):
            targets = {"": obj}
        else:
            targets = _module_components(obj)

        if not targets:
            _say(f"{kind}: holds no nn.Module -- caching UNMANAGED, its VRAM is off ComfyUI's books")
            return resolve(kind, config, lambda _cfg: obj)

        load_device = comfy.model_management.get_torch_device()
        offload_device = comfy.model_management.unet_offload_device()

        patchers = []
        for name, module in targets.items():
            try:
                patchers.append(_wrap(_patch_target(module),
                                     load_device, offload_device))
            except Exception as exc:
                _say(f"{kind}: component {name or kind!r} could NOT be wrapped "
                     f"({exc}); its VRAM stays off the books")

        if not patchers:
            return resolve(kind, config, lambda _cfg: obj)

        with _LOCK:
            entry = _PATCHERS.setdefault(key, (obj, patchers, reloadable))
        _say(f"{kind}: ComfyUI-managed, {len(entry[1])} component(s) wrapped in ModelPatchers")

    obj, patchers, _reloadable = entry

    # Models take turns: give the VRAM back from any other reloadable family
    # before pulling this one in. TRELLIS2 does the same thing explicitly at its
    # stage boundaries (nodes/stages.py:325-335, called 20x).
    _offload_others(key)
    # Every call, not just on a miss: another node may have offloaded us since.
    # No force_full_load: 140 vendored files here were deliberately converted to
    # comfy.ops.manual_cast (see e.g. TRELLIS/trellis/models/sparse_structure_flow.py:10-15)
    # precisely so ModelPatcher CAN lowvram-offload them -- comfy.ops classes carry
    # comfy_cast_weights, torch.nn classes do not. Forcing a full load would throw
    # that away and OOM where a partial load would have fit. None of the sibling
    # packs passes it either.
    #
    # The gap it would have covered is narrower than it looks: only stock diffusers
    # classes arriving via from_pretrained lack the cast hooks, and for those a
    # partial load can silently skip a module. That is worth fixing per-family, not
    # by disabling partial loading globally.
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
                   for k, (_obj, patchers, reloadable) in _PATCHERS.items()
                   if reloadable and (kind is None or k[0] == kind)
                   for pat in patchers]

    for patcher in targets:
        patcher.unpatch_model(device_to=patcher.offload_device)
    if targets:
        comfy.model_management.soft_empty_cache()
