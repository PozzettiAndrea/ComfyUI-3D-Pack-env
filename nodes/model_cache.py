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
    """Drop cached models -- all of them, or just one kind."""
    with _LOCK:
        if kind is None:
            _CACHE.clear()
        else:
            for k in [k for k in _CACHE if k[0] == kind]:
                del _CACHE[k]


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
