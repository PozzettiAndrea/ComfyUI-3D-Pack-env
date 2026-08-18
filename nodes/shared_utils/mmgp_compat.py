"""Keep mmgp's safetensors patch from breaking transformers.

mmgp replaces `safetensors.safe_open` at import with its own streaming loader
(`mmgp.safetensors2.safe_open`), whose signature is:

    safe_open(filename, framework="pt", device="cpu",
              writable_tensors=True, streaming=False)

transformers 5.x calls it as (modeling_utils.py:4469-4470):

    backend, device = ("pread", "mps") if is_mps else ("mmap", "cpu")
    file_pointer = safe_open(file, framework="pt", device=device, backend=backend)

so every transformers safetensors load raises

    TypeError: safe_open() got an unexpected keyword argument 'backend'

which surfaces far from the cause -- the traceback blames diffusers loading a
text encoder, and it only appears once the weights are *safetensors*, so
converting a checkpoint away from .bin looks like it caused it.

The value transformers passes is "mmap", which is safetensors' own default, so
dropping it on mmgp's path changes nothing. mmgp is left in charge otherwise.

This is a shim for someone else's monkeypatch and should go away when mmgp
accepts **kwargs; 3.7.12 is the latest release and does not.
"""

import sys


def keep_safe_open_compatible():
    """Make the currently-installed safe_open tolerate a `backend` kwarg.

    Safe to call repeatedly and safe to call when mmgp was never imported --
    it only wraps a patch that is actually mmgp's.
    """
    try:
        import safetensors
    except ImportError:
        return

    current = getattr(safetensors, "safe_open", None)
    if current is None or getattr(current, "_comfy3d_backend_shim", False):
        return
    # Only wrap mmgp's replacement. The real safetensors binding already takes
    # `backend`, and wrapping it would silently discard a real argument.
    if getattr(current, "__module__", "") != "mmgp.safetensors2":
        return

    def safe_open(*args, backend=None, **kwargs):
        return current(*args, **kwargs)

    safe_open._comfy3d_backend_shim = True
    safe_open._comfy3d_wrapped = current
    safetensors.safe_open = safe_open

    # `from safetensors import safe_open` binds the name at import time, so any
    # module already loaded still holds the unwrapped one. transformers
    # lazy-loads modeling_utils, which is why it usually picks up mmgp's rather
    # than the original -- but fix it in place if it got there first.
    for name in ("transformers.modeling_utils", "diffusers.models.model_loading_utils"):
        mod = sys.modules.get(name)
        if mod is not None and getattr(mod, "safe_open", None) is current:
            mod.safe_open = safe_open
