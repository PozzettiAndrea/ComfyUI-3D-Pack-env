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


def keep_safetensors_compatible():
    """Make mmgp's safetensors replacements tolerate a `backend` kwarg.

    Covers both functions mmgp swaps out:

      * safetensors.safe_open   -- transformers passes backend= (see above)
      * safetensors.torch.load_file -- safetensors' OWN load_model() calls it
        as load_file(filename, device=device, backend=backend), so the moment
        mmgp.offload is imported, any load_model() raises
        "torch_load_file() got an unexpected keyword argument 'backend'".
        StableFast3D loads exactly that way.

    Safe to call repeatedly and safe to call when mmgp was never imported --
    each wrap only applies to a patch that is actually mmgp's.
    """
    _keep_safe_open_compatible()
    _keep_load_file_compatible()


def _keep_load_file_compatible():
    """Let mmgp's torch_load_file swallow `backend`.

    The value safetensors passes is its own default ("mmap"), so dropping it
    on mmgp's path changes nothing. Patching the module attribute is enough:
    load_model() resolves load_file through safetensors.torch's globals at call
    time, so it picks this up.
    """
    try:
        import safetensors.torch
    except ImportError:
        return

    current = getattr(safetensors.torch, "load_file", None)
    if current is None or getattr(current, "_comfy3d_backend_shim", False):
        return
    if getattr(current, "__module__", "") != "mmgp.safetensors2":
        return

    def load_file(*args, backend=None, **kwargs):
        return current(*args, **kwargs)

    load_file._comfy3d_backend_shim = True
    load_file._comfy3d_wrapped = current
    safetensors.torch.load_file = load_file


def _keep_safe_open_compatible():
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


#: Old name, kept so existing call sites keep working.
keep_safe_open_compatible = keep_safetensors_compatible
