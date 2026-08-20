"""The one place this pack reads a raw torch pickle.

Everywhere else loads checkpoints through comfy.utils.load_torch_file, which is
what ComfyUI expects: it prefers safetensors, forces weights_only=True on
.pt/.ckpt, and unwraps the usual "state_dict" wrapper. Two cases cannot go
through it, so they come here instead of scattering torch.load across the tree:

  * A file holding a BARE tensor rather than a mapping. load_torch_file tests
    `"state_dict" in obj`, which raises on a Tensor. Hunyuan3D-V1's
    uc_text_emb.pt is a (1, 77, 2048) tensor saved directly.

  * A checkpoint that needs a custom Unpickler. Craftsman's .ckpt references
    module paths that no longer exist in this pack, so it is read through a
    pickle shim that rewrites them on the way in -- which requires
    weights_only=False, and therefore requires trusting the file.

Prefer converting a file to safetensors over adding a caller here; that is what
was done for Era3D's prompt embeds.
"""

import torch


def load_raw_pickle(path, pickle_module=None):
    """Read a torch pickle that comfy.utils.load_torch_file cannot handle.

    pickle_module=None keeps weights_only=True, so only tensors are unpickled.
    Passing a pickle_module implies weights_only=False and means the caller has
    taken responsibility for the file's provenance.
    """
    kwargs = {"map_location": torch.device("cpu")}
    if pickle_module is None:
        kwargs["weights_only"] = True
    else:
        kwargs.update(weights_only=False, pickle_module=pickle_module)
    return torch.load(path, **kwargs)
