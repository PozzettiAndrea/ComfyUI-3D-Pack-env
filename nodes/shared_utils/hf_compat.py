"""Shims for HuggingFace helpers the vendored model trees still import.

Four vendored DINOv2 encoders (StableFast3D, InstantMesh, TriplaneGaussian,
CharacterGen) were copied from transformers when
`find_pruneable_heads_and_indices` lived in `transformers.pytorch_utils`. It was
removed in transformers 5.x. Its only caller in each of those files is a
`prune_heads()` method that inference never reaches -- but the import sits at
module scope, so its absence took the whole pack down at import.

Reproduced here rather than pinning transformers back: the pack's own
`comfy-env.toml` leaves transformers unpinned on purpose, and one dead helper is
not a reason to hold the whole stack back.
"""

import torch


def find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
    """Find attention heads to prune and the surviving row indices.

    Verbatim behaviour of the transformers 4.x helper of the same name: returns
    the requested heads minus any already pruned, and a LongTensor indexing the
    rows of a (n_heads * head_size) weight that survive.
    """
    mask = torch.ones(n_heads, head_size)
    heads = set(heads) - already_pruned_heads
    for head in heads:
        # Each already-pruned head below this one shifts its position down.
        head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
        mask[head] = 0
    mask = mask.view(-1).contiguous().eq(1)
    index = torch.arange(len(mask))[mask].long()
    return heads, index
