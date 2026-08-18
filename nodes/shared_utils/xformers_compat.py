"""A drop-in stand-in for `xformers.ops.memory_efficient_attention`.

This pack deliberately ships no xformers: it caps flash-attn at <=2.8.2 while
the cuda-wheels farm builds 2.8.3, and installing it took the whole pack to 0
registered nodes (see the note in nodes/comfy-env.toml).

Several vendored files still assume it. They all follow upstream's pattern of
importing xformers when `is_xformers_available()` says so and binding the name
to None otherwise -- and then calling `xformers.ops.memory_efficient_attention`
unguarded. So the absent-xformers branch is not a fallback; it is an
AttributeError ("'NoneType' object has no attribute 'ops'") the moment the
model runs. Seen on Era3D and Wonder3D; the same call exists in CharacterGen,
TriplaneGaussian, CRM, Hunyuan3D V1 and imagedream.

Those files now import `xformers` from here instead of binding None, which
leaves all 27 call sites untouched.

torch's scaled_dot_product_attention computes the same function and on torch
2.x dispatches to the same flash/mem-efficient kernels, so this is a routing
change, not a numerical one.
"""

import types

import torch
import torch.nn.functional as F


def memory_efficient_attention(query, key, value, attn_bias=None, p=0.0,
                               scale=None, *, op=None):
    """xformers' signature, computed with torch SDPA.

    Handles both layouts the callers use:
      * 4D (B, M, H, K) -- xformers' native layout, which is NOT SDPA's
        (B, H, M, K); the transpose is required or heads and tokens swap and
        the result is silently wrong rather than an error.
      * 3D (B*H, M, K) -- what diffusers' attn.head_to_batch_dim() produces,
        which SDPA already accepts as (N, L, E).

    `op` is accepted and ignored: it selects an xformers kernel, and there is
    no xformers here to select from.
    """
    if attn_bias is not None and not isinstance(attn_bias, torch.Tensor):
        # xformers accepts its own mask objects (BlockDiagonalMask, ...). None
        # of the call sites in this pack pass one; refuse rather than silently
        # dropping a mask and returning a plausible wrong answer.
        raise TypeError(
            f"xformers_compat.memory_efficient_attention got a non-tensor "
            f"attn_bias ({type(attn_bias).__name__}); only tensor masks and "
            f"None are supported.")

    if query.ndim == 4:
        q, k, v = (t.transpose(1, 2) for t in (query, key, value))
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_bias, dropout_p=p, scale=scale)
        return out.transpose(1, 2)

    return F.scaled_dot_product_attention(
        query, key, value, attn_mask=attn_bias, dropout_p=p, scale=scale)


# Shaped like the real package so `xformers.ops.memory_efficient_attention(...)`
# at the call sites keeps working untouched.
ops = types.SimpleNamespace(memory_efficient_attention=memory_efficient_attention)
xformers = types.SimpleNamespace(ops=ops)
