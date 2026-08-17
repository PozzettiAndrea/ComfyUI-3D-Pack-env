import torch.nn as nn

import comfy.ops

# Raw torch layers are not visible to ComfyUI's VRAM manager: ModelPatcher
# only lowvram-offloads modules carrying `comfy_cast_weights`, which every
# comfy.ops class has and no torch.nn class does.
ops = comfy.ops.manual_cast

def init_linear(l, stddev):
    nn.init.normal_(l.weight, std=stddev)
    if l.bias is not None:
        nn.init.constant_(l.bias, 0.0)

class MLP(nn.Module):
    def __init__(self, *,
                 width: int,
                 init_scale: float):
        super().__init__()
        self.width = width
        self.c_fc = ops.Linear(width, width * 4)
        self.c_proj = ops.Linear(width * 4, width)
        self.gelu = nn.GELU()
        init_linear(self.c_fc, init_scale)
        init_linear(self.c_proj, init_scale)

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))
