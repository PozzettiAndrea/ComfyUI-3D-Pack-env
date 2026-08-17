import torch.nn as nn
import torch.nn.functional as F

import comfy.ops

# Raw torch layers are not visible to ComfyUI's VRAM manager: ModelPatcher
# only lowvram-offloads modules carrying `comfy_cast_weights`, which every
# comfy.ops class has and no torch.nn class does.
ops = comfy.ops.manual_cast


class SdfMlp(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, bias=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.fc1 = ops.Linear(input_dim, hidden_dim, bias=bias)
        self.fc2 = ops.Linear(hidden_dim, hidden_dim, bias=bias)
        self.fc3 = ops.Linear(hidden_dim, 4, bias=bias)


    def forward(self, input):
        x = F.relu(self.fc1(input))
        x = F.relu(self.fc2(x))
        out = self.fc3(x)
        return out


class RgbMlp(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, bias=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.fc1 = ops.Linear(input_dim, hidden_dim, bias=bias)
        self.fc2 = ops.Linear(hidden_dim, hidden_dim, bias=bias)
        self.fc3 = ops.Linear(hidden_dim, 3, bias=bias)

    def forward(self, input):
        x = F.relu(self.fc1(input))
        x = F.relu(self.fc2(x))
        out = self.fc3(x)

        return out

    