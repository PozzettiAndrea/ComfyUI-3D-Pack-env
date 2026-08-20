"""DINOv2 backbone, vendored from facebookresearch/dinov2.

TRELLIS used torch.hub.load('facebookresearch/dinov2', ...) to get this at
runtime, which meant every cold start depended on github.com AND on the GitHub
API call torch.hub makes to check the repo is not a fork -- a call that carries
$GITHUB_TOKEN when one is exported, so a revoked token turned a request that
succeeds anonymously into "HTTP Error 401: Unauthorized" and took the whole
pipeline down.

Only the inference subset is vendored (models/vision_transformer.py and
layers/), not the training or evaluation code. The weights are unchanged and
still come from dl.fbaipublicfiles.com, which needs no credentials and is not
GitHub.
"""
