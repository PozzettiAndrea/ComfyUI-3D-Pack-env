"""ComfyUI-3D-Pack-enved -- comfy-env packaging of ComfyUI-3D-Pack.

Runs in the HOST ComfyUI process, so it stays light: the only third-party
import here is comfy_env itself. All model code lives in nodes/ and is
imported inside the isolated environment by register_nodes().

Everything web-facing now lives together:
  javascript/     browser assets (three.js / gaussian-splat viewers), served
                  to the frontend via [tool.comfy] web in pyproject.toml
  this file       the single aiohttp route those viewers call, /viewfile
Upstream kept that route in a separate webserver/ package that reached into
nodes/ for a logger; it has been folded in here because it must run in the
host process, where nodes/ (and therefore torch) must never be imported.
"""

import logging
import os

from comfy_env import register_nodes

log = logging.getLogger("comfy3d")

ROOT_PATH = os.path.dirname(os.path.realpath(__file__))

SUPPORTED_VIEW_EXTENSIONS = (
    ".png", ".jpg", ".jpeg", ".mtl", ".obj", ".glb", ".ply", ".splat",
)

# Config comes from nodes/Configs/system.conf. pyhocon is OPTIONAL: the host env is
# supposed to hold comfy-env and nothing else, so fall back to the upstream
# defaults when it is absent.
_DEFAULT_CLIENTS_IP = ["127.0.0.1", "0.0.0.0", "172.17.0.0", "172.17.0.1"]


def _load_system_conf():
    conf_path = os.path.join(ROOT_PATH, "nodes", "Configs", "system.conf")
    try:
        from pyhocon import ConfigFactory

        with open(conf_path) as fh:
            conf = ConfigFactory.parse_string(fh.read())
        return list(conf["web"]["clients_ip"]), conf["huggingface.token"]
    except Exception as e:
        log.info("[Comfy3D] nodes/Configs/system.conf not parsed (%s); using defaults", e)
        return _DEFAULT_CLIENTS_IP, ""


def _register_viewfile_route(clients_ip):
    """GET /viewfile -- lets web/ viewers fetch a mesh off disk.

    Same allow-list check as upstream: only requests from a configured local
    address, only the supported 3D/image extensions, only files that exist.
    """
    import server

    web = server.web

    @server.PromptServer.instance.routes.get("/viewfile")
    async def view_file(request):
        query = request.rel_url.query
        if request.remote in clients_ip and "filepath" in query:
            filepath = query["filepath"]
            log.info("[Comfy3D] view_file: %s", filepath)
            if filepath.lower().endswith(SUPPORTED_VIEW_EXTENSIONS) and os.path.exists(filepath):
                return web.FileResponse(filepath)
        return web.Response(status=404)


try:
    _clients_ip, _hf_token = _load_system_conf()
    _register_viewfile_route(_clients_ip)
    if isinstance(_hf_token, str) and _hf_token:
        try:
            from huggingface_hub import login

            login(token=_hf_token)
        except Exception as e:
            log.warning("[Comfy3D] huggingface login skipped: %s", e)
except Exception as e:
    # A missing viewer route must never cost us the nodes.
    log.warning("[Comfy3D] /viewfile route not registered: %s", e)

# Scans nodes/ for comfy-env.toml, uses the isolated env, runs a metadata scan
# inside it and synthesises proxy classes. The host process never imports
# torch, diffusers or any model code.
NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS = register_nodes()

# Frontend assets are declared in pyproject.toml ([tool.comfy] web =
# "javascript"), served at /extensions/ComfyUI-3D-Pack-enved/. No WEB_DIRECTORY
# attribute -- it would double-register the dir under a second mount key.
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
