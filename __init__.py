"""ComfyUI-3D-Pack-env -- comfy-env packaging of ComfyUI-3D-Pack.

Runs in the HOST ComfyUI process, so it must stay light: the only third-party
import here is comfy_env itself. All model code lives in nodes/ and is imported
inside the isolated environment by register_nodes().
"""

import logging
import os

from comfy_env import register_nodes

log = logging.getLogger("comfy3d")

ROOT_PATH = os.path.dirname(os.path.realpath(__file__))

# --- Host-side web server -------------------------------------------------
# webserver/server.py registers the /viewfile aiohttp route used by the 3D
# viewer in web/. It must run in the MAIN process (PromptServer lives here),
# not in the worker. Its config comes from Configs/system.conf -- parsed with
# pyhocon when available, otherwise defaulted, because the host environment is
# supposed to contain comfy-env and nothing else.
_DEFAULT_WEB_CONF = {
    "clients_ip": ["127.0.0.1", "0.0.0.0", "172.17.0.0", "172.17.0.1"],
}


def _load_system_conf():
    conf_path = os.path.join(ROOT_PATH, "Configs", "system.conf")
    try:
        from pyhocon import ConfigFactory  # optional in the host env
        with open(conf_path) as fh:
            conf = ConfigFactory.parse_string(fh.read())
        return conf["web"], conf["huggingface.token"]
    except Exception as e:
        log.info("[Comfy3D] using default web config (%s)", e)
        return _DEFAULT_WEB_CONF, ""


try:
    from .webserver.server import set_web_conf

    _web_conf, _hf_token = _load_system_conf()
    set_web_conf(_web_conf)
    if isinstance(_hf_token, str) and _hf_token:
        try:
            from huggingface_hub import login

            login(token=_hf_token)
        except Exception as e:
            log.warning("[Comfy3D] huggingface login skipped: %s", e)
except Exception as e:
    # A missing viewer route must never cost us the nodes.
    log.warning("[Comfy3D] web server route not registered: %s", e)

# --- Nodes ----------------------------------------------------------------
# Scans nodes/ for comfy-env.toml, materialises/uses the isolated env, runs a
# metadata scan inside it and synthesises proxy classes. The host process never
# imports torch, diffusers or any model code.
NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS = register_nodes()

WEB_DIRECTORY = "./web"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
