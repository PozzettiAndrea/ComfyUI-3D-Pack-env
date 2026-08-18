from typing import *

BACKEND = 'xformers' 
DEBUG = False


def _first_available(preferred):
    """Fall back to an installed attention backend.

    BACKEND is a static default of 'xformers', and full_attn.py imports the
    chosen backend at module import -- so an uninstalled default is not a
    degraded path, it is ModuleNotFoundError before any node can run. This pack
    ships flash_attn and no xformers (xformers pinned flash-attn <=2.8.2 and
    could not be satisfied), which made every TRELLIS import fail.

    sdpa is the last resort and always exists on torch 2.x, where it dispatches
    to flash kernels anyway. 'naive' is never auto-selected: it is the slow
    reference path and picking it silently would look like a hang.
    """
    import importlib.util

    def installed(name):
        try:
            return importlib.util.find_spec(name) is not None
        except (ImportError, ValueError):
            return False

    if preferred in ('sdpa', 'naive'):
        return preferred
    if preferred == 'xformers' and installed('xformers'):
        return preferred
    if preferred == 'flash_attn' and installed('flash_attn'):
        return preferred
    for candidate in ('flash_attn', 'xformers'):
        if installed(candidate):
            print(f"[ATTENTION] {preferred!r} is not installed; using {candidate!r}")
            return candidate
    print(f"[ATTENTION] {preferred!r} is not installed; using 'sdpa'")
    return 'sdpa'


def __from_env():
    import os
    
    global BACKEND
    global DEBUG
    
    env_attn_backend = os.environ.get('ATTN_BACKEND')
    env_sttn_debug = os.environ.get('ATTN_DEBUG')
    
    if env_attn_backend is not None and env_attn_backend in ['xformers', 'flash_attn', 'sdpa', 'naive']:
        BACKEND = env_attn_backend
    if env_sttn_debug is not None:
        DEBUG = env_sttn_debug == '1'

    BACKEND = _first_available(BACKEND)

    print(f"[ATTENTION] Using backend: {BACKEND}")
        

__from_env()
    

def set_backend(backend: Literal['xformers', 'flash_attn']):
    global BACKEND
    BACKEND = backend

def set_debug(debug: bool):
    global DEBUG
    DEBUG = debug


from .full_attn import *
from .modules import *
