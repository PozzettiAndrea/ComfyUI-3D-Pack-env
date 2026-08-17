# Runs before the ComfyUI server boots. Installs nothing -- reports environment
# health and applies the platform hygiene comfy-env needs.
from pathlib import Path

from comfy_env import setup_env, copy_files

setup_env()

SCRIPT_DIR = Path(__file__).resolve().parent
COMFYUI_DIR = SCRIPT_DIR.parent.parent

# Stage the example inputs into ComfyUI's input dir so the example workflows
# resolve their images and meshes without the user copying anything by hand.
# copy_files is non-recursive (src.glob("*"), files only), so the 3d subfolder
# needs its own call. It never overwrites an existing file.
copy_files(SCRIPT_DIR / "assets", COMFYUI_DIR / "input")
copy_files(SCRIPT_DIR / "assets" / "3d", COMFYUI_DIR / "input" / "3d")
