#__init__.py
import os
import sys
import torch
import logging
import platform
import folder_paths

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('ComfyUI-Hi3DGen')

# trellis_fork / stablex are imported package-relatively; no sys.path setup.

# Verify trellis package is importable
try:
    from . import trellis_fork
    # logger.info("Trellis package imported successfully")
except ImportError as e:
    logger.error(f"Failed to import trellis package: {e}")
    logger.error(f"Current sys.path: {sys.path}")
    raise

# Verify stablex package is importable
try:
    from . import stablex
    # logger.info("stablex package imported successfully")
except ImportError as e:
    logger.error(f"Failed to import stablex package: {e}")
    logger.error(f"Current sys.path: {sys.path}")
    raise

# Register model paths with ComfyUI
try:
    folder_paths.add_model_folder_path("trellis", os.path.join(folder_paths.models_dir, "trellis"))
    folder_paths.add_model_folder_path("checkpoints", os.path.join(folder_paths.models_dir, "checkpoints"))
except Exception as e:
    logger.error(f"Error registering model paths: {e}")

# Register model paths with ComfyUI
try:
    folder_paths.add_model_folder_path("stablex", os.path.join(folder_paths.models_dir, "stablex"))
except Exception as e:
    logger.error(f"Error registering model paths: {e}")

