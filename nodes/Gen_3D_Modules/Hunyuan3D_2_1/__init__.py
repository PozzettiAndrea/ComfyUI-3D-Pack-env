# __init__.py for Hunyuan3D-2.1
import os
import sys
import torch
import logging
import platform

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('ComfyUI-Hunyuan3D-2.1')

# hy3dshape / hy3dpaint are imported package-relatively; no sys.path setup.

# Apply torchvision fix before other imports
try:
    from .hy3dpaint.utils.torchvision_fix import apply_fix
    apply_fix()
    logger.info("Torchvision fix applied for Hunyuan3D-2.1")
except Exception as e:
    logger.error(f"Warning: Failed to apply torchvision fix: {e}")

# Import mmgp for memory management
try:
    from mmgp import offload, profile_type
    # mmgp replaces safetensors.safe_open on import with a signature that
    # predates the `backend` kwarg transformers 5.x passes; re-widen it.
    from ...shared_utils.mmgp_compat import keep_safe_open_compatible
    keep_safe_open_compatible()
    logger.info("mmgp available for memory management")
except ImportError:
    logger.warning("Warning: mmgp module not found")
except Exception as e:
    logger.error(f"Warning: Failed to import mmgp: {e}")

# Import key modules with aliases to avoid conflicts with old Hunyuan3D
from .hy3dshape.hy3dshape import FaceReducer as FaceReducer_2_1
from .hy3dshape.hy3dshape import FloaterRemover as FloaterRemover_2_1  
from .hy3dshape.hy3dshape import DegenerateFaceRemover as DegenerateFaceRemover_2_1
from .hy3dshape.hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline as Hunyuan3DDiTFlowMatchingPipeline_2_1
from .hy3dshape.hy3dshape.pipelines import export_to_trimesh as export_to_trimesh_2_1
from .hy3dshape.hy3dshape.rembg import BackgroundRemover as BackgroundRemover_2_1
from .hy3dpaint.textureGenPipeline import Hunyuan3DPaintPipeline as Hunyuan3DPaintPipeline_2_1
from .hy3dpaint.textureGenPipeline import Hunyuan3DPaintConfig as Hunyuan3DPaintConfig_2_1
from .hy3dpaint.convert_utils import create_glb_with_pbr_materials as create_glb_with_pbr_materials_2_1

logger.info("Hunyuan3D-2.1 modules loaded successfully")

# Export modules with safe aliases
__all__ = [
    'FaceReducer_2_1', 
    'FloaterRemover_2_1', 
    'DegenerateFaceRemover_2_1',
    'Hunyuan3DDiTFlowMatchingPipeline_2_1', 
    'export_to_trimesh_2_1',
    'BackgroundRemover_2_1',
    'Hunyuan3DPaintPipeline_2_1', 
    'Hunyuan3DPaintConfig_2_1',
    'create_glb_with_pbr_materials_2_1',
] 