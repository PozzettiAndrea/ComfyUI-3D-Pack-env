import os
import gc
import math
import copy
import time
from enum import Enum
from collections import OrderedDict
import folder_paths as comfy_paths
from omegaconf import OmegaConf
import json

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2
import torchvision.transforms.functional as TF
import numpy as np
from safetensors.torch import load_file
from einops import rearrange

from diffusers import (
    DiffusionPipeline, 
    StableDiffusionPipeline
)

from diffusers import (
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    DDIMScheduler,
    DDIMParallelScheduler,
    LCMScheduler,
    KDPM2AncestralDiscreteScheduler,
    KDPM2DiscreteScheduler,
)


from plyfile import PlyData
import trimesh
from PIL import Image

from . import model_cache
from .mesh_processer.interop import wire_in as _wire_in, wire_out as _wire_out
from .mesh_processer.interop import _EXTRAS_KEY as _MESH_EXTRAS_KEY
from .shared_utils.model_downloader import (
    get_model_dir, download_file, download_files, download_repo, download_url,
)
from .mesh_processer.mesh import Mesh
from .mesh_processer.mesh_utils import (
    ply_to_points_cloud, 
    get_target_axis_and_scale, 
    switch_ply_axis_and_scale, 
    switch_mesh_axis_and_scale, 
    calculate_max_sh_degree_from_gs_ply,
    marching_cubes_density_to_mesh,
    color_func_to_albedo,
    interpolate_texture_map_attr,
    decimate_mesh,
)

from .MVs_Algorithms.FlexiCubes.flexicubes_trainer import FlexiCubesTrainer
from .MVs_Algorithms.DiffRastMesh.diff_mesh import DiffMesh, DiffMeshCameraController
from .MVs_Algorithms.DiffRastMesh.diff_mesh import DiffRastRenderer
from .MVs_Algorithms.GaussianSplatting.main_3DGS import GaussianSplatting3D, GaussianSplattingCameraController, GSParams
from .MVs_Algorithms.GaussianSplatting.main_3DGS_renderer import GaussianSplattingRenderer
from .MVs_Algorithms.NeRF.Instant_NGP import InstantNGP

from .Gen_3D_Modules.TriplaneGaussian.triplane_gaussian_transformers import TGS
from .Gen_3D_Modules.TriplaneGaussian.utils.config import ExperimentConfig as ExperimentConfigTGS, load_config as load_config_tgs
from .Gen_3D_Modules.TriplaneGaussian.data import CustomImageOrbitDataset
from .Gen_3D_Modules.TriplaneGaussian.utils.misc import todevice, get_device
from .Gen_3D_Modules.LGM.core.options import config_defaults
from .Gen_3D_Modules.LGM.mvdream.pipeline_mvdream import MVDreamPipeline
from .Gen_3D_Modules.LGM.large_multiview_gaussian_model import LargeMultiviewGaussianModel
from .Gen_3D_Modules.LGM.nerf_marching_cubes_converter import GSConverterNeRFMarchingCubes
from .Gen_3D_Modules.TripoSR.system import TSR
from .Gen_3D_Modules.StableFast3D.sf3d import utils as sf3d_utils
from .Gen_3D_Modules.StableFast3D.sf3d.system import SF3D
from .Gen_3D_Modules.InstantMesh.utils.camera_util import oribt_camera_poses_to_input_cameras
from .Gen_3D_Modules.CRM.model.crm.model import ConvolutionalReconstructionModel
from .Gen_3D_Modules.CRM.model.crm.sampler import CRMSampler
from .Gen_3D_Modules.Wonder3D.pipelines.pipeline_mvdiffusion_image import MVDiffusionImagePipeline
from .Gen_3D_Modules.Wonder3D.data.single_image_dataset import SingleImageDataset as MVSingleImageDataset
from .Gen_3D_Modules.Wonder3D.utils.misc import load_config as load_config_wonder3d
from .Gen_3D_Modules.Zero123Plus.pipeline import Zero123PlusPipeline
from .Gen_3D_Modules.Era3D.mvdiffusion.pipelines.pipeline_mvdiffusion_unclip import StableUnCLIPImg2ImgPipeline
from .Gen_3D_Modules.Era3D.mvdiffusion.data.single_image_dataset import SingleImageDataset as Era3DSingleImageDataset
from .Gen_3D_Modules.Era3D.utils.misc import load_config as load_config_era3d
from .Gen_3D_Modules.Unique3D.custum_3d_diffusion.custum_pipeline.unifield_pipeline_img2mvimg import StableDiffusionImage2MVCustomPipeline
from .Gen_3D_Modules.Unique3D.custum_3d_diffusion.custum_pipeline.unifield_pipeline_img2img import StableDiffusionImageCustomPipeline
from .Gen_3D_Modules.Unique3D.scripts.mesh_init import fast_geo
from .Gen_3D_Modules.Unique3D.scripts.utils import from_py3d_mesh, to_py3d_mesh, to_pyml_mesh, simple_clean_mesh
from .Gen_3D_Modules.Unique3D.scripts.project_mesh import multiview_color_projection, multiview_color_projection_texture, get_cameras_list, get_orbit_cameras_list
from .Gen_3D_Modules.Unique3D.mesh_reconstruction.recon import reconstruct_stage1
from .Gen_3D_Modules.Unique3D.mesh_reconstruction.refine import run_mesh_refine
from .Gen_3D_Modules.CharacterGen.character_inference import Inference2D_API, Inference3D_API
from .Gen_3D_Modules.CharacterGen.Stage_3D.lrm.utils.config import load_config as load_config_cg3d
from .Gen_3D_Modules import craftsman
from .Gen_3D_Modules.craftsman.systems.base import BaseSystem
from .Gen_3D_Modules.craftsman.utils.config import ExperimentConfig as ExperimentConfigCraftsman, load_config as load_config_craftsman
from .Gen_3D_Modules.CRM_T2I_V2.model.crm.sampler import CRMSamplerV2
from .Gen_3D_Modules.CRM_T2I_V2.model.t2i_adapter_v2 import T2IAdapterV2
from .Gen_3D_Modules.CRM_T2I_V3.model.crm.sampler import CRMSamplerV3
from .Gen_3D_Modules.Hunyuan3D_V1.mvd.hunyuan3d_mvd_std_pipeline import HunYuan3D_MVD_Std_Pipeline
from .Gen_3D_Modules.Hunyuan3D_V1.mvd.hunyuan3d_mvd_lite_pipeline import Hunyuan3D_MVD_Lite_Pipeline
from .Gen_3D_Modules.Hunyuan3D_V1.mvd.hunyuan3d_mvd_std_pipeline import HunYuan3D_MVD_Std_Pipeline
from .Gen_3D_Modules.Hunyuan3D_V1.infer import Views2Mesh
from .Gen_3D_Modules.Hunyuan3D_V2.hy3dgen.shapegen import FaceReducer, FloaterRemover, DegenerateFaceRemover, Hunyuan3DDiTFlowMatchingPipeline
from .Gen_3D_Modules.Hunyuan3D_V2.hy3dgen.texgen import Hunyuan3DPaintPipeline
from .Gen_3D_Modules.Hunyuan3D_V2.hy3dgen.rembg import BackgroundRemover
from .Gen_3D_Modules.TRELLIS.trellis.pipelines import TrellisImageTo3DPipeline
from .Gen_3D_Modules.TRELLIS.trellis.utils import postprocessing_utils
from .Gen_3D_Modules.TripoSG.pipelines.pipeline_triposg import TripoSGPipeline
from .Gen_3D_Modules.TripoSG.pipelines.pipeline_triposg_scribble import TripoSGScribblePipeline
from .Gen_3D_Modules.Stable3DGen.pipeline_builders import StableGenPipelineBuilder
from .Gen_3D_Modules.MV_Adapter.mvadapter_node_utils import prepare_pipeline as mvadapter_prepare_pipeline, run_pipeline as mvadapter_run_pipeline, prepare_tg2mv_pipeline as mvadapter_prepare_tg2mv_pipeline, run_tg2mv_pipeline as mvadapter_run_tg2mv_pipeline, prepare_texture_pipeline as mvadapter_prepare_texture_pipeline, download_texture_checkpoints
from .Gen_3D_Modules.Hunyuan3D_2_1 import FaceReducer_2_1, Hunyuan3DDiTFlowMatchingPipeline_2_1, export_to_trimesh_2_1, BackgroundRemover_2_1, Hunyuan3DPaintPipeline_2_1, Hunyuan3DPaintConfig_2_1, create_glb_with_pbr_materials_2_1
from .Gen_3D_Modules.Hunyuan3D_2_1.hy3dpaint.utils.torchvision_fix import apply_fix
apply_fix()
from .Gen_3D_Modules.PartCrafter.partcrafter_src.pipelines.pipeline_partcrafter import PartCrafterPipeline
from .Gen_3D_Modules.PartCrafter.partcrafter_src.utils.data_utils import get_colored_mesh_composition
import zipfile


os.environ['SPCONV_ALGO'] = 'native'

from .shared_utils.image_utils import (
    prepare_torch_img, torch_imgs_to_pils, troch_image_dilate, 
    pils_rgba_to_rgb, pil_make_image_grid, pil_split_image, pils_to_torch_imgs, pils_resize_foreground
)
from .shared_utils.camera_utils import (
    ORBITPOSE_PRESET_DICT, ELEVATION_MIN, ELEVATION_MAX, AZIMUTH_MIN, AZIMUTH_MAX, 
    compose_orbit_camposes
)
from .shared_utils.log_utils import cstr
from . import _vendor_paths


def _comfy_model_dir(folder):
    """ComfyUI's own directory for `folder` (checkpoints, clip_vision, ...)."""
    try:
        path = comfy_paths.get_folder_paths(folder)[0]
    except Exception:
        path = os.path.join(comfy_paths.models_dir, folder)
    os.makedirs(path, exist_ok=True)
    return path


#: Mesh formats a texgen/projection node can open, matching ComfyUI's own
#: Load3D node so the two agree on what "a 3D file" is.
#:
#: mesh_path stays a STRING rather than becoming a dropdown of input/3d: the
#: shipped workflows LINK it from another node's STRING output (Save 3D Mesh,
#: ShapeGen), and a COMBO input cannot accept a STRING link -- making it a
#: combo fails validation with "received_type(STRING) mismatch input_type".
#: Resolving the string against ComfyUI's input dir gets the same result
#: without breaking those links.
_MESH_INPUT_EXTS = [".glb", ".gltf", ".obj", ".fbx", ".stl", ".ply"]
_NO_MESH_PLACEHOLDER = "(no 3D files found in input/3d)"


def _mesh_input_combo():
    """A dropdown of the 3D files in ComfyUI's input folder.

    The option list here is only a placeholder. Our INPUT_TYPES is captured
    once by the isolation metadata scan and cached, so a static listing would
    never show a file uploaded after startup; the comfy_env_* markers opt the
    combo into comfy-env's live rescan, which runs in the HOST process on every
    /object_info and replaces the options wholesale.

    Two sources, mirroring native Load3D: input/3d recursively (values relative
    to the input root, e.g. "3d/foo.glb") and the input root itself.
    """
    return (
        [_NO_MESH_PLACEHOLDER],
        {
            "comfy_env_dynamic_dir": "3d",
            "comfy_env_sources": [
                {"dir": "3d", "recursive": True, "rel_to_input": True},
                {"dir": "", "recursive": False, "rel_to_input": False},
            ],
            "comfy_env_exts": _MESH_INPUT_EXTS,
            "comfy_env_placeholder": _NO_MESH_PLACEHOLDER,
        },
    )


def _instantmesh_safetensors(ckpt_path):
    """Path of the .safetensors twin of an InstantMesh .ckpt, converting once.

    TencentARC/InstantMesh publishes only .ckpt -- torch pickle archives, which
    torch.load has to unpickle (arbitrary code, and slow: the file carries the
    whole training checkpoint, of which we keep only the `lrm_generator.`
    weights). There is no safetensors upstream to download, so make one: read
    the pickle once, strip it to the weights this model actually loads, and
    write the twin next to it. Every later load reads that instead.

    Returns the .ckpt path unchanged if conversion is not possible, so a
    failure here costs speed, never the node.
    """
    st_path = os.path.splitext(ckpt_path)[0] + ".safetensors"
    if os.path.isfile(st_path):
        return st_path
    if not os.path.isfile(ckpt_path):
        return ckpt_path
    try:
        from safetensors.torch import save_file

        state_dict = _instantmesh_state_dict(ckpt_path, _allow_safetensors=False)
        # contiguous(): safetensors refuses views into a shared buffer, which
        # is how torch.load hands back slices of a checkpoint.
        save_file({k: v.contiguous() for k, v in state_dict.items()}, st_path)
        cstr(f"[InstantMesh] converted {os.path.basename(ckpt_path)} -> "
             f"{os.path.basename(st_path)} ({len(state_dict)} tensors); "
             f"later loads skip the pickle").msg.print()
        return st_path
    except Exception as e:
        cstr(f"[InstantMesh] safetensors conversion skipped ({type(e).__name__}: {e}); "
             f"loading the .ckpt").msg.print()
        return ckpt_path


def _instantmesh_state_dict(ckpt_path, _allow_safetensors=True):
    """The `lrm_generator.` weights, from the safetensors twin when there is one."""
    st_path = os.path.splitext(ckpt_path)[0] + ".safetensors"
    if _allow_safetensors and os.path.isfile(st_path):
        from safetensors.torch import load_file

        return load_file(st_path, device="cpu")

    # weights_only=True: this is a pickle from the internet and we want tensors
    # from it, nothing else. torch>=2.6 defaults to it; be explicit so the
    # intent survives a downgrade.
    obj = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state_dict = obj["state_dict"] if "state_dict" in obj else obj
    return {k[14:]: v for k, v in state_dict.items() if k.startswith("lrm_generator.")}


def _resolve_input_mesh(mesh_path, mesh_file=None):
    """Resolve the chosen mesh to a file on disk.

    Two ways in, because both are needed. `mesh_file` is the dropdown of
    input/3d -- the convenient one. `mesh_path` is an optional STRING socket
    and takes precedence when set, because the shipped workflows LINK it from
    another node's STRING output (Save 3D Mesh, ShapeGen), and a COMBO input
    cannot accept a STRING link -- offering only the dropdown fails validation
    with "received_type(STRING) mismatch input_type".

    Either way the value may be relative to ComfyUI's input dir, absolute, or
    relative to the cwd.
    """
    if isinstance(mesh_path, str) and mesh_path.strip():
        chosen = mesh_path.strip()
    else:
        chosen = mesh_file

    mesh_path = chosen
    if not mesh_path or mesh_path == _NO_MESH_PLACEHOLDER:
        raise Exception(
            "No mesh selected. Put a mesh (%s) in ComfyUI's input/3d folder, "
            "or upload one, then pick it in mesh_path."
            % "/".join(e.lstrip(".") for e in _MESH_INPUT_EXTS))

    input_dir = comfy_paths.get_input_directory()
    for candidate in (mesh_path, os.path.join(input_dir, mesh_path)):
        if os.path.isfile(candidate):
            return candidate

    raise Exception(
        f"Mesh file not found: {mesh_path!r}. Looked in ComfyUI's input dir "
        f"({input_dir}) and relative to {os.getcwd()}.")


def _ensure_comfy_model(folder, filename, *, url=None, repo_id=None, repo_file=None):
    """Fetch a model into ComfyUI's OWN model folder and return its bare name.

    Deliberately not models/3d-pack/: the point is for a NATIVE loader
    (CheckpointLoaderSimple, CLIPVisionLoader, UpscaleModelLoader) to resolve
    it, and those look the name up with folder_paths.get_full_path(folder,
    name). Returning the bare filename -- not a path -- is what lets the value
    feed such a loader's combo widget.

    Wiring that output into the loader also ORDERS the graph: with no link,
    ComfyUI is free to run the loader before this node has downloaded anything,
    and the loader fails on a file that is about to exist.

    Already-present files are never re-fetched, so this is safe to call on
    every execution.
    """
    target_dir = _comfy_model_dir(folder)
    dest = os.path.join(target_dir, filename)
    if os.path.isfile(dest):
        return filename

    cstr(f"[Comfy3D] fetching {folder}/{filename} ...").msg.print()
    if url is not None:
        download_url(url, filename, target_dir)
    else:
        src = download_file(repo_id, repo_file or filename, target_dir)
        # download_file preserves the repo's own subfolder layout; the native
        # loaders only look in the folder root, so put it where they look.
        if os.path.abspath(src) != os.path.abspath(dest):
            os.replace(src, dest)
            # and remove the now-empty repo subdirs it left behind
            parent = os.path.dirname(src)
            while os.path.normpath(parent) != os.path.normpath(target_dir):
                try:
                    os.rmdir(parent)
                except OSError:
                    break
                parent = os.path.dirname(parent)
    return filename


def _ensure_upscale_model(url, filename):
    """Back-compat wrapper: an upscaler into ComfyUI's upscale_models/."""
    return _ensure_comfy_model("upscale_models", filename, url=url)


def _alias_model_index_libraries(base_dir):
    """Make a model_index.json's vendored `library` names importable.

    diffusers resolves each sub-model's library with a bare
    importlib.import_module (pipeline_loading_utils.py:459), which never sees
    _vendor_paths.resolve(). Reading the names out of the file rather than
    hard-coding them keeps this correct if the index changes.
    """
    index_path = os.path.join(base_dir, "model_index.json")
    try:
        with open(index_path) as fh:
            index = json.load(fh)
    except (OSError, ValueError) as e:
        cstr(f"[Comfy3D] could not read {index_path}: {e}").warning.print()
        return
    libraries = [
        v[0] for v in index.values()
        if isinstance(v, list) and v and isinstance(v[0], str)
    ]
    aliased = _vendor_paths.alias_modules(*libraries)
    if aliased:
        cstr(f"[Comfy3D] aliased vendored modules: {', '.join(aliased)}").msg.print()


def _comfy_progress_bar(total):
    """comfy.utils.ProgressBar(total), or None when it is unavailable.

    Same guard as shared_utils/model_downloader.py: this code runs inside the
    isolated worker env, where ComfyUI is importable in practice but is not a
    declared dependency of it. A missing progress bar must never cost us the
    node, so callers treat None as "just skip the UI bar".
    """
    try:
        import comfy.utils
    except ImportError:
        return None
    return comfy.utils.ProgressBar(total)

from .shared_utils.common_utils import parse_save_filename, get_list_filenames, resume_or_download_model_from_hf

DIFFUSERS_PIPE_DICT = OrderedDict([
    ("MVDreamPipeline", MVDreamPipeline),
    ("Wonder3DMVDiffusionPipeline", MVDiffusionImagePipeline),
    ("Zero123PlusPipeline", Zero123PlusPipeline),
    ("DiffusionPipeline", DiffusionPipeline),
    ("StableDiffusionPipeline", StableDiffusionPipeline),
    ("Era3DPipeline", StableUnCLIPImg2ImgPipeline),
    ("Unique3DImage2MVCustomPipeline", StableDiffusionImage2MVCustomPipeline),
    ("Unique3DImageCustomPipeline", StableDiffusionImageCustomPipeline),
    ("HunYuan3DMVDStdPipeline", HunYuan3D_MVD_Std_Pipeline),
    ("Hunyuan3DMVDLitePipeline", Hunyuan3D_MVD_Lite_Pipeline),
    ("Hunyuan3DMVDStdPipeline", HunYuan3D_MVD_Std_Pipeline),
    ("Hunyuan3DDiTFlowMatchingPipeline", Hunyuan3DDiTFlowMatchingPipeline),
    ("Hunyuan3DPaintPipeline", Hunyuan3DPaintPipeline),
    ("TripoSGPipeline", TripoSGPipeline),
    ("TripoSGScribblePipeline", TripoSGScribblePipeline),
])

DIFFUSERS_SCHEDULER_DICT = OrderedDict([
    ("EulerAncestralDiscreteScheduler", EulerAncestralDiscreteScheduler),
    ("Wonder3DMVDiffusionPipeline", MVDiffusionImagePipeline),
    ("EulerDiscreteScheduler,", EulerDiscreteScheduler),
    ("DDIMScheduler,", DDIMScheduler),
    ("DDIMParallelScheduler,", DDIMParallelScheduler),
    ("LCMScheduler,", LCMScheduler),
    ("KDPM2AncestralDiscreteScheduler,", KDPM2AncestralDiscreteScheduler),
    ("KDPM2DiscreteScheduler,", KDPM2DiscreteScheduler),
])

NODES_PATH = os.path.dirname(os.path.realpath(__file__))
# nodes/ sits one level below the pack root. model_configs/, Configs/,
# Gen_3D_Modules/ and MVs_Algorithms/ all stay at the root, so ROOT_PATH keeps
# its original meaning and every path derived from it below is unchanged.
ROOT_PATH = os.path.dirname(NODES_PATH)

# nodes/model_configs/ holds the 231 git-tracked *.json / *.yaml configs and the
# custom_pipeline code that from_pretrained needs -- named for what it actually
# contains rather than "Checkpoints", which it never only held.
#
# Downloaded weights still land here too, which is the part that is not yet
# right. Moving them to ComfyUI's models/ tree needs the download policy fixed
# first: HF_DOWNLOAD_IGNORE below excludes *.json/*.yaml/*.py from the snapshot,
# so those configs come from git, out of this same directory. Repoint this alone
# and the downloads land next to no model_index.json and fail.
CKPT_ROOT_PATH = os.path.join(NODES_PATH, "model_configs")
CKPT_DIFFUSERS_PATH = os.path.join(CKPT_ROOT_PATH, "Diffusers")
CONFIG_ROOT_PATH = os.path.join(NODES_PATH, "Configs")
MODULE_ROOT_PATH = os.path.join(NODES_PATH, "Gen_3D_Modules")

MANIFEST = {
    "name": "ComfyUI-3D-Pack",
    "version": (0,0,2),
    "author": "Mr. For Example",
    "project": "https://github.com/MrForExample/ComfyUI-3D-Pack",
    "description": "An extensive node suite that enables ComfyUI to process 3D inputs (Mesh & UV Texture, etc) using cutting edge algorithms (3DGS, NeRF, etc.)",
}

SUPPORTED_3D_EXTENSIONS = (
    '.obj',
    '.ply',
    '.glb',
)

SUPPORTED_3DGS_EXTENSIONS = (
    '.ply',
)

SUPPORTED_CHECKPOINTS_EXTENSIONS = (
    '.ckpt', 
    '.bin', 
    '.safetensors',
)

WEIGHT_DTYPE = torch.float16


def get_device():
    """The device ComfyUI wants us on, resolved on every call.

    This used to be a module-level constant:

        DEVICE_STR = "cuda" if torch.cuda.is_available() else "cpu"
        DEVICE = torch.device(DEVICE_STR)

    which is a private reimplementation of get_torch_device() that gets the
    answer wrong. It ignores --cpu, --gpu-only and --directml, never returns
    mps/xpu/npu/mlu, and cannot follow a device change because it is frozen at
    import. No sibling pack in this tree defines such a constant; they all call
    get_torch_device() at the point of use (e.g. TRELLIS2 stages.py:261,
    DepthAnythingV3 nodes_inference.py:80, SAM3 _model_cache.py:38).
    """
    import comfy.model_management

    return comfy.model_management.get_torch_device()


def get_device_str():
    """get_device() as a string, for third-party APIs that want one."""
    return str(get_device())


def get_autocast_device_type():
    """The device TYPE for torch.autocast -- "cuda", not "cuda:0".

    get_device_str() now returns an indexed device, which torch.autocast happens
    to accept today but is not what it documents. Native has a helper for exactly
    this (model_management.get_autocast_device, :1275), which returns dev.type.
    """
    import comfy.model_management

    return comfy.model_management.get_autocast_device(get_device())

# Excluded from every snapshot_download. The config/code entries are the point:
# *.yaml/*.json/*.py are pinned in git under model_configs/ and reviewed there,
# so a download must not overwrite them with whatever upstream currently serves.
#
# The image entries used to read ".png"/".jpg"/".gif" -- snapshot_download
# matches with fnmatch, so those only ever matched a file *named* ".png", and
# every sample image in every repo was being fetched.
#
# .gitattributes is HuggingFace's LFS config for their repo. Landing it in this
# tree tells git to LFS-ify every *.ckpt/*.safetensors/*.bin underneath it,
# which is exactly where downloaded weights go.
HF_DOWNLOAD_IGNORE = [
    "*.yaml", "*.json", "*.py",
    "*.png", "*.jpg", "*.jpeg", "*.gif", "*.webp",
    # Repo furniture, not weights. These are git-tracked here, and a download
    # was silently rewriting them: fetching tencent/Hunyuan3D-1 replaced this
    # pack's committed LICENSE.txt, Notice and README.md with whatever upstream
    # serves today. .gitattributes is worse than cosmetic -- it carries HF's
    # LFS rules for *.ckpt/*.safetensors/*.bin, into the tree those land in.
    ".gitattributes", "*.md", "*.txt", "Notice", "LICENSE",
]


class Preview_3DGS:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "gs_file_path": ("STRING", {"default": '', "multiline": False}),
            },
        }
    
    OUTPUT_NODE = True
    RETURN_TYPES = ()
    FUNCTION = "preview_gs"
    CATEGORY = "Comfy3D/Visualize"
    
    def preview_gs(self, gs_file_path):
        
        gs_folder_path, filename = os.path.split(gs_file_path)
        
        if not os.path.isabs(gs_file_path):
            gs_file_path = os.path.join(comfy_paths.output_directory, gs_folder_path)
        
        if not filename.lower().endswith(SUPPORTED_3DGS_EXTENSIONS):
            cstr(f"[{self.__class__.__name__}] File name {filename} does not end with supported 3DGS file extensions: {SUPPORTED_3DGS_EXTENSIONS}").error.print()
            gs_file_path = ""
        
        previews = [
            {
                "filepath": gs_file_path,
            }
        ]
        return {"ui": {"previews": previews}, "result": ()}
    
class Preview_3DMesh:

    @classmethod
    def INPUT_TYPES(cls):
        # Mesh in, nothing else. The node used to require a STRING path and take
        # the mesh optionally, so every graph ran the mesh through Save 3D Mesh
        # purely to get a path to hand back -- 69 links across the shipped
        # workflows did exactly that, and not one used the path widget.
        return {
            "required": {
                "mesh": ("TRIMESH",),
            },
        }

    OUTPUT_NODE = True
    # Pass the mesh through so the preview drops inline rather than ending a
    # branch. An unconnected `mesh` yields None, which downstream ignores.
    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("mesh",)
    FUNCTION = "preview_mesh"
    CATEGORY = "Comfy3D/Visualize"

    def preview_mesh(self, mesh):

        if mesh is not None:
            # .glb because threeVisualizer already has GLTFLoader. Written under
            # output/ so ComfyUI's own /view can serve it: /view needs no
            # client-IP allow-list, unlike this pack's /viewfile, which 404s for
            # any browser that is not on the ComfyUI host.
            # _wire_in, and Mesh.write rather than trimesh.export: the texture
            # only survives that way. A painted mesh carries a UV *atlas*, so
            # ft != f, and to_trimesh deliberately refuses to put a mismatched
            # uv on the wire Trimesh -- the albedo rides in extras as a tensor
            # instead. trimesh.export() therefore has nothing to bake and the
            # preview comes out grey. Mesh.write_glb already handles exactly
            # this: align_v_to_vt() expands the atlas to per-vertex UV, then
            # the albedo goes into the GLB buffer as a baseColorTexture.
            m = _wire_in(mesh)
            mesh_file_path = parse_save_filename(
                os.path.join("Preview3DMesh", "preview_%Y-%m-%d-%H%M%S%f.glb"),
                comfy_paths.output_directory,
                SUPPORTED_3D_EXTENSIONS,
                self.__class__.__name__,
            )
            if mesh_file_path is not None:
                m.write(mesh_file_path)
            else:
                mesh_file_path = ""

        else:
            mesh_file_path = ""

        # No extension/abspath validation left to do: this node now writes the
        # only path it previews, so the checks that guarded a hand-typed string
        # can no longer fail.
        cstr(f"[{self.__class__.__name__}] preview -> {mesh_file_path or '(no mesh)'}").msg.print()

        previews = [
            {
                "filepath": mesh_file_path,
            }
        ]
        return {"ui": {"previews": previews}, "result": (mesh,)}

class Load_3D_Mesh:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh_file_path": ("STRING", {"default": '', "multiline": False}),
                "resize":  ("BOOLEAN", {"default": False},),
                "renormal":  ("BOOLEAN", {"default": True},),
                "retex":  ("BOOLEAN", {"default": False},),
                "optimizable": ("BOOLEAN", {"default": False},),
                "clean": ("BOOLEAN", {"default": False},),
                "resize_bound": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1000.0, "step": 0.001}),
            },
        }

    RETURN_TYPES = (
        "TRIMESH",
    )
    RETURN_NAMES = (
        "mesh",
    )
    FUNCTION = "load_mesh"
    CATEGORY = "Comfy3D/Import|Export"
    
    def load_mesh(self, mesh_file_path, resize, renormal, retex, optimizable, clean, resize_bound):
        mesh = None
        
        if not os.path.isabs(mesh_file_path):
            mesh_file_path = os.path.join(comfy_paths.input_directory, mesh_file_path)
        
        # Raise, do not return None on a TRIMESH socket.
        #
        # These branches used to print an error and hand None downstream, so a
        # missing file surfaced several nodes later as
        #   AttributeError: 'NoneType' object has no attribute 'v'
        # inside the rasteriser, naming neither the file nor this node. The
        # shipped workflows point at paths from upstream's Windows layout, so
        # this is the common case, not an edge one.
        if not os.path.exists(mesh_file_path):
            raise FileNotFoundError(
                f"[{self.__class__.__name__}] mesh not found: {mesh_file_path}. "
                f"Paths are taken relative to ComfyUI's input directory "
                f"({comfy_paths.input_directory}) unless absolute.")

        folder, filename = os.path.split(mesh_file_path)
        if not filename.lower().endswith(SUPPORTED_3D_EXTENSIONS):
            raise ValueError(
                f"[{self.__class__.__name__}] {filename} is not a supported 3D "
                f"file; expected one of {SUPPORTED_3D_EXTENSIONS}.")

        with torch.inference_mode(not optimizable):
            mesh = Mesh.load(mesh_file_path, resize, renormal, retex, clean, resize_bound)
        return (_wire_out(mesh), )
    
class Load_3DGS:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "gs_file_path": ("STRING", {"default": '', "multiline": False}),
            },
        }

    RETURN_TYPES = (
        "GS_PLY",
    )
    RETURN_NAMES = (
        "gs_ply",
    )
    FUNCTION = "load_gs"
    CATEGORY = "Comfy3D/Import|Export"
    
    def load_gs(self, gs_file_path):
        gs_ply = None
        
        if not os.path.isabs(gs_file_path):
            gs_file_path = os.path.join(comfy_paths.input_directory, gs_file_path)
        
        # Raise rather than return None, for the same reason as Load_3D_Mesh:
        # a None on a typed socket fails later, somewhere less informative.
        if not os.path.exists(gs_file_path):
            raise FileNotFoundError(
                f"[{self.__class__.__name__}] 3DGS file not found: {gs_file_path}. "
                f"Paths are taken relative to ComfyUI's input directory "
                f"({comfy_paths.input_directory}) unless absolute.")

        folder, filename = os.path.split(gs_file_path)
        if not filename.lower().endswith(SUPPORTED_3DGS_EXTENSIONS):
            raise ValueError(
                f"[{self.__class__.__name__}] {filename} is not a supported 3DGS "
                f"file; expected one of {SUPPORTED_3DGS_EXTENSIONS}.")

        gs_ply = PlyData.read(gs_file_path)
        return (gs_ply, )
    
class Save_3D_Mesh:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("TRIMESH",),
                "save_path": ("STRING", {"default": 'Mesh_%Y-%m-%d-%M-%S-%f.glb', "multiline": False}),
            },
        }

    OUTPUT_NODE = True
    RETURN_TYPES = (
        "STRING",
    )
    RETURN_NAMES = (
        "save_path",
    )
    FUNCTION = "save_mesh"
    CATEGORY = "Comfy3D/Import|Export"
    
    def save_mesh(self, mesh, save_path):
        mesh = _wire_in(mesh)
        save_path = parse_save_filename(save_path, comfy_paths.output_directory, SUPPORTED_3D_EXTENSIONS, self.__class__.__name__)
        
        if save_path is not None:
            mesh.write(save_path)

        return (save_path, )
    
class Save_3DGS:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "gs_ply": ("GS_PLY",),
                "save_path": ("STRING", {"default": '3DGS_%Y-%m-%d-%M-%S-%f.ply', "multiline": False}),
            },
        }

    OUTPUT_NODE = True
    RETURN_TYPES = (
        "STRING",
    )
    RETURN_NAMES = (
        "save_path",
    )
    FUNCTION = "save_gs"
    CATEGORY = "Comfy3D/Import|Export"
    
    def save_gs(self, gs_ply, save_path):
        
        save_path = parse_save_filename(save_path, comfy_paths.output_directory, SUPPORTED_3DGS_EXTENSIONS, self.__class__.__name__)
        
        if save_path is not None:
            gs_ply.write(save_path)
        
        return (save_path, )

class Image_Add_Pure_Color_Background:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "masks": ("MASK",),
                "R": ("INT", {"default": 255, "min": 0, "max": 255}),
                "G": ("INT", {"default": 255, "min": 0, "max": 255}),
                "B": ("INT", {"default": 255, "min": 0, "max": 255}),
            },
        }
        
    RETURN_TYPES = (
        "IMAGE",
    )
    RETURN_NAMES = (
        "images",
    )
    
    FUNCTION = "image_add_bg"
    CATEGORY = "Comfy3D/Preprocessor"

    def image_add_bg(self, images, masks, R, G, B):
        """
        bg_mask = bg_mask.unsqueeze(3)
        inv_bg_mask = torch.ones_like(bg_mask) - bg_mask
        color = torch.tensor([R, G, B]).to(image.dtype) / 255
        color_bg = color.repeat(bg_mask.shape)
        image = inv_bg_mask * image + bg_mask * color_bg
        """

        image_pils = torch_imgs_to_pils(images, masks)
        image_pils = pils_rgba_to_rgb(image_pils, (R, G, B))

        images = pils_to_torch_imgs(image_pils, images.dtype, images.device)
        return (images,)
    
class Resize_Image_Foreground:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "masks": ("MASK",),
                "foreground_ratio": ("FLOAT", {"default": 0.85, "min": 0.01, "max": 1.0, "step": 0.01}),
            },
        }
        
    RETURN_TYPES = (
        "IMAGE",
        "MASK",
    )
    RETURN_NAMES = (
        "images",
        "masks",
    )
    
    FUNCTION = "resize_img_foreground"
    CATEGORY = "Comfy3D/Preprocessor"

    def resize_img_foreground(self, images, masks, foreground_ratio):
        image_pils = torch_imgs_to_pils(images, masks)
        image_pils = pils_resize_foreground(image_pils, foreground_ratio)
        
        images = pils_to_torch_imgs(image_pils, images.dtype, images.device, force_rgb=False)
        images, masks = images[:, :, :, 0:-1], images[:, :, :, -1]
        return (images, masks,)
    
class Make_Image_Grid:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "grid_side_num": ("INT", {"default": 1, "min": 1, "max": 8192}),
                "use_rows": ("BOOLEAN", {"default": True},),
            },
        }
        
    RETURN_TYPES = (
        "IMAGE",
    )
    RETURN_NAMES = (
        "image_grid",
    )
    
    FUNCTION = "make_image_grid"
    CATEGORY = "Comfy3D/Preprocessor"
    
    def make_image_grid(self, images, grid_side_num, use_rows):
        pil_image_list = torch_imgs_to_pils(images)

        if use_rows:
            rows = grid_side_num
            clos = None
        else:
            clos = grid_side_num
            rows = None

        image_grid = pil_make_image_grid(pil_image_list, rows, clos)

        image_grid = TF.to_tensor(image_grid).permute(1, 2, 0).unsqueeze(0)  # [1, H, W, 3]

        return (image_grid,)

class Split_Image_Grid:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "grid_side_num": ("INT", {"default": 1, "min": 1, "max": 8192}),
                "use_rows": ("BOOLEAN", {"default": True},),
            },
        }
        
    RETURN_TYPES = (
        "IMAGE",
    )
    RETURN_NAMES = (
        "images",
    )
    
    FUNCTION = "split_image_grid"
    CATEGORY = "Comfy3D/Preprocessor"
    
    def split_image_grid(self, image, grid_side_num, use_rows):
        images = []
        for image_pil in torch_imgs_to_pils(image):

            if use_rows:
                rows = grid_side_num
                clos = None
            else:
                clos = grid_side_num
                rows = None

            image_pils = pil_split_image(image_pil, rows, clos)

            images.append(pils_to_torch_imgs(image_pils, image.dtype, image.device))
            
        images = torch.cat(images, dim=0)
        return (images,)

class Get_Masks_From_Normal_Maps:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "normal_maps": ("IMAGE",),
            },
        }
        
    RETURN_TYPES = (
        "MASK",
    )
    RETURN_NAMES = (
        "normal_masks",
    )
    
    FUNCTION = "make_image_grid"
    CATEGORY = "Comfy3D/Preprocessor"
    
    def make_image_grid(self, normal_maps):
        from .Gen_3D_Modules.Unique3D.scripts.utils import get_normal_map_masks
        pil_normal_list = torch_imgs_to_pils(normal_maps)
        normal_masks = get_normal_map_masks(pil_normal_list)
        normal_masks = torch.stack(normal_masks, dim=0).to(normal_maps.dtype).to(normal_maps.device)
        return (normal_masks,)

class Rotate_Normal_Maps_Horizontally:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "normal_maps": ("IMAGE",),
                "normal_masks": ("MASK",),
                "clockwise": ("BOOLEAN", {"default": True},),
            },
        }
        
    RETURN_TYPES = (
        "IMAGE",
    )
    RETURN_NAMES = (
        "normal_maps",
    )
    
    FUNCTION = "make_image_grid"
    CATEGORY = "Comfy3D/Preprocessor"
    
    def make_image_grid(self, normal_maps, normal_masks, clockwise):
        rotate_direction = 1 if clockwise is True else -1
        if normal_maps.shape[0] > 1:
            from .Gen_3D_Modules.Unique3D.scripts.utils import rotate_normals_torch
            pil_image_list = torch_imgs_to_pils(normal_maps, normal_masks)
            pil_image_list = rotate_normals_torch(pil_image_list, return_types='pil', rotate_direction=rotate_direction)
            normal_maps = pils_to_torch_imgs(pil_image_list, normal_maps.dtype, normal_maps.device)
        return (normal_maps,)
    
class Fast_Clean_Mesh:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("TRIMESH",),
                "apply_smooth": ("BOOLEAN", {"default": True},),
                "smooth_step": ("INT", {"default": 1, "min": 0, "max": 0xffffffffffffffff}),
                "apply_sub_divide": ("BOOLEAN", {"default": True},),
                "sub_divide_threshold": ("FLOAT", {"default": 0.25, "step": 0.001}),
            },
        }

    RETURN_TYPES = (
        "TRIMESH",
    )
    RETURN_NAMES = (
        "mesh",
    )
    FUNCTION = "clean_mesh"
    CATEGORY = "Comfy3D/Preprocessor"

    def clean_mesh(self, mesh, apply_smooth, smooth_step, apply_sub_divide, sub_divide_threshold):

        mesh = _wire_in(mesh)
        meshes = simple_clean_mesh(to_pyml_mesh(mesh.v, mesh.f), apply_smooth=apply_smooth, stepsmoothnum=smooth_step, apply_sub_divide=apply_sub_divide, sub_divide_threshold=sub_divide_threshold).to(get_device())
        vertices, faces, _ = from_py3d_mesh(meshes)

        mesh = Mesh(v=vertices, f=faces, device=get_device())

        return (_wire_out(mesh),)
    
class Decimate_Mesh:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("TRIMESH",),
                "target": ("INT", {"default": 5e4, "min": 0, "max": 0xffffffffffffffff}),
                "remesh": ("BOOLEAN", {"default": True},),
                "optimalplacement": ("BOOLEAN", {"default": True},),
            },
        }

    RETURN_TYPES = (
        "TRIMESH",
    )
    RETURN_NAMES = (
        "mesh",
    )
    FUNCTION = "process_mesh"
    CATEGORY = "Comfy3D/Preprocessor"

    def process_mesh(self, mesh, target, remesh, optimalplacement):
        mesh = _wire_in(mesh)
        vertices, faces = decimate_mesh(mesh.v.detach().cpu().numpy(), mesh.f.detach().cpu().numpy(), target, remesh, optimalplacement)
        mesh.v, mesh.f = torch.from_numpy(vertices).to(get_device()), torch.from_numpy(faces).to(torch.int64).to(get_device())
        mesh.auto_normal()
        return (_wire_out(mesh),)

class Switch_3DGS_Axis:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "gs_ply": ("GS_PLY",),
                "axis_x_to": (["+x", "-x", "+y", "-y", "+z", "-z"],),
                "axis_y_to": (["+y", "-y", "+z", "-z", "+x", "-x"],),
                "axis_z_to": (["+z", "-z", "+x", "-x", "+y", "-y"],),
            },
        }

    RETURN_TYPES = (
        "GS_PLY",
    )
    RETURN_NAMES = (
        "switched_gs_ply",
    )
    FUNCTION = "switch_axis_and_scale"
    CATEGORY = "Comfy3D/Preprocessor"
    
    def switch_axis_and_scale(self, gs_ply, axis_x_to, axis_y_to, axis_z_to):
        switched_gs_ply = None
        if axis_x_to[1] != axis_y_to[1] and axis_x_to[1] != axis_z_to[1] and axis_y_to[1] != axis_z_to[1]:
            target_axis, target_scale, coordinate_invert_count = get_target_axis_and_scale([axis_x_to, axis_y_to, axis_z_to])
            switched_gs_ply = switch_ply_axis_and_scale(gs_ply, target_axis, target_scale, coordinate_invert_count)
        else:
            cstr(f"[{self.__class__.__name__}] axis_x_to: {axis_x_to}, axis_y_to: {axis_y_to}, axis_z_to: {axis_z_to} have to be on separated axis").error.print()
        
        return (switched_gs_ply, )
    
class Switch_Mesh_Axis:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("TRIMESH",),
                "axis_x_to": (["+x", "-x", "+y", "-y", "+z", "-z"],),
                "axis_y_to": (["+y", "-y", "+z", "-z", "+x", "-x"],),
                "axis_z_to": (["+z", "-z", "+x", "-x", "+y", "-y"],),
                "flip_normal": ("BOOLEAN", {"default": False},),
                "scale": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 100, "step": 0.01}),
            },
        }

    RETURN_TYPES = (
        "TRIMESH",
    )
    RETURN_NAMES = (
        "switched_mesh",
    )
    FUNCTION = "switch_axis_and_scale"
    CATEGORY = "Comfy3D/Preprocessor"
    
    def switch_axis_and_scale(self, mesh, axis_x_to, axis_y_to, axis_z_to, flip_normal, scale):
        
        mesh = _wire_in(mesh)
        switched_mesh = None
        
        if axis_x_to[1] != axis_y_to[1] and axis_x_to[1] != axis_z_to[1] and axis_y_to[1] != axis_z_to[1]:
            target_axis, target_scale, coordinate_invert_count = get_target_axis_and_scale([axis_x_to, axis_y_to, axis_z_to], scale)
            switched_mesh = switch_mesh_axis_and_scale(mesh, target_axis, target_scale, flip_normal)
        else:
            cstr(f"[{self.__class__.__name__}] axis_x_to: {axis_x_to}, axis_y_to: {axis_y_to}, axis_z_to: {axis_z_to} have to be on separated axis").error.print()
        
        return (_wire_out(switched_mesh), )
    
class Convert_3DGS_To_Pointcloud:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "gs_ply": ("GS_PLY",),
            },
        }

    RETURN_TYPES = (
        "POINTCLOUD",
    )
    RETURN_NAMES = (
        "points_cloud",
    )
    FUNCTION = "convert_gs_ply"
    CATEGORY = "Comfy3D/Preprocessor"
    
    def convert_gs_ply(self, gs_ply):
        
        points_cloud = ply_to_points_cloud(gs_ply)
        
        return (points_cloud, )
    
class Convert_Mesh_To_Pointcloud:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("TRIMESH",),
            },
        }

    RETURN_TYPES = (
        "POINTCLOUD",
    )
    RETURN_NAMES = (
        "points_cloud",
    )
    FUNCTION = "convert_mesh"
    CATEGORY = "Comfy3D/Preprocessor"
    
    def convert_mesh(self, mesh):
        
        mesh = _wire_in(mesh)
        points_cloud = mesh.convert_to_pointcloud()
        
        return (points_cloud, )
    
class Stack_Orbit_Camera_Poses:
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "orbit_radius_start": ("FLOAT", {"default": 1.75, "step": 0.0001}),
                "orbit_radius_stop": ("FLOAT", {"default": 1.75, "step": 0.0001}),
                "orbit_radius_step": ("FLOAT", {"default": 0.1, "step": 0.0001}),
                "elevation_start": ("FLOAT", {"default": 0.0, "min": ELEVATION_MIN, "max": ELEVATION_MAX, "step": 0.0001}),
                "elevation_stop": ("FLOAT", {"default": 0.0, "min": ELEVATION_MIN, "max": ELEVATION_MAX, "step": 0.0001}),
                "elevation_step": ("FLOAT", {"default": 0.0, "min": ELEVATION_MIN, "max": ELEVATION_MAX, "step": 0.0001}),
                "azimuth_start": ("FLOAT", {"default": 0.0, "min": AZIMUTH_MIN, "max": AZIMUTH_MAX, "step": 0.0001}),
                "azimuth_stop": ("FLOAT", {"default": 0.0, "min": AZIMUTH_MIN, "max": AZIMUTH_MAX, "step": 0.0001}),
                "azimuth_step": ("FLOAT", {"default": 0.0, "min": AZIMUTH_MIN, "max": AZIMUTH_MAX, "step": 0.0001}),
                "orbit_center_X_start": ("FLOAT", {"default": 0.0, "step": 0.0001}),
                "orbit_center_X_stop": ("FLOAT", {"default": 0.0, "step": 0.0001}),
                "orbit_center_X_step": ("FLOAT", {"default": 0.1, "step": 0.0001}),
                "orbit_center_Y_start": ("FLOAT", {"default": 0.0, "step": 0.0001}),
                "orbit_center_Y_stop": ("FLOAT", {"default": 0.0, "step": 0.0001}),
                "orbit_center_Y_step": ("FLOAT", {"default": 0.1, "step": 0.0001}),
                "orbit_center_Z_start": ("FLOAT", {"default": 0.0, "step": 0.0001}),
                "orbit_center_Z_stop": ("FLOAT", {"default": 0.0, "step": 0.0001}),
                "orbit_center_Z_step": ("FLOAT", {"default": 0.1, "step": 0.0001}),
            },
        }

    RETURN_TYPES = (
        "ORBIT_CAMPOSES",   # [[orbit radius, elevation, azimuth, orbit center X, orbit center Y, orbit center Z], ...]
        "FLOAT",
        "FLOAT",
        "FLOAT",
        "FLOAT",
        "FLOAT",
        "FLOAT",
    )
    RETURN_NAMES = (
        "orbit_camposes",
        "orbit_radius_list",
        "elevation_list", 
        "azimuth_list", 
        "orbit_center_X_list",  
        "orbit_center_Y_list",  
        "orbit_center_Z_list",
    )
    OUTPUT_IS_LIST = (
        False,
        True,
        True,
        True,
        True,
        True,
        True,
    )
    
    FUNCTION = "get_camposes"
    CATEGORY = "Comfy3D/Preprocessor"
    
    class Pose_Config(Enum):
        STOP_LARGER_STEP_POS = 0
        START_LARGER_STEP_POS = 1
        START_LARGER_STEP_NEG = 2
        STOP_LARGER_STEP_NEG = 3
    
    class Pose_Type:
        def __init__(self, start, stop, step, min_value=-math.inf, max_value=math.inf, is_linear = True):
            if abs(step) < 0.0001:
                step = 0.0001 * (-1.0 if step < 0 else 1.0)
            
            if is_linear and ( (step > 0 and stop < start) or (step < 0 and stop > start)):
                cstr(f"[{self.__class__.__name__}] stop value: {stop} cannot be reached from start value {start} with step value {step}, will reverse the sign of step value to {-step}").warning.print()
                self.step = -step
            else:
                self.step = step
                      
            self.start = start
            self.stop = stop
            
            self.min = min_value
            self.max = max_value
            
            self.is_linear = is_linear  # linear or circular (i.e. min and max value are connected, e.g. -180 & 180 degree in azimuth angle) value
                
    
    def stack_camposes(self, pose_type_index=None, last_camposes=[[]]):
        if pose_type_index == None:
            pose_type_index = len(self.all_pose_types) - 1
            
        if pose_type_index == -1:
            return last_camposes
        else:
            current_pose_type = self.all_pose_types[pose_type_index]
            
            all_camposes = []
            
            # There are four different kind of situation we need to deal with to make this function generalize for any combination of inputs
            if current_pose_type.step > 0:
                if current_pose_type.start < current_pose_type.stop or current_pose_type.is_linear:
                    pose_config = Stack_Orbit_Camera_Poses.Pose_Config.STOP_LARGER_STEP_POS
                else:
                    pose_config = Stack_Orbit_Camera_Poses.Pose_Config.START_LARGER_STEP_POS
            else:
                if current_pose_type.start > current_pose_type.stop or current_pose_type.is_linear:
                    pose_config = Stack_Orbit_Camera_Poses.Pose_Config.START_LARGER_STEP_NEG
                else:
                    pose_config = Stack_Orbit_Camera_Poses.Pose_Config.STOP_LARGER_STEP_NEG
                    
            p = current_pose_type.start
            p_passed_min_max_seam = False
            
            while ( (pose_config == Stack_Orbit_Camera_Poses.Pose_Config.STOP_LARGER_STEP_POS and p <= current_pose_type.stop) or 
                    (pose_config == Stack_Orbit_Camera_Poses.Pose_Config.START_LARGER_STEP_POS and (not p_passed_min_max_seam or p <= current_pose_type.stop)) or
                    (pose_config == Stack_Orbit_Camera_Poses.Pose_Config.START_LARGER_STEP_NEG and p >= current_pose_type.stop) or 
                    (pose_config == Stack_Orbit_Camera_Poses.Pose_Config.STOP_LARGER_STEP_NEG and (not p_passed_min_max_seam or p >= current_pose_type.stop)) ):
                
                # If current pose value surpass the either min/max value then we map its vaule to the oppsite sign
                if pose_config == Stack_Orbit_Camera_Poses.Pose_Config.START_LARGER_STEP_POS and p > current_pose_type.max:
                    p = current_pose_type.min + p % current_pose_type.max
                    p_passed_min_max_seam = True    
                elif pose_config == Stack_Orbit_Camera_Poses.Pose_Config.STOP_LARGER_STEP_NEG and p < current_pose_type.min:
                    p = current_pose_type.max + p % current_pose_type.min
                    p_passed_min_max_seam = True
                    
                new_camposes = copy.deepcopy(last_camposes)
                    
                for campose in new_camposes:
                    campose.insert(0, p)
                    
                all_camposes.extend(new_camposes)
                
                p += current_pose_type.step
                    
            return self.stack_camposes(pose_type_index-1, all_camposes)
    
    def get_camposes(self, 
                     orbit_radius_start, 
                     orbit_radius_stop, 
                     orbit_radius_step, 
                     elevation_start, 
                     elevation_stop, 
                     elevation_step, 
                     azimuth_start, 
                     azimuth_stop, 
                     azimuth_step, 
                     orbit_center_X_start, 
                     orbit_center_X_stop, 
                     orbit_center_X_step, 
                     orbit_center_Y_start, 
                     orbit_center_Y_stop, 
                     orbit_center_Y_step, 
                     orbit_center_Z_start, 
                     orbit_center_Z_stop, 
                     orbit_center_Z_step):
        
        """
            Return the combination of all the pose types interpolation values
            Return values in two ways:
            orbit_camposes: CAMPOSES type list can directly input to other 3D process node (e.g. GaussianSplatting)
            all the camera pose types seperated in different list, becasue some 3D model's conditioner only takes a sub set of all camera pose types (e.g. StableZero123)
        """
        
        orbit_radius_list = []
        elevation_list = []
        azimuth_list = []
        orbit_center_X_list = []
        orbit_center_Y_list = []
        orbit_center_Z_list = []
        
        self.all_pose_types = []
        self.all_pose_types.append( Stack_Orbit_Camera_Poses.Pose_Type(orbit_radius_start, orbit_radius_stop, orbit_radius_step) )
        self.all_pose_types.append( Stack_Orbit_Camera_Poses.Pose_Type(elevation_start, elevation_stop, elevation_step, ELEVATION_MIN, ELEVATION_MAX) )
        self.all_pose_types.append( Stack_Orbit_Camera_Poses.Pose_Type(azimuth_start, azimuth_stop, azimuth_step, AZIMUTH_MIN, AZIMUTH_MAX, False) )
        self.all_pose_types.append( Stack_Orbit_Camera_Poses.Pose_Type(orbit_center_X_start, orbit_center_X_stop, orbit_center_X_step) )
        self.all_pose_types.append( Stack_Orbit_Camera_Poses.Pose_Type(orbit_center_Y_start, orbit_center_Y_stop, orbit_center_Y_step) )
        self.all_pose_types.append( Stack_Orbit_Camera_Poses.Pose_Type(orbit_center_Z_start, orbit_center_Z_stop, orbit_center_Z_step) )
        
        orbit_camposes = self.stack_camposes()
        
        for campose in orbit_camposes:
            orbit_radius_list.append(campose[0])
            elevation_list.append(campose[1])
            azimuth_list.append(campose[2])
            orbit_center_X_list.append(campose[3])
            orbit_center_Y_list.append(campose[4])
            orbit_center_Z_list.append(campose[5])
        
        return (orbit_camposes, orbit_radius_list, elevation_list, azimuth_list, orbit_center_X_list, orbit_center_Y_list, orbit_center_Z_list, )

class Get_Camposes_From_List_Indexed:
    
    RETURN_TYPES = ("ORBIT_CAMPOSES",)
    FUNCTION = "get_indexed_camposes"
    CATEGORY = "Comfy3D/Preprocessor"
    DESCRIPTION = """
        Selects and returns the camera poses at the specified indices as an list.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "original_orbit_camera_poses": ("ORBIT_CAMPOSES",),    # [orbit radius, elevation, azimuth, orbit center X,  orbit center Y,  orbit center Z]
                "indexes": ("STRING", {"default": "0, 1, 2", "multiline": True}),
            },
        }
    
    def get_indexed_camposes(self, original_orbit_camera_poses, indexes):
        
        # Parse the indexes string into a list of integers
        index_list = [int(index.strip()) for index in indexes.split(',')]
        
        # Select the camposes at the specified indices
        orbit_camera_poses = []
        for pose_list in original_orbit_camera_poses:
            new_pose_list = [pose_list[i] for i in index_list]
            orbit_camera_poses.append(new_pose_list)

        return (orbit_camera_poses,)

class Mesh_Orbit_Renderer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("TRIMESH",),
                "render_image_width": ("INT", {"default": 1024, "min": 128, "max": 8192}),
                "render_image_height": ("INT", {"default": 1024, "min": 128, "max": 8192}),
                "render_orbit_camera_poses": ("ORBIT_CAMPOSES",),    # [orbit radius, elevation, azimuth, orbit center X,  orbit center Y,  orbit center Z]
                "render_orbit_camera_fovy": ("FLOAT", {"default": 49.1, "min": 0.0, "max": 180.0, "step": 0.1}),
                "render_background_color_r": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "render_background_color_g": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "render_background_color_b": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "force_cuda_rasterize": ("BOOLEAN", {"default": True},),
            },
            
            "optional": {
                "render_depth": ("BOOLEAN", {"default": False},),
                "render_normal": ("BOOLEAN", {"default": False},),
            }
        }
        
    RETURN_TYPES = (
        "IMAGE",
        "MASK",
        "IMAGE",
        "IMAGE",
        "IMAGE",
    )
    RETURN_NAMES = (
        "rendered_mesh_images",   # [Number of Poses, H, W, 3]
        "rendered_mesh_masks",    # [Number of Poses, H, W, 1]
        "all_rendered_depths",    # [Number of Poses, H, W, 3]
        "all_rendered_normals",   # [Number of Poses, H, W, 3]
        "all_rendered_viewcos",   # [Number of Poses, H, W, 3]
    )
    
    FUNCTION = "render_mesh"
    CATEGORY = "Comfy3D/Preprocessor"
    
    def render_mesh(
        self, 
        mesh, 
        render_image_width, 
        render_image_height, 
        render_orbit_camera_poses, 
        render_orbit_camera_fovy,
        render_background_color_r, 
        render_background_color_g, 
        render_background_color_b,
        force_cuda_rasterize,
        render_depth=False,
        render_normal=False,
    ):
        
        mesh = _wire_in(mesh)
        renderer = DiffRastRenderer(mesh, force_cuda_rasterize)
        
        optional_render_types = []
        if render_depth:
            optional_render_types.append('depth')
        if render_normal:
            optional_render_types.append('normal')
        
        cam_controller = DiffMeshCameraController(
            renderer, 
            render_image_width, 
            render_image_height, 
            render_orbit_camera_fovy, 
            static_bg=[render_background_color_r, render_background_color_g, render_background_color_b]
        )
        
        extra_kwargs = {"optional_render_types": optional_render_types}
        all_rendered_images, all_rendered_masks, extra_outputs = cam_controller.render_all_pose(render_orbit_camera_poses, **extra_kwargs)
        all_rendered_masks = all_rendered_masks.squeeze(-1)  # [N, H, W, 1] -> [N, H, W]
        if 'depth' in extra_outputs:
            all_rendered_depths = extra_outputs['depth'].repeat(1, 1, 1, 3)   # [N, H, W, 1] -> [N, H, W, 3]
        else:
            all_rendered_depths = None
        
        if 'normal' in extra_outputs:
            all_rendered_normals = extra_outputs['normal']
            all_rendered_viewcos = extra_outputs['viewcos']
        else:
            all_rendered_normals = None
            all_rendered_viewcos = None
        
        return (all_rendered_images, all_rendered_masks, all_rendered_depths, all_rendered_normals, all_rendered_viewcos)
        
   
class Gaussian_Splatting_Orbit_Renderer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "gs_ply": ("GS_PLY",),
                "render_image_width": ("INT", {"default": 1024, "min": 128, "max": 8192}),
                "render_image_height": ("INT", {"default": 1024, "min": 128, "max": 8192}),
                "render_orbit_camera_poses": ("ORBIT_CAMPOSES",),    # [orbit radius, elevation, azimuth, orbit center X,  orbit center Y,  orbit center Z]
                "render_orbit_camera_fovy": ("FLOAT", {"default": 49.1, "min": 0.0, "max": 180.0, "step": 0.1}),
                "render_background_color_r": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "render_background_color_g": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "render_background_color_b": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
            }
        }
        
    RETURN_TYPES = (
        "IMAGE",
        "MASK",
        "IMAGE",
    )
    RETURN_NAMES = (
        "rendered_gs_images",    # [Number of Poses, H, W, 3]
        "rendered_gs_masks",     # [Number of Poses, H, W, 1]
        "rendered_gs_depths",   # [Number of Poses, H, W, 3]
    )
    
    FUNCTION = "render_gs"
    CATEGORY = "Comfy3D/Preprocessor"
    
    def render_gs(
        self, 
        gs_ply, 
        render_image_width, 
        render_image_height, 
        render_orbit_camera_poses, 
        render_orbit_camera_fovy,
        render_background_color_r, 
        render_background_color_g, 
        render_background_color_b,
    ):
        
        sh_degree, _ = calculate_max_sh_degree_from_gs_ply(gs_ply)
        renderer = GaussianSplattingRenderer(sh_degree=sh_degree)
        renderer.initialize(gs_ply)
        
        cam_controller = GaussianSplattingCameraController(
            renderer, 
            render_image_width, 
            render_image_height, 
            render_orbit_camera_fovy, 
            static_bg=[render_background_color_r, render_background_color_g, render_background_color_b]
        )
        
        all_rendered_images, all_rendered_masks, extra_outputs = cam_controller.render_all_pose(render_orbit_camera_poses)
        all_rendered_images = all_rendered_images.permute(0, 2, 3, 1)   # [N, 3, H, W] -> [N, H, W, 3]
        all_rendered_masks = all_rendered_masks.squeeze(1)  # [N, 1, H, W] -> [N, H, W]
        
        if 'depth' in extra_outputs:
            all_rendered_depths = extra_outputs['depth'].permute(0, 2, 3, 1).repeat(1, 1, 1, 3)   # [N, 1, H, W] -> [N, H, W, 3]
        else:
            all_rendered_depths = None
        
        return (all_rendered_images, all_rendered_masks, all_rendered_depths)
    
class Gaussian_Splatting_3D:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reference_images": ("IMAGE",), 
                "reference_masks": ("MASK",),
                "reference_orbit_camera_poses": ("ORBIT_CAMPOSES",),    # [orbit radius, elevation, azimuth, orbit center X,  orbit center Y,  orbit center Z]
                "reference_orbit_camera_fovy": ("FLOAT", {"default": 49.1, "min": 0.0, "max": 180.0, "step": 0.1}),
                "training_iterations": ("INT", {"default": 30_000, "min": 1, "max": 0xffffffffffffffff}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 0xffffffffffffffff}),
                "ms_ssim_loss_weight": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, }),
                "alpha_loss_weight": ("FLOAT", {"default": 3, "min": 0.0, }),
                "offset_loss_weight": ("FLOAT", {"default": 0.0, "min": 0.0, }),
                "offset_opacity_loss_weight": ("FLOAT", {"default": 0.0, "min": 0.0, }),
                "invert_background_probability": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1}),
                "feature_learning_rate": ("FLOAT", {"default": 0.0025, "min": 0.000001, "step": 0.000001}),
                "opacity_learning_rate": ("FLOAT", {"default": 0.05,  "min": 0.000001, "step": 0.000001}),
                "scaling_learning_rate": ("FLOAT", {"default": 0.005,  "min": 0.000001, "step": 0.000001}),
                "rotation_learning_rate": ("FLOAT", {"default": 0.001,  "min": 0.000001, "step": 0.000001}),
                "position_learning_rate_init": ("FLOAT", {"default": 0.00016,  "min": 0.000001, "step": 0.000001}),
                "position_learning_rate_final": ("FLOAT", {"default": 0.0000016, "min": 0.0000001, "step": 0.0000001}),
                "position_learning_rate_delay_mult": ("FLOAT", {"default": 0.01, "min": 0.000001, "step": 0.000001}),
                "position_learning_rate_max_steps": ("INT", {"default": 30_000, "min": 1, "max": 0xffffffffffffffff}),
                "initial_gaussians_num": ("INT", {"default": 10_000, "min": 1, "max": 0xffffffffffffffff}),
                "K_nearest_neighbors": ("INT", {"default": 3, "min": 1, "max": 0xffffffffffffffff}),
                "percent_dense": ("FLOAT", {"default": 0.01,  "min": 0.00001, "step": 0.00001}),
                "density_start_iterations": ("INT", {"default": 500, "min": 0, "max": 0xffffffffffffffff}),
                "density_end_iterations": ("INT", {"default": 15_000, "min": 0, "max": 0xffffffffffffffff}),
                "densification_interval": ("INT", {"default": 100, "min": 1, "max": 0xffffffffffffffff}),
                "opacity_reset_interval": ("INT", {"default": 3000, "min": 1, "max": 0xffffffffffffffff}),
                "densify_grad_threshold": ("FLOAT", {"default": 0.0002, "min": 0.00001, "step": 0.00001}),
                "gaussian_sh_degree": ("INT", {"default": 3, "min": 0}),
            },
            
            "optional": {
                "points_cloud_to_initialize_gaussian": ("POINTCLOUD",),
                "ply_to_initialize_gaussian": ("GS_PLY",),
                "mesh_to_initialize_gaussian": ("TRIMESH",),
            }
        }

    RETURN_TYPES = (
        "GS_PLY",
    )
    RETURN_NAMES = (
        "gs_ply",
    )
    FUNCTION = "run_gs"
    CATEGORY = "Comfy3D/Algorithm"
    
    def run_gs(
        self,
        reference_images,
        reference_masks,
        reference_orbit_camera_poses,
        reference_orbit_camera_fovy,
        training_iterations,
        batch_size,
        ms_ssim_loss_weight,
        alpha_loss_weight,
        offset_loss_weight,
        offset_opacity_loss_weight,
        invert_background_probability,
        feature_learning_rate,
        opacity_learning_rate,
        scaling_learning_rate,
        rotation_learning_rate,
        position_learning_rate_init,
        position_learning_rate_final,
        position_learning_rate_delay_mult,
        position_learning_rate_max_steps,
        initial_gaussians_num,
        K_nearest_neighbors,
        percent_dense,
        density_start_iterations,
        density_end_iterations,
        densification_interval,
        opacity_reset_interval,
        densify_grad_threshold,
        gaussian_sh_degree,
        points_cloud_to_initialize_gaussian=None,
        ply_to_initialize_gaussian=None,
        mesh_to_initialize_gaussian=None,
    ):
        
        mesh_to_initialize_gaussian = _wire_in(mesh_to_initialize_gaussian)
        gs_ply = None
        
        ref_imgs_num = len(reference_images)
        ref_masks_num = len(reference_masks)
        if ref_imgs_num == ref_masks_num:
            
            ref_cam_poses_num = len(reference_orbit_camera_poses)
            if ref_imgs_num == ref_cam_poses_num:
                
                if batch_size > ref_imgs_num:
                    cstr(f"[{self.__class__.__name__}] Batch size {batch_size} is bigger than number of reference images {ref_imgs_num}! Set batch size to {ref_imgs_num} instead").warning.print()
                    batch_size = ref_imgs_num
                
                with torch.inference_mode(False):
                
                    gs_params = GSParams(
                        training_iterations,
                        batch_size,
                        ms_ssim_loss_weight,
                        alpha_loss_weight,
                        offset_loss_weight,
                        offset_opacity_loss_weight,
                        invert_background_probability,
                        feature_learning_rate,
                        opacity_learning_rate,
                        scaling_learning_rate,
                        rotation_learning_rate,
                        position_learning_rate_init,
                        position_learning_rate_final,
                        position_learning_rate_delay_mult,
                        position_learning_rate_max_steps,
                        initial_gaussians_num,
                        K_nearest_neighbors,
                        percent_dense,
                        density_start_iterations,
                        density_end_iterations,
                        densification_interval,
                        opacity_reset_interval,
                        densify_grad_threshold,
                        gaussian_sh_degree
                    )
                    
                    
                    if points_cloud_to_initialize_gaussian is not None:
                        gs_init_input = points_cloud_to_initialize_gaussian
                    elif ply_to_initialize_gaussian is not None:
                        gs_init_input = ply_to_initialize_gaussian
                    else:
                        gs_init_input = mesh_to_initialize_gaussian
                    
                    gs = GaussianSplatting3D(gs_params, gs_init_input)
                    gs.prepare_training(reference_images, reference_masks, reference_orbit_camera_poses, reference_orbit_camera_fovy)
                    gs.training()

                    gs_ply = gs.renderer.gaussians.to_ply()
                
            else:
                cstr(f"[{self.__class__.__name__}] Number of reference images {ref_imgs_num} does not equal to number of reference camera poses {ref_cam_poses_num}").error.print()
        else:
            cstr(f"[{self.__class__.__name__}] Number of reference images {ref_imgs_num} does not equal to number of masks {ref_masks_num}").error.print()

        return (gs_ply, )
    
class Fitting_Mesh_With_Multiview_Images:
    
    def __init__(self):
        self.need_update = False
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reference_images": ("IMAGE",), 
                "reference_masks": ("MASK",),
                "reference_orbit_camera_poses": ("ORBIT_CAMPOSES",),    # [orbit radius, elevation, azimuth, orbit center X,  orbit center Y,  orbit center Z]
                "reference_orbit_camera_fovy": ("FLOAT", {"default": 49.1, "min": 0.0, "max": 180.0, "step": 0.1}),
                "mesh": ("TRIMESH",),
                "mesh_albedo_width": ("INT", {"default": 1024, "min": 128, "max": 8192}),
                "mesh_albedo_height": ("INT", {"default": 1024, "min": 128, "max": 8192}),
                "training_iterations": ("INT", {"default": 1024, "min": 1, "max": 100000}),
                "batch_size": ("INT", {"default": 3, "min": 1, "max": 0xffffffffffffffff}),
                "texture_learning_rate": ("FLOAT", {"default": 0.001, "min": 0.00001, "step": 0.00001}),
                "train_mesh_geometry": ("BOOLEAN", {"default": False},),
                "geometry_learning_rate": ("FLOAT", {"default": 0.0001, "min": 0.00001, "step": 0.00001}),
                "ms_ssim_loss_weight": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "remesh_after_n_iteration": ("INT", {"default": 512, "min": 128, "max": 100000}),
                "invert_background_probability": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1}),
                "force_cuda_rasterize": ("BOOLEAN", {"default": True},),
            },
        }

    RETURN_TYPES = (
        "TRIMESH",
        "IMAGE",
    )
    RETURN_NAMES = (
        "trained_mesh",
        "baked_texture",    # [1, H, W, 3]
    )
    FUNCTION = "fitting_mesh"
    CATEGORY = "Comfy3D/Algorithm"
    
    def fitting_mesh(
        self, 
        reference_images, 
        reference_masks, 
        reference_orbit_camera_poses, 
        reference_orbit_camera_fovy, 
        mesh, 
        mesh_albedo_width,
        mesh_albedo_height,
        training_iterations, 
        batch_size, 
        texture_learning_rate, 
        train_mesh_geometry, 
        geometry_learning_rate, 
        ms_ssim_loss_weight, 
        remesh_after_n_iteration,
        invert_background_probability,
        force_cuda_rasterize,
    ):
        
        mesh = _wire_in(mesh)
        if mesh.vt is None:
            mesh.auto_uv()
            
        mesh.set_new_albedo(mesh_albedo_width, mesh_albedo_height)
        
        trained_mesh = None
        baked_texture = None
        
        ref_imgs_num = len(reference_images)
        ref_masks_num = len(reference_masks)
        if ref_imgs_num == ref_masks_num:
            
            ref_cam_poses_num = len(reference_orbit_camera_poses)
            if ref_imgs_num == ref_cam_poses_num:
                
                if batch_size > ref_imgs_num:
                    cstr(f"[{self.__class__.__name__}] Batch size {batch_size} is bigger than number of reference images {ref_imgs_num}! Set batch size to {ref_imgs_num} instead").warning.print()
                    batch_size = ref_imgs_num
                    
                with torch.inference_mode(False):
                
                    mesh_fitter = DiffMesh(
                        mesh, 
                        training_iterations, 
                        batch_size, 
                        texture_learning_rate, 
                        train_mesh_geometry, 
                        geometry_learning_rate, 
                        ms_ssim_loss_weight, 
                        remesh_after_n_iteration, 
                        invert_background_probability, 
                        force_cuda_rasterize
                    )
                    
                    mesh_fitter.prepare_training(reference_images, reference_masks, reference_orbit_camera_poses, reference_orbit_camera_fovy)
                    mesh_fitter.training()
                    
                    trained_mesh, baked_texture = mesh_fitter.get_mesh_and_texture()
                    
            else:
                cstr(f"[{self.__class__.__name__}] Number of reference images {ref_imgs_num} does not equal to number of reference camera poses {ref_cam_poses_num}").error.print()
        else:
            cstr(f"[{self.__class__.__name__}] Number of reference images {ref_imgs_num} does not equal to number of masks {ref_masks_num}").error.print()
        
        return (_wire_out(trained_mesh), baked_texture, )

class Load_Triplane_Gaussian_Transformers:
    

    # Announces in the UI that this node fetches weights if they are absent.
    DISPLAY_NAME = "(Down)Load Triplane Gaussian Transformers"

    checkpoints_dir = "TriplaneGaussian"
    default_ckpt_name = "model_lvis_rel.ckpt"
    default_repo_id = "VAST-AI/TriplaneGaussian"
    config_path = "TriplaneGaussian_config.yaml"
    
    @classmethod
    def ckpt_dir(cls):
        """Absolute checkpoints directory.

        Derived on demand, not assigned as an INPUT_TYPES side effect:
        INPUT_TYPES need not have run in the process that executes
        this node. See Load_Hunyuan3D_V1.svrm_config().
        """
        return os.path.join(CKPT_ROOT_PATH, cls.checkpoints_dir)

    @classmethod
    def config_file(cls):
        """Absolute path to this node's config file.

        Derived on demand, not assigned as an INPUT_TYPES side effect:
        INPUT_TYPES need not have run in the process that executes
        this node. See Load_Hunyuan3D_V1.svrm_config().
        """
        return os.path.join(CONFIG_ROOT_PATH, cls.config_path)

    @classmethod
    def INPUT_TYPES(cls):
        all_models_names = get_list_filenames(cls.ckpt_dir(), SUPPORTED_CHECKPOINTS_EXTENSIONS)
        if cls.default_ckpt_name not in all_models_names:
            all_models_names += [cls.default_ckpt_name]
        
        return {
            "required": {
                "model_name": (all_models_names, ),
            },
        }
    
    RETURN_TYPES = (
        "TGS_MODEL",
    )
    RETURN_NAMES = (
        "tgs_model",
    )
    FUNCTION = "load_TGS"
    CATEGORY = "Comfy3D/Import|Export"
    
    def load_TGS(self, model_name):

        device = get_device()

        ckpt_path = resume_or_download_model_from_hf(
            self.ckpt_dir(), self.default_repo_id, model_name, self.__class__.__name__)

        # Emit a recipe, not the model. TGS carries activation functions built
        # as closures (get_activation returns a lambda), and a lambda has no
        # importable qualname, so returning it live killed the worker at the
        # transport:
        #     Can't get local object 'get_activation.<locals>.<lambda>'
        # Fourth loader of this shape, after instantmesh_lrm, crm_mvdiffusion
        # and crm_recon.
        cstr(f"[{self.__class__.__name__}] ckpt ready at {ckpt_path}").msg.print()

        return (model_cache.recipe(
            "tgs",
            ckpt=ckpt_path,
            config=self.config_file(),
        ), )
    
def _build_tgs(config):
    """Materialize a tgs recipe. Only on a cache miss."""
    cfg: ExperimentConfigTGS = load_config_tgs(config["config"])
    cfg.system.weights = config["ckpt"]
    # On CPU: the ModelPatcher decides when this reaches the GPU.
    return TGS(cfg=cfg.system)


def resolve_tgs_model(config):
    """TGS_MODEL recipe -> live model, cached across runs.

    Non-dicts pass through, so consumers can call this unconditionally.
    """
    if not isinstance(config, dict):
        return config
    return model_cache.managed("tgs", config, _build_tgs, reloadable=True)


class Triplane_Gaussian_Transformers:
    
    config_path = "TriplaneGaussian_config.yaml"
    
    @classmethod
    def config_file(cls):
        """Absolute path to this node's config file.

        Derived on demand, not assigned as an INPUT_TYPES side effect:
        INPUT_TYPES need not have run in the process that executes
        this node. See Load_Hunyuan3D_V1.svrm_config().
        """
        return os.path.join(CONFIG_ROOT_PATH, cls.config_path)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reference_image": ("IMAGE", ),
                "reference_mask": ("MASK",),
                "tgs_model": ("TGS_MODEL", ),
                "cam_dist": ("FLOAT", {"default": 1.9, "min": 0.01, "step": 0.01}),
            },
        }
    
    RETURN_TYPES = (
        "GS_PLY",
    )
    RETURN_NAMES = (
        "gs_ply",
    )
    FUNCTION = "run_TGS"
    CATEGORY = "Comfy3D/Algorithm"
    
    def run_TGS(self, reference_image, reference_mask, tgs_model, cam_dist):

        # A TGS_MODEL arrives as a recipe; resolve_tgs_model passes a live
        # model straight through.
        tgs_model = resolve_tgs_model(tgs_model)        
        cfg: ExperimentConfigTGS = load_config_tgs(self.config_file())

        cfg.data.cond_camera_distance = cam_dist
        cfg.data.eval_camera_distance = cam_dist
        dataset = CustomImageOrbitDataset(reference_image, reference_mask, cfg.data)
        dataloader = DataLoader(
            dataset,
            batch_size=cfg.data.eval_batch_size, 
            shuffle=False,
            collate_fn=dataset.collate
        )

        gs_ply = []
        for batch in dataloader:
            batch = todevice(batch)
            gs_ply.extend(tgs_model(batch))
        
        return (gs_ply[0], )
    
class Load_Diffusers_Pipeline:
    
    # Announces in the UI that this node fetches weights if they are absent.
    DISPLAY_NAME = "(Down)Load Diffusers Pipeline"

    @classmethod

    def INPUT_TYPES(cls):
        return {
            "required": {
                "diffusers_pipeline_name": (list(DIFFUSERS_PIPE_DICT.keys()),),
                "repo_id": ("STRING", {"default": "ashawkey/imagedream-ipmv-diffusers", "multiline": False}),
            },
            "optional": {
                "checkpoint_sub_dir": ("STRING", {"default": "", "multiline": False}),
            }
        }
    
    RETURN_TYPES = (
        "DIFFUSERS_PIPE",
    )
    RETURN_NAMES = (
        "pipe",
    )
    FUNCTION = "load_diffusers_pipe"
    CATEGORY = "Comfy3D/Import|Export"
    
    def load_diffusers_pipe(self, diffusers_pipeline_name, repo_id, checkpoint_sub_dir=""):

        # Download here (cheap when already present, and it keeps the loader
        # node as the thing that reports network failures), but do NOT build
        # the pipeline -- that happens in resolve_diffusers_pipe on first use.
        ckpt_download_dir = os.path.join(CKPT_DIFFUSERS_PATH, repo_id)
        download_repo(repo_id, ckpt_download_dir, ignore_patterns=HF_DOWNLOAD_IGNORE)

        return (model_cache.recipe(
            "diffusers_pipe",
            pipeline_name=diffusers_pipeline_name,
            repo_id=repo_id,
            checkpoint_sub_dir=checkpoint_sub_dir,
        ), )


def _try_enable_xformers(obj, disabled=False):
    """Switch `obj` to xformers attention if xformers is installed and wanted.

    xformers is deliberately not a dependency of this pack -- it caps flash-attn
    at <=2.8.2 while the wheel farm builds 2.8.3, and that ImportError used to
    take every node in the pack down with it. See nodes/comfy-env.toml.

    Skipping this is a backend change, not a lost optimisation: on torch 2.x
    diffusers already defaults to AttnProcessor2_0, which is PyTorch SDPA and
    dispatches to flash kernels. This call was switching *off* that default.
    Kept so an environment that does have xformers still behaves as before.
    """
    if disabled or not hasattr(obj, "enable_xformers_memory_efficient_attention"):
        return
    try:
        obj.enable_xformers_memory_efficient_attention()
    except (ModuleNotFoundError, ImportError, ValueError) as exc:
        # ModuleNotFoundError is what diffusers raises when xformers is absent
        # (attention_processor.py:405); ValueError when the device cannot use it.
        cstr(f"[Comfy3D] xformers attention unavailable, using the default "
             f"backend instead ({type(exc).__name__})").msg.print()
        _enable_custom_xformers_processors(obj)


def _enable_custom_xformers_processors(obj):
    """Install the XFormers* processors on modules that define their own.

    Some vendored models -- Era3D is the one that bites -- subclass Attention
    and OVERRIDE set_use_memory_efficient_attention_xformers to swap in their
    own processor, with no is_xformers_available() check:

        class CustomJointAttention(Attention):
            def set_use_memory_efficient_attention_xformers(self, *a, **kw):
                self.set_processor(XFormersJointAttnProcessor())

    Those XFormers* processors are the ones the model's blocks are written
    against; the plain MVAttnProcessor / JointAttnProcessor siblings produce a
    shape the block cannot add to its residual:

        RuntimeError: The size of tensor a (24) must match the size of
        tensor b (4096) at non-singleton dimension 1

    diffusers' recursive enable never reaches them: it walks the module tree
    and the first PLAIN Attention raises ModuleNotFoundError, aborting the
    whole pass. So the override is called directly here, on exactly the modules
    that define one.

    Safe without xformers installed, which is the entire point: those
    processors call xformers.ops.memory_efficient_attention, which resolves to
    the SDPA-backed stand-in in shared_utils/xformers_compat.py.
    """
    from diffusers.models.attention_processor import Attention

    modules = getattr(obj, "modules", None)
    if modules is None:
        return
    patched = 0
    for m in modules():
        if not isinstance(m, Attention):
            continue
        # Only subclasses that define their own override -- never the plain
        # Attention, whose implementation is the one that just raised.
        fn = type(m).set_use_memory_efficient_attention_xformers
        if fn is Attention.set_use_memory_efficient_attention_xformers:
            continue
        try:
            m.set_use_memory_efficient_attention_xformers(True)
            patched += 1
        except Exception:
            pass
    if patched:
        cstr(f"[Comfy3D] installed {patched} model-specific xformers "
             f"processor(s), routed through the SDPA stand-in").msg.print()


# recipe kind -> builder. A consumer takes a DIFFUSERS_PIPE without knowing
# which loader filled it, so it cannot pick a per-loader resolver; it calls
# resolve_diffusers_pipe and this table decides what to build.
_PIPE_BUILDERS = {}


def _pipe_builder(kind):
    """Register a builder for one recipe kind."""
    def register(fn):
        _PIPE_BUILDERS[kind] = fn
        return fn
    return register


@_pipe_builder("diffusers_pipe")
def _build_diffusers_pipe(config):
    """Materialize a DIFFUSERS_PIPE recipe. Called only on a cache miss."""
    ckpt_download_dir = os.path.join(CKPT_DIFFUSERS_PATH, config["repo_id"])
    sub_dir = config.get("checkpoint_sub_dir")
    ckpt_path = os.path.join(ckpt_download_dir, sub_dir) if sub_dir else ckpt_download_dir

    diffusers_pipeline_class = DIFFUSERS_PIPE_DICT[config["pipeline_name"]]
    # Several of these repos name a VENDORED module as a sub-model's library:
    # Wonder3D.models.unet_mv2d_condition, Era3D.mvdiffusion.models...,
    # TripoSG.models... diffusers imports those bare, so they must be resolvable
    # under their legacy top-level name before from_pretrained runs.
    _alias_model_index_libraries(ckpt_path)
    # No custom_pipeline: it was '' in every shipped workflow, the classes in
    # DIFFUSERS_PIPE_DICT are imported directly, and it is the one argument that
    # would have diffusers fetch and exec code from a repo id.
    # trust_remote_code: some of these repos ship a custom sub-model class as a
    # .py beside the weights (ashawkey/imagedream-ipmv-diffusers/unet/mv_unet.py),
    # and diffusers refuses to exec it without this. The file is TRACKED IN THIS
    # REPO and HF_DOWNLOAD_IGNORE lists "*.py", so a download can never replace
    # it -- what runs is pack-pinned code, not something fetched at load time.
    # Cast dtype, do NOT place on device: comfy/sd.py does the same
    # (`first_stage_model.to(self.vae_dtype)`, then the module goes to a
    # ModelPatcher). model_cache.managed() wraps this so load_models_gpu()
    # chooses placement -- that is what makes eviction and partial loading
    # possible. Moving here spends the VRAM before ComfyUI is asked if
    # there is room, which is where every OOM in the 20:05 log was raised.
    pipe = diffusers_pipeline_class.from_pretrained(
        ckpt_path,
        torch_dtype=WEIGHT_DTYPE,
        trust_remote_code=True,
    )

    _try_enable_xformers(pipe)

    # Replay the mutator chain in the order the graph applied it.
    for op in config.get("ops", []):
        if op["op"] == "scheduler":
            scheduler_class = DIFFUSERS_SCHEDULER_DICT[op["name"]]
            pipe.scheduler = scheduler_class.from_config(
                pipe.scheduler.config, timestep_spacing='trailing'
            )
        elif op["op"] == "state_dict":
            checkpoints_dir_abs = os.path.join(CKPT_DIFFUSERS_PATH, op["repo_id"])
            ckpt_path = resume_or_download_model_from_hf(
                checkpoints_dir_abs, op["repo_id"], op["model_name"], "Set_Diffusers_Pipeline_State_Dict"
            )
            state_dict = torch.load(ckpt_path, map_location='cpu')
            pipe.unet.load_state_dict(state_dict, strict=True)
            _try_enable_xformers(pipe)
            pipe = pipe.to(get_device())
        elif op["op"] == "unique3d_unet":
            _apply_unique3d_unet(pipe, op["config_name"])
        else:
            raise ValueError(f"unknown DIFFUSERS_PIPE op {op['op']!r}")

    return pipe


def resolve_diffusers_pipe(config):
    """DIFFUSERS_PIPE recipe -> live pipeline, cached across runs.

    Consumers call this on their pipe input before using it.

    A DIFFUSERS_PIPE is not always a recipe: ten loaders still return a live
    pipeline on the same socket, so any consumer can be handed either shape.
    Pass those straight back rather than freezing one into a cache key --
    freeze() would json.dumps(..., default=str) the object and cache it under
    its repr. That guard is what lets a consumer call this unconditionally.
    """
    if not isinstance(config, dict):
        return config
    kind = config.get("kind", "diffusers_pipe")
    builder = _PIPE_BUILDERS.get(kind)
    if builder is None:
        raise ValueError(
            f"DIFFUSERS_PIPE recipe of unknown kind {kind!r}; known kinds are "
            f"{sorted(_PIPE_BUILDERS)}. A loader emitted a recipe without "
            f"registering a builder for it.")
    # managed(), not resolve(): a pipeline's unet/vae/text_encoder each get a
    # ModelPatcher, so ComfyUI can evict them like any checkpoint. resolve()
    # would hold them in a plain dict that unload_all_models() cannot reach.
    # reloadable=True: this resolver runs on EVERY execution of every consumer,
    # so if the model has been offloaded to make room, the load_models_gpu()
    # inside managed() brings it straight back. That property is what makes it
    # safe for other models to evict this one. Loaders that hand a live object
    # to a consumer do not have it, and are left resident.
    return model_cache.managed(kind, config, builder, reloadable=True)


# Same dispatch under an honest name. _PIPE_BUILDERS is keyed by `kind`, not by
# socket type, so any socket carrying a recipe resolves through here -- an
# LGM_MODEL no more needs its own registry than a DIFFUSERS_PIPE does.
resolve_model_recipe = resolve_diffusers_pipe


class Set_Diffusers_Pipeline_Scheduler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": ("DIFFUSERS_PIPE",),
                "diffusers_scheduler_name": (list(DIFFUSERS_SCHEDULER_DICT.keys()),),
            },
        }

    RETURN_TYPES = (
        "DIFFUSERS_PIPE",
    )
    RETURN_NAMES = (
        "pipe",
    )
    FUNCTION = "set_pipe_scheduler"
    CATEGORY = "Comfy3D/Import|Export"

    def set_pipe_scheduler(self, pipe, diffusers_scheduler_name):
        # Appends to the recipe instead of mutating a live pipeline: the
        # scheduler swap is applied in _build_diffusers_pipe at load time.
        return (model_cache.with_op(pipe, "scheduler", name=diffusers_scheduler_name), )

class Set_Diffusers_Pipeline_State_Dict:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": ("DIFFUSERS_PIPE",),
                "repo_id": ("STRING", {"default": "TencentARC/InstantMesh", "multiline": False}),
                "model_name": ("STRING", {"default": "diffusion_pytorch_model.bin", "multiline": False}),
            },
        }

    RETURN_TYPES = (
        "DIFFUSERS_PIPE",
    )
    RETURN_NAMES = (
        "pipe",
    )
    FUNCTION = "set_pipe_state_dict"
    CATEGORY = "Comfy3D/Import|Export"

    def set_pipe_state_dict(self, pipe, repo_id, model_name):
        # Download now so a bad repo_id fails in this node rather than later
        # inside the consumer; the weights are applied at load time.
        checkpoints_dir_abs = os.path.join(CKPT_DIFFUSERS_PATH, repo_id)
        resume_or_download_model_from_hf(checkpoints_dir_abs, repo_id, model_name, self.__class__.__name__)

        return (model_cache.with_op(pipe, "state_dict", repo_id=repo_id, model_name=model_name), )

class Wonder3D_MVDiffusion_Model:
    
    config_path = "Wonder3D_config.yaml"
    fix_cam_pose_dir = "Wonder3D/data/fixed_poses/nine_views"

    # Derived, not assigned as an INPUT_TYPES side effect. Same trap as the
    # loaders: INPUT_TYPES need not have run in the process that executes the
    # node, and here it bit a *consumer* rather than a loader.
    @classmethod
    def config_file(cls):
        """Absolute path to Wonder3D_config.yaml."""
        return os.path.join(CONFIG_ROOT_PATH, cls.config_path)

    @classmethod
    def cam_pose_dir(cls):
        """Absolute path to the fixed nine-view camera poses."""
        return os.path.join(MODULE_ROOT_PATH, cls.fix_cam_pose_dir)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mvdiffusion_pipe": ("DIFFUSERS_PIPE",),
                "reference_image": ("IMAGE",),
                "reference_mask": ("MASK",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "mv_guidance_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "step": 0.01}),
                "num_inference_steps": ("INT", {"default": 50, "min": 1}),
            },
        }
        
    RETURN_TYPES = (
        "IMAGE",
        "IMAGE",
        "ORBIT_CAMPOSES",
    )
    RETURN_NAMES = (
        "multiview_images",
        "multiview_normals",
        "orbit_camposes",
    )
    FUNCTION = "run_mvdiffusion"
    CATEGORY = "Comfy3D/Algorithm"
    
    @torch.no_grad()
    def run_mvdiffusion(
        self, 
        mvdiffusion_pipe, 
        reference_image, # [1, H, W, 3]
        reference_mask,  # [1, H, W]
        seed,
        mv_guidance_scale, 
        num_inference_steps, 
    ):

        mvdiffusion_pipe = resolve_diffusers_pipe(mvdiffusion_pipe)

        cfg = load_config_wonder3d(self.config_file())

        batch = self.prepare_data(reference_image, reference_mask)

        mvdiffusion_pipe.set_progress_bar_config(disable=True)
        seed = int(seed)
        generator = torch.Generator(device=mvdiffusion_pipe.unet.device).manual_seed(seed)

        # repeat  (2B, Nv, 3, H, W)
        imgs_in = torch.cat([batch['imgs_in']] * 2, dim=0).to(WEIGHT_DTYPE)

        # (2B, Nv, Nce)
        camera_embeddings = torch.cat([batch['camera_embeddings']] * 2, dim=0).to(WEIGHT_DTYPE)

        task_embeddings = torch.cat([batch['normal_task_embeddings'], batch['color_task_embeddings']], dim=0).to(WEIGHT_DTYPE)

        camera_embeddings = torch.cat([camera_embeddings, task_embeddings], dim=-1).to(WEIGHT_DTYPE)

        # (B*Nv, 3, H, W)
        imgs_in = rearrange(imgs_in, "Nv C H W -> (Nv) C H W")
        # (B*Nv, Nce)
        # camera_embeddings = rearrange(camera_embeddings, "B Nv Nce -> (B Nv) Nce")

        out = mvdiffusion_pipe(
            imgs_in,
            # camera_embeddings,
            generator=generator,
            guidance_scale=mv_guidance_scale,
            num_inference_steps=num_inference_steps,
            output_type='pt',
            num_images_per_prompt=1,
            **cfg.pipe_validation_kwargs,
        ).images

        num_views = out.shape[0] // 2
        # [N, 3, H, W] -> [N, H, W, 3]
        mv_images = out[num_views:].permute(0, 2, 3, 1)
        mv_normals = out[:num_views].permute(0, 2, 3, 1)
        
        orbit_radius = [4.0] * 6
        orbit_center = [0.0] * 6
        orbit_elevations, orbit_azimuths = ORBITPOSE_PRESET_DICT["Wonder3D(6)"]
        orbit_camposes = compose_orbit_camposes(orbit_radius, orbit_elevations, orbit_azimuths, orbit_center, orbit_center, orbit_center)
    
        return (mv_images, mv_normals, orbit_camposes)
    
    def prepare_data(self, ref_image, ref_mask):
        single_image = torch_imgs_to_pils(ref_image, ref_mask)[0]
        dataset = MVSingleImageDataset(fix_cam_pose_dir=self.cam_pose_dir(), num_views=6, img_wh=[256, 256], bg_color='white', single_image=single_image)
        return dataset[0]

class MVDream_Model:
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mvdream_pipe": ("DIFFUSERS_PIPE",),
                "reference_image": ("IMAGE", ), 
                "reference_mask": ("MASK",),
                "prompt": ("STRING", {
                    "default": "",
                    "multiline": True
                }),
                "prompt_neg": ("STRING", {
                    "default": "ugly, blurry, pixelated obscure, unnatural colors, poor lighting, dull, unclear, cropped, lowres, low quality, artifacts, duplicate", 
                    "multiline": True
                }),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "mv_guidance_scale": ("FLOAT", {"default": 5.0, "min": 0.0, "step": 0.01}),
                "num_inference_steps": ("INT", {"default": 30, "min": 1}),
                "elevation": ("FLOAT", {"default": 0.0, "min": ELEVATION_MIN, "max": ELEVATION_MAX, "step": 0.0001}),
            },
        }
    
    RETURN_TYPES = (
        "IMAGE",
        "ORBIT_CAMPOSES",   
    )
    RETURN_NAMES = (
        "multiview_images",
        "orbit_camposes",
    )
    FUNCTION = "run_mvdream"
    CATEGORY = "Comfy3D/Algorithm"
    
    def run_mvdream(
        self, 
        mvdream_pipe, 
        reference_image, # [1, H, W, 3]
        reference_mask,  # [1, H, W]
        prompt, 
        prompt_neg, 
        seed,
        mv_guidance_scale, 
        num_inference_steps, 
        elevation,
    ):
        # A DIFFUSERS_PIPE may arrive as a recipe dict or as a live
        # pipeline; resolve_diffusers_pipe passes the latter through.
        mvdream_pipe = resolve_diffusers_pipe(mvdream_pipe)
        if len(reference_image.shape) == 4:
            reference_image = reference_image.squeeze(0)
        if len(reference_mask.shape) == 3:
            reference_mask = reference_mask.squeeze(0)
            
        generator = torch.manual_seed(seed)
            
        reference_mask = reference_mask.unsqueeze(2)
        # give the white background to reference_image
        reference_image = (reference_image * reference_mask + (1 - reference_mask)).detach().cpu().numpy()

        # generate multi-view images
        mv_images = mvdream_pipe(prompt, reference_image, generator=generator, negative_prompt=prompt_neg, guidance_scale=mv_guidance_scale, num_inference_steps=num_inference_steps, elevation=elevation)
        mv_images = torch.from_numpy(np.stack([mv_images[1], mv_images[2], mv_images[3], mv_images[0]], axis=0)).float() # [4, H, W, 3], float32

        orbit_radius = [4.0] * 4
        orbit_center = [0.0] * 4
        orbit_elevations, orbit_azimuths = ORBITPOSE_PRESET_DICT["MVDream(4)"]
        orbit_camposes = compose_orbit_camposes(orbit_radius, orbit_elevations, orbit_azimuths, orbit_center, orbit_center, orbit_center)

        return (mv_images, orbit_camposes)
    
class Load_Large_Multiview_Gaussian_Model:
    

    # Announces in the UI that this node fetches weights if they are absent.
    DISPLAY_NAME = "(Down)Load Large Multiview Gaussian Model"

    checkpoints_dir = "LGM"
    default_ckpt_name = "model_fp16.safetensors"
    default_repo_id = "ashawkey/LGM"
    
    @classmethod
    def ckpt_dir(cls):
        """Absolute checkpoints directory, derived on demand.

        A method, not a `cls.checkpoints_dir_abs` assigned as an INPUT_TYPES
        side effect: that only holds while the same node both defines and uses
        it, and INPUT_TYPES may never have run in the process that loads the
        model. See Load_Hunyuan3D_V1.svrm_config().
        """
        return os.path.join(CKPT_ROOT_PATH, cls.checkpoints_dir)

    @classmethod
    def INPUT_TYPES(cls):
        all_models_names = get_list_filenames(cls.ckpt_dir(), SUPPORTED_CHECKPOINTS_EXTENSIONS)
        if cls.default_ckpt_name not in all_models_names:
            all_models_names += [cls.default_ckpt_name]
            
        return {
            "required": {
                "model_name": (all_models_names, ),
                "lgb_config": (['big', 'default', 'small', 'tiny'], )
            },
        }
    
    RETURN_TYPES = (
        "LGM_MODEL",
    )
    RETURN_NAMES = (
        "lgm_model",
    )
    FUNCTION = "load_LGM"
    CATEGORY = "Comfy3D/Import|Export"
    
    def load_LGM(self, model_name, lgb_config):
        # Download now, build later. Returning the live model put a 528 MB
        # LargeMultiviewGaussianModel on the wire, and it does not even pickle:
        # __init__ assigns four lambdas (pos_act, scale_act, opacity_act,
        # rgb_act) as instance attributes, and a local lambda has no importable
        # qualname. The recipe is ~100 bytes of JSON instead.
        ckpt_path = resume_or_download_model_from_hf(self.ckpt_dir(), self.default_repo_id, model_name, self.__class__.__name__)
        cstr(f"[{self.__class__.__name__}] weights ready at {ckpt_path}").msg.print()
        return (model_cache.recipe("lgm", ckpt=ckpt_path, config=lgb_config), )


@_pipe_builder("lgm")
def _build_lgm(config):
    """Materialize an lgm recipe. Called only on a cache miss."""
    model = LargeMultiviewGaussianModel(config_defaults[config["config"]])
    ckpt_path = config["ckpt"]
    if ckpt_path.endswith('safetensors'):
        ckpt = load_file(ckpt_path, device='cpu')
    else:
        ckpt = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt, strict=False)
    # .half() on CPU: casting before the patcher sees the model means it
    # measures the size that will actually be resident.
    return model.half().eval()
    
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
    
class Large_Multiview_Gaussian_Model:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "multiview_images": ("IMAGE", ),
                "lgm_model": ("LGM_MODEL", ),
            },
        }
    
    OUTPUT_NODE = True
    RETURN_TYPES = (
        "GS_PLY",
    )
    RETURN_NAMES = (
        "gs_ply",
    )
    FUNCTION = "run_LGM"
    CATEGORY = "Comfy3D/Algorithm"
    
    @torch.no_grad()
    def run_LGM(self, multiview_images, lgm_model):
        lgm_model = resolve_model_recipe(lgm_model)
        ref_image_torch = prepare_torch_img(multiview_images, lgm_model.opt.input_size, lgm_model.opt.input_size, get_device_str()) # [4, 3, 256, 256]
        ref_image_torch = TF.normalize(ref_image_torch, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        rays_embeddings = lgm_model.prepare_default_rays(get_device_str())
        ref_image_torch = torch.cat([ref_image_torch, rays_embeddings], dim=1).unsqueeze(0) # [1, 4, 9, 256, 256]
        
        with torch.autocast(device_type=get_autocast_device_type(), dtype=WEIGHT_DTYPE):
            # generate gaussians
            gaussians = lgm_model.forward_gaussians(ref_image_torch)
        
        # convert gaussians to ply
        gs_ply = lgm_model.gs.to_ply(gaussians)
            
        return (gs_ply, )
    
class Convert_3DGS_to_Mesh_with_NeRF_and_Marching_Cubes:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "gs_ply": ("GS_PLY",),
                "gs_config": (['big', 'default', 'small', 'tiny'], ),
                "training_nerf_iterations": ("INT", {"default": 512, "min": 1, "max": 0xffffffffffffffff}),
                "training_nerf_resolution": ("INT", {"default": 128, "min": 1, "max": 0xffffffffffffffff}),
                "marching_cude_grids_resolution": ("INT", {"default": 256, "min": 1, "max": 0xffffffffffffffff}),
                "marching_cude_grids_batch_size": ("INT", {"default": 128, "min": 1, "max": 0xffffffffffffffff}),
                "marching_cude_threshold": ("FLOAT", {"default": 10.0, "min": 0.0, "step": 0.01}),
                "training_mesh_iterations": ("INT", {"default": 2048, "min": 1, "max": 0xffffffffffffffff}),
                "training_mesh_resolution": ("INT", {"default": 512, "min": 1, "max": 0xffffffffffffffff}),
                "remesh_after_n_iteration": ("INT", {"default": 512, "min": 128, "max": 100000}),
                "training_albedo_iterations": ("INT", {"default": 512, "min": 1, "max": 0xffffffffffffffff}),
                "training_albedo_resolution": ("INT", {"default": 512, "min": 1, "max": 0xffffffffffffffff}),
                "texture_resolution": ("INT", {"default": 1024, "min": 128, "max": 8192}),
                "force_cuda_rast": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = (
        "TRIMESH",
        "IMAGE",
        "MASK",
    )
    RETURN_NAMES = (
        "mesh",
        "imgs",
        "alphas",
    )
    FUNCTION = "convert_gs_ply"
    CATEGORY = "Comfy3D/Algorithm"
    
    def convert_gs_ply(
        self, 
        gs_ply, 
        gs_config, 
        training_nerf_iterations,
        training_nerf_resolution,
        marching_cude_grids_resolution,
        marching_cude_grids_batch_size,
        marching_cude_threshold,
        training_mesh_iterations,
        training_mesh_resolution,
        remesh_after_n_iteration,
        training_albedo_iterations,
        training_albedo_resolution,
        texture_resolution,
        force_cuda_rast,
    ):
        with torch.inference_mode(False):
            chosen_config = config_defaults[gs_config]
            chosen_config.force_cuda_rast = force_cuda_rast
            converter = GSConverterNeRFMarchingCubes(config_defaults[gs_config], gs_ply).cuda()
            imgs, alphas = converter.fit_nerf(training_nerf_iterations, training_nerf_resolution)
            converter.fit_mesh(
                training_mesh_iterations, remesh_after_n_iteration, training_mesh_resolution, 
                marching_cude_grids_resolution, marching_cude_grids_batch_size, marching_cude_threshold
            )
            converter.fit_mesh_uv(training_albedo_iterations, training_albedo_resolution, texture_resolution)
        
        return(_wire_out(converter.get_mesh()), imgs, alphas)
    
class Load_TripoSR_Model:

    # Announces in the UI that this node fetches weights if they are absent.
    DISPLAY_NAME = "(Down)Load TripoSR Model"

    checkpoints_dir = "TripoSR"
    default_ckpt_name = "model.ckpt"
    default_repo_id = "stabilityai/TripoSR"
    config_path = "TripoSR_config.yaml"
    
    @classmethod
    def ckpt_dir(cls):
        """Absolute checkpoints directory, derived on demand.

        A method, not a `cls.checkpoints_dir_abs` assigned as an INPUT_TYPES
        side effect: that only holds while the same node both defines and uses
        it, and INPUT_TYPES may never have run in the process that loads the
        model. See Load_Hunyuan3D_V1.svrm_config().
        """
        return os.path.join(CKPT_ROOT_PATH, cls.checkpoints_dir)

    @classmethod
    def config_file(cls):
        """Absolute path to the TripoSR config. Derived, not set in INPUT_TYPES."""
        return os.path.join(CONFIG_ROOT_PATH, cls.config_path)

    @classmethod
    def INPUT_TYPES(cls):
        all_models_names = get_list_filenames(cls.ckpt_dir(), SUPPORTED_CHECKPOINTS_EXTENSIONS)
        if cls.default_ckpt_name not in all_models_names:
            all_models_names += [cls.default_ckpt_name]
            
        return {
            "required": {
                "model_name": (all_models_names, ),
                "chunk_size": ("INT", {"default": 8192, "min": 1, "max": 10000})
            },
        }
    
    RETURN_TYPES = (
        "TSR_MODEL",
    )
    RETURN_NAMES = (
        "tsr_model",
    )
    FUNCTION = "load_TSR"
    CATEGORY = "Comfy3D/Import|Export"
    
    def load_TSR(self, model_name, chunk_size):
        
        ckpt_path = resume_or_download_model_from_hf(self.ckpt_dir(), self.default_repo_id, model_name, self.__class__.__name__)

        def build(_config):
            model = TSR.from_pretrained(
                weight_path=ckpt_path,
                config_path=self.config_file()
            )
            model.renderer.set_chunk_size(chunk_size)
            return model

        tsr_model = model_cache.managed(
            "triposr", {"ckpt": ckpt_path, "chunk_size": chunk_size}, build)

        cstr(f"[{self.__class__.__name__}] loaded model ckpt from {ckpt_path}").msg.print()
        
        return (tsr_model, )
    
class TripoSR:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tsr_model": ("TSR_MODEL", ),
                "reference_image": ("IMAGE",),
                "reference_mask": ("MASK",),
                "geometry_extract_resolution": ("INT", {"default": 256, "min": 1, "max": 0xffffffffffffffff}),
                "marching_cude_threshold": ("FLOAT", {"default": 25.0, "min": 0.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = (
        "TRIMESH",
    )
    RETURN_NAMES = (
        "mesh",
    )
    
    FUNCTION = "run_TSR"
    CATEGORY = "Comfy3D/Algorithm"

    @torch.no_grad()
    def run_TSR(self, tsr_model, reference_image, reference_mask, geometry_extract_resolution, marching_cude_threshold):
        mesh = None
        
        image = reference_image[0]
        mask = reference_mask[0].unsqueeze(2)
        image = torch.cat((image, mask), dim=2).detach().cpu().numpy()
        
        image = Image.fromarray(np.clip(255. * image, 0, 255).astype(np.uint8))
        image = self.fill_background(image)
        image = image.convert('RGB')
        
        scene_codes = tsr_model([image], get_device())
        meshes = tsr_model.extract_mesh(scene_codes, resolution=geometry_extract_resolution, threshold=marching_cude_threshold)
        mesh = Mesh.load_trimesh(given_mesh=meshes[0])

        return (_wire_out(mesh),)
    
    # Default model are trained on images with this background 
    def fill_background(self, image):
        image = np.array(image).astype(np.float32) / 255.0
        image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
        image = Image.fromarray((image * 255.0).astype(np.uint8))
        return image
    
class Load_SF3D_Model:

    # Announces in the UI that this node fetches weights if they are absent.
    DISPLAY_NAME = "(Down)Load SF3D Model"

    checkpoints_dir = "StableFast3D"
    default_ckpt_name = "model.safetensors"
    # Ungated source for the SF3D weights: model.safetensors (4,024,289,892 B)
    # and config.yaml (2,405 B), downloadable without a token.
    default_repo_id = "patrickbdevaney/stable-fast-3d"
    config_path = "StableFast3D_config.yaml"
    
    @classmethod
    def ckpt_dir(cls):
        """Absolute checkpoints directory, derived on demand.

        A method, not a `cls.checkpoints_dir_abs` assigned as an INPUT_TYPES
        side effect: that only holds while the same node both defines and uses
        it, and INPUT_TYPES may never have run in the process that loads the
        model. See Load_Hunyuan3D_V1.svrm_config().
        """
        return os.path.join(CKPT_ROOT_PATH, cls.checkpoints_dir)

    @classmethod
    def config_file(cls):
        """Absolute path to the StableFast3D config. Derived, not set in INPUT_TYPES."""
        return os.path.join(CONFIG_ROOT_PATH, cls.config_path)

    @classmethod
    def INPUT_TYPES(cls):
        all_models_names = get_list_filenames(cls.ckpt_dir(), SUPPORTED_CHECKPOINTS_EXTENSIONS)
        if cls.default_ckpt_name not in all_models_names:
            all_models_names += [cls.default_ckpt_name]
            
        return {
            "required": {
                "model_name": (all_models_names, ),
            },
        }
    
    RETURN_TYPES = (
        "SF3D_MODEL",
    )
    RETURN_NAMES = (
        "sf3d_model",
    )
    FUNCTION = "load_SF3D"
    CATEGORY = "Comfy3D/Import|Export"
    
    def load_SF3D(self, model_name):
        
        ckpt_path = resume_or_download_model_from_hf(self.ckpt_dir(), self.default_repo_id, model_name, self.__class__.__name__)

        def build(_config):
            # SF3D loads through safetensors' own load_model(), which passes
            # backend= to load_file. If any node has imported mmgp.offload by
            # now -- Hunyuan3D 2.1 texgen does -- load_file is mmgp's
            # replacement, which has no such parameter.
            from .shared_utils.mmgp_compat import keep_safetensors_compatible
            keep_safetensors_compatible()

            model = SF3D.from_pretrained(
                config_path=self.config_file(),
                weight_path=ckpt_path
            )
            return model.eval()

        sf3d_model = model_cache.managed("sf3d", {"ckpt": ckpt_path}, build)

        cstr(f"[{self.__class__.__name__}] loaded model ckpt from {ckpt_path}").msg.print()
        
        return (sf3d_model, )
    
class StableFast3D:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sf3d_model": ("SF3D_MODEL", ),
                "reference_image": ("IMAGE",),
                "reference_mask": ("MASK",),
                "texture_resolution": ("INT", {"default": 1024, "min": 128, "max": 8192}),
                "remesh_option": (["None", "Triangle"], ),
            }
        }

    RETURN_TYPES = (
        "TRIMESH",
    )
    RETURN_NAMES = (
        "mesh",
    )
    
    FUNCTION = "run_SF3D"
    CATEGORY = "Comfy3D/Algorithm"

    @torch.no_grad()
    def run_SF3D(self, sf3d_model, reference_image, reference_mask, texture_resolution, remesh_option):
        single_image = torch_imgs_to_pils(reference_image, reference_mask)[0]
        
        with torch.autocast(device_type=get_autocast_device_type(), dtype=WEIGHT_DTYPE):
            model_batch = self.create_batch(single_image)
            model_batch = {k: v.cuda() for k, v in model_batch.items()}
            trimesh_mesh, _ = sf3d_model.generate_mesh(
                model_batch, texture_resolution, remesh_option
            )
        mesh = Mesh.load_trimesh(given_mesh=trimesh_mesh[0])

        return (_wire_out(mesh),)
    
    # Default model are trained on images with this background 
    def create_batch(self, input_image: Image):
        COND_WIDTH = 512
        COND_HEIGHT = 512
        COND_DISTANCE = 1.6
        COND_FOVY_DEG = 40
        BACKGROUND_COLOR = [0.5, 0.5, 0.5]

        # Cached. Doesn't change
        c2w_cond = sf3d_utils.default_cond_c2w(COND_DISTANCE)
        intrinsic, intrinsic_normed_cond = sf3d_utils.create_intrinsic_from_fov_deg(
            COND_FOVY_DEG, COND_HEIGHT, COND_WIDTH
        )
        
        img_cond = (
            torch.from_numpy(
                np.asarray(input_image.resize((COND_WIDTH, COND_HEIGHT))).astype(np.float32)
                / 255.0
            )
            .float()
            .clip(0, 1)
        )
        mask_cond = img_cond[:, :, -1:]
        rgb_cond = torch.lerp(
            torch.tensor(BACKGROUND_COLOR)[None, None, :], img_cond[:, :, :3], mask_cond
        )

        batch_elem = {
            "rgb_cond": rgb_cond,
            "mask_cond": mask_cond,
            "c2w_cond": c2w_cond.unsqueeze(0),
            "intrinsic_cond": intrinsic.unsqueeze(0),
            "intrinsic_normed_cond": intrinsic_normed_cond.unsqueeze(0),
        }
        # Add batch dim
        batched = {k: v.unsqueeze(0) for k, v in batch_elem.items()}
        return batched
    
class Load_CRM_MVDiffusion_Model:

    # Announces in the UI that this node fetches weights if they are absent.
    DISPLAY_NAME = "(Down)Load CRM MVDiffusion Model"

    checkpoints_dir = "CRM"
    default_ckpt_name = ["pixel-diffusion.pth", "ccm-diffusion.pth"]
    default_conf_name = ["sd_v2_base_ipmv_zero_SNR.yaml", "sd_v2_base_ipmv_chin8_zero_snr.yaml"]
    default_repo_id = "Zhengyi/CRM"
    config_path = "CRM_configs"
    
    @classmethod
    def ckpt_dir(cls):
        """Absolute checkpoints directory, derived on demand.

        A method, not a `cls.checkpoints_dir_abs` assigned as an INPUT_TYPES
        side effect: that only holds while the same node both defines and uses
        it, and INPUT_TYPES may never have run in the process that loads the
        model. See Load_Hunyuan3D_V1.svrm_config().
        """
        return os.path.join(CKPT_ROOT_PATH, cls.checkpoints_dir)

    @classmethod
    def config_dir(cls):
        """Absolute path to the CRM config directory. Derived, not set in INPUT_TYPES."""
        return os.path.join(CONFIG_ROOT_PATH, cls.config_path)

    @classmethod
    def INPUT_TYPES(cls):
        all_models_names = get_list_filenames(cls.ckpt_dir(), SUPPORTED_CHECKPOINTS_EXTENSIONS)
        for ckpt_name in cls.default_ckpt_name:
            if ckpt_name not in all_models_names:
                all_models_names += [ckpt_name]
            
        return {
            "required": {
                "model_name": (all_models_names, ),
                "crm_config_path": (cls.default_conf_name, ),
            },
        }
    
    # CRM_MVDiffusion_to_Unique3D.json loads RealESRGAN_x4plus.pth through the
    # native UpscaleModelLoader, and models/upscale_models/ ships empty -- so
    # the graph failed on a file the user had to find themselves. Same model
    # and same GitHub release as Load_CRM_T2I_V3_Models fetches.
    UPSCALER_NAME = "RealESRGAN_x4plus.pth"
    UPSCALER_URL = ("https://github.com/xinntao/Real-ESRGAN/releases/download/"
                    "v0.1.0/RealESRGAN_x4plus.pth")

    # upscale_model_name is APPENDED so slot 0 keeps its index in saved graphs.
    # "*" (ComfyUI's Any type), not "COMBO". A native loader's model_name input
    # is typed as its OPTION LIST, not the string "COMBO", and
    # comfy_execution.validation.validate_node_input compares the two directly:
    #     received_type(COMBO) mismatch input_type(['RealESRGAN_x4plus.pth'])
    # Only "*" short-circuits that check, so it is the one output type that
    # will actually connect to a combo widget.
    RETURN_TYPES = (
        "CRM_MVDIFFUSION_SAMPLER",
        "*",
    )
    RETURN_NAMES = (
        "crm_mvdiffusion_sampler",
        "upscale_model_name",
    )
    FUNCTION = "load_CRM"
    CATEGORY = "Comfy3D/Import|Export"
    
    def load_CRM(self, model_name, crm_config_path):
        """Download, and describe what to build. Do NOT build it.

        This used to instantiate a 2B-parameter model here and then construct
        the sampler -- which runs a CLIP forward pass to cache the negative
        prompt embedding. A node called "(Down)Load" was doing inference, every
        failure in the model or the text encoder surfaced only after a 6 GB
        download, and the live sampler then had to cross the worker/host
        boundary twice on CRM_MVDIFFUSION_SAMPLER, where only trimesh has a
        serializer -- so a 2B-parameter object went through generic pickle.

        Downloading stays here, so this is still the node that reports a
        network failure. The consumers call resolve_crm_sampler().
        """
        crm_config_path = os.path.join(self.config_dir(), crm_config_path)
        ckpt_path = resume_or_download_model_from_hf(
            self.ckpt_dir(), self.default_repo_id, model_name, self.__class__.__name__)
        upscale_model_name = _ensure_upscale_model(self.UPSCALER_URL, self.UPSCALER_NAME)

        cstr(f"[{self.__class__.__name__}] ckpt ready at {ckpt_path}").msg.print()

        return (model_cache.recipe(
            "crm_mvdiffusion",
            ckpt=ckpt_path,
            config=crm_config_path,
        ), upscale_model_name, )
    
def _build_crm_mvdiffusion(config):
    """Materialize a crm_mvdiffusion recipe. Only on a cache miss."""
    from .Gen_3D_Modules.CRM.imagedream.ldm.util import (
        instantiate_from_config, get_obj_from_str)

    crm_config = OmegaConf.load(config["config"])

    model = instantiate_from_config(crm_config.model)
    model.load_state_dict(torch.load(config["ckpt"], map_location="cpu"), strict=False)
    # Placed on the device here, not left to model_cache.managed() as the
    # other builders do: the sampler below encodes the negative prompt in its
    # constructor, and that forward pass needs real weights on the same device
    # as the tokens it makes. managed() still wraps the result afterwards, so
    # eviction works from the next execution on.
    model = model.to(get_device()).to(WEIGHT_DTYPE)
    model.device = get_device()

    # The sampler encodes the negative prompt at construction, so it has to be
    # built where the model is -- which is exactly why it does not belong in
    # the loader.
    return get_obj_from_str(crm_config.sampler.target)(
        model, device=get_device(), dtype=WEIGHT_DTYPE, **crm_config.sampler.params)


def resolve_crm_sampler(config):
    """CRM_MVDIFFUSION_SAMPLER recipe -> live sampler, cached across runs.

    Non-dicts pass through, as with resolve_diffusers_pipe, so consumers can
    call this unconditionally and a live object is never frozen into a key.
    """
    if not isinstance(config, dict):
        return config
    return model_cache.managed(
        "crm_mvdiffusion", config, _build_crm_mvdiffusion, reloadable=True)


class CRM_Images_MVDiffusion_Model:
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "crm_mvdiffusion_sampler": ("CRM_MVDIFFUSION_SAMPLER",),
                "reference_image": ("IMAGE",),
                "reference_mask": ("MASK",),
                "prompt": ("STRING", {
                    "default": "3D assets",
                    "multiline": True
                }),
                "prompt_neg": ("STRING", {
                    "default": "uniform low no texture ugly, boring, bad anatomy, blurry, pixelated,  obscure, unnatural colors, poor lighting, dull, and unclear.", 
                    "multiline": True
                }),
                "seed": ("INT", {"default": 1234, "min": 0, "max": 0xffffffffffffffff}),
                "mv_guidance_scale": ("FLOAT", {"default": 5.5, "min": 0.0, "step": 0.01}),
                "num_inference_steps": ("INT", {"default": 50, "min": 1}),
                
            },
        }
    
    RETURN_TYPES = (
        "IMAGE",
        "ORBIT_CAMPOSES",   
    )
    RETURN_NAMES = (
        "multiview_images",
        "orbit_camposes",
    )
    FUNCTION = "run_model"
    CATEGORY = "Comfy3D/Algorithm"
    
    def run_model(
        self, 
        crm_mvdiffusion_sampler, 
        reference_image, # [1, H, W, 3]
        reference_mask,  # [1, H, W]
        prompt, 
        prompt_neg, 
        seed,
        mv_guidance_scale, 
        num_inference_steps, 
    ):
        # A CRM_MVDIFFUSION_SAMPLER arrives as a recipe; resolve_crm_sampler
        # passes a live sampler straight through.
        crm_mvdiffusion_sampler = resolve_crm_sampler(crm_mvdiffusion_sampler)

        pixel_img = torch_imgs_to_pils(reference_image, reference_mask)[0]
        pixel_img = CRMSampler.process_pixel_img(pixel_img)
        
        multiview_images = CRMSampler.stage1_sample(
            crm_mvdiffusion_sampler,
            pixel_img,
            prompt,
            prompt_neg,
            seed,
            mv_guidance_scale, 
            num_inference_steps
        ).to(dtype=reference_image.dtype, device=reference_image.device)

        orbit_radius = [4.0] * 6
        orbit_center = [0.0] * 6
        orbit_elevations, orbit_azimuths = ORBITPOSE_PRESET_DICT["CRM(6)"]
        orbit_camposes = compose_orbit_camposes(orbit_radius, orbit_elevations, orbit_azimuths, orbit_center, orbit_center, orbit_center)
        
        return (multiview_images, orbit_camposes)
    
class CRM_CCMs_MVDiffusion_Model:
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "crm_mvdiffusion_sampler": ("CRM_MVDIFFUSION_SAMPLER",),
                "reference_image": ("IMAGE",),
                "reference_mask": ("MASK",),
                "multiview_images": ("IMAGE",),
                "prompt": ("STRING", {
                    "default": "3D assets",
                    "multiline": True
                }),
                "prompt_neg": ("STRING", {
                    "default": "uniform low no texture ugly, boring, bad anatomy, blurry, pixelated,  obscure, unnatural colors, poor lighting, dull, and unclear.", 
                    "multiline": True
                }),
                "seed": ("INT", {"default": 1234, "min": 0, "max": 0xffffffffffffffff}),
                "mv_guidance_scale": ("FLOAT", {"default": 5.5, "min": 0.0, "step": 0.01}),
                "num_inference_steps": ("INT", {"default": 50, "min": 1}),
            },
        }
    
    RETURN_TYPES = (
        "IMAGE",
    )
    RETURN_NAMES = (
        "multiview_CCMs",
    )
    FUNCTION = "run_model"
    CATEGORY = "Comfy3D/Algorithm"
    
    @torch.no_grad()
    def run_model(
        self, 
        crm_mvdiffusion_sampler, 
        reference_image, # [1, H, W, 3]
        reference_mask,  # [1, H, W]
        multiview_images, # [6, H, W, 3]
        prompt, 
        prompt_neg, 
        seed,
        mv_guidance_scale, 
        num_inference_steps, 
    ):
        # A CRM_MVDIFFUSION_SAMPLER arrives as a recipe; resolve_crm_sampler
        # passes a live sampler straight through.
        crm_mvdiffusion_sampler = resolve_crm_sampler(crm_mvdiffusion_sampler)

        pixel_img = torch_imgs_to_pils(reference_image, reference_mask)[0]
        pixel_img = CRMSampler.process_pixel_img(pixel_img)
        
        multiview_CCMs = CRMSampler.stage2_sample(
            crm_mvdiffusion_sampler,
            pixel_img,
            multiview_images,
            prompt,
            prompt_neg,
            seed,
            mv_guidance_scale, 
            num_inference_steps
        )
        
        return(multiview_CCMs, )
    
class Load_Convolutional_Reconstruction_Model:

    # Announces in the UI that this node fetches weights if they are absent.
    DISPLAY_NAME = "(Down)Load Convolutional Reconstruction Model"

    checkpoints_dir = "CRM"
    default_ckpt_name = "CRM.pth"
    default_repo_id = "Zhengyi/CRM"
    config_path = "CRM_configs/specs_objaverse_total.json"
    
    @classmethod
    def ckpt_dir(cls):
        """Absolute checkpoints directory.

        Derived on demand, not assigned as an INPUT_TYPES side effect:
        INPUT_TYPES need not have run in the process that executes
        this node. See Load_Hunyuan3D_V1.svrm_config().
        """
        return os.path.join(CKPT_ROOT_PATH, cls.checkpoints_dir)

    @classmethod
    def config_file(cls):
        """Absolute path to this node's config file.

        Derived on demand, not assigned as an INPUT_TYPES side effect:
        INPUT_TYPES need not have run in the process that executes
        this node. See Load_Hunyuan3D_V1.svrm_config().
        """
        return os.path.join(CONFIG_ROOT_PATH, cls.config_path)

    @classmethod
    def INPUT_TYPES(cls):
        all_models_names = get_list_filenames(cls.ckpt_dir(), SUPPORTED_CHECKPOINTS_EXTENSIONS)
        if cls.default_ckpt_name not in all_models_names:
            all_models_names += [cls.default_ckpt_name]
            
        return {
            "required": {
                "model_name": (all_models_names, ),
            },
        }
    
    RETURN_TYPES = (
        "CRM_MODEL",
    )
    RETURN_NAMES = (
        "crm_model",
    )
    FUNCTION = "load_CRM"
    CATEGORY = "Comfy3D/Import|Export"
    
    def load_CRM(self, model_name):
        """Download, and describe what to build. Do NOT build it.

        ConvolutionalReconstructionModel holds nvdiffrast state
        (RasterizeCRStateWrapper), a C++ object, so returning it live killed
        the worker at the transport:

            cannot pickle '_nvdiffrast_c.RasterizeCRStateWrapper' object

        Same as instantmesh_lrm and crm_mvdiffusion: emit a recipe and let the
        consumer resolve it inside the worker, where the model already lives.
        """
        ckpt_path = resume_or_download_model_from_hf(
            self.ckpt_dir(), self.default_repo_id, model_name, self.__class__.__name__)

        cstr(f"[{self.__class__.__name__}] ckpt ready at {ckpt_path}").msg.print()

        return (model_cache.recipe(
            "crm_recon",
            ckpt=ckpt_path,
            config=self.config_file(),
        ), )
    
def _build_crm_recon(config):
    """Materialize a crm_recon recipe. Only on a cache miss."""
    crm_conf = json.load(open(config["config"]))
    model = ConvolutionalReconstructionModel(crm_conf)
    model.load_state_dict(torch.load(config["ckpt"], map_location="cpu"), strict=False)
    return model


def resolve_crm_model(config):
    """CRM_MODEL recipe -> live model, cached across runs.

    Non-dicts pass through, so consumers can call this unconditionally without
    freezing a live model into a cache key.
    """
    if not isinstance(config, dict):
        return config
    return model_cache.managed("crm_recon", config, _build_crm_recon, reloadable=True)


class Convolutional_Reconstruction_Model:
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "crm_model": ("CRM_MODEL", ),
                "multiview_images": ("IMAGE",),
                "multiview_CCMs": ("IMAGE",),
            }
        }

    RETURN_TYPES = (
        "TRIMESH",
    )
    RETURN_NAMES = (
        "mesh",
    )
    
    FUNCTION = "run_CRM"
    CATEGORY = "Comfy3D/Algorithm"
    
    @torch.no_grad()
    def run_CRM(self, crm_model, multiview_images, multiview_CCMs):

        # A CRM_MODEL arrives as a recipe; resolve_crm_model passes a live
        # model straight through.
        crm_model = resolve_crm_model(crm_model)

        np_imgs = np.concatenate(multiview_images.cpu().numpy(), 1) # (256, 256*6==1536, 3)
        np_xyzs = np.concatenate(multiview_CCMs.cpu().numpy(), 1) # (256, 1536, 3)
        
        mesh = CRMSampler.generate3d(crm_model, np_imgs, np_xyzs, get_device())
        
        return (_wire_out(mesh),)
    
class Zero123Plus_Diffusion_Model:
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "zero123plus_pipe": ("DIFFUSERS_PIPE",),
                "reference_image": ("IMAGE",),
                "reference_mask": ("MASK",),
                "seed": ("INT", {"default": 1234, "min": 0, "max": 0xffffffffffffffff}),
                "guidance_scale": ("FLOAT", {"default": 4.0, "min": 0.0, "step": 0.01}),
                "num_inference_steps": ("INT", {"default": 28, "min": 1}),
            },
        }
    
    RETURN_TYPES = (
        "IMAGE",
        "ORBIT_CAMPOSES",   
    )
    RETURN_NAMES = (
        "multiviews",
        "orbit_camposes",
    )
    FUNCTION = "run_model"
    CATEGORY = "Comfy3D/Algorithm"
    
    @torch.no_grad()
    def run_model(
        self,
        zero123plus_pipe,
        reference_image,
        reference_mask,
        seed,
        guidance_scale,
        num_inference_steps,
    ):
        
        # A DIFFUSERS_PIPE may arrive as a recipe dict or as a live
        # pipeline; resolve_diffusers_pipe passes the latter through.
        zero123plus_pipe = resolve_diffusers_pipe(zero123plus_pipe)
        single_image = torch_imgs_to_pils(reference_image, reference_mask)[0]

        seed = int(seed)
        generator = torch.Generator(device=zero123plus_pipe.unet.device).manual_seed(seed)

        # sampling
        output_image = zero123plus_pipe(
            single_image, 
            generator=generator,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps, 
        ).images[0]

        multiview_images = np.asarray(output_image, dtype=np.float32) / 255.0
        multiview_images = torch.from_numpy(multiview_images).permute(2, 0, 1).contiguous()     # (3, 960, 640)
        multiview_images = rearrange(multiview_images, 'c (n h) (m w) -> (n m) h w c', n=3, m=2)        # (6, 320, 320, 3)
        multiview_images = multiview_images.to(dtype=reference_image.dtype, device=reference_image.device)

        orbit_radius = [4.0] * 6
        orbit_center = [0.0] * 6
        orbit_elevations, orbit_azimuths = ORBITPOSE_PRESET_DICT["Zero123Plus(6)"]
        orbit_camposes = compose_orbit_camposes(orbit_radius, orbit_elevations, orbit_azimuths, orbit_center, orbit_center, orbit_center)

        return (multiview_images, orbit_camposes)
    
class Load_InstantMesh_Reconstruction_Model:

    # Announces in the UI that this node fetches weights if they are absent.
    DISPLAY_NAME = "(Down)Load InstantMesh Reconstruction Model"

    checkpoints_dir = "InstantMesh"
    default_ckpt_names = ["instant_mesh_large.ckpt", "instant_mesh_base.ckpt", "instant_nerf_large.ckpt", "instant_nerf_base.ckpt"]
    default_repo_id = "TencentARC/InstantMesh"
    config_root_dir = "InstantMesh_configs"
    
    @classmethod
    def ckpt_dir(cls):
        """Absolute checkpoints directory.

        Derived on demand, not assigned as an INPUT_TYPES side effect:
        INPUT_TYPES need not have run in the process that executes
        this node. See Load_Hunyuan3D_V1.svrm_config().
        """
        return os.path.join(CKPT_ROOT_PATH, cls.checkpoints_dir)

    @classmethod
    def config_dir(cls):
        """Absolute config directory.

        Derived on demand, not assigned as an INPUT_TYPES side effect:
        INPUT_TYPES need not have run in the process that executes
        this node. See Load_Hunyuan3D_V1.svrm_config().
        """
        return os.path.join(CONFIG_ROOT_PATH, cls.config_root_dir)

    @classmethod
    def INPUT_TYPES(cls):
        all_models_names = get_list_filenames(cls.ckpt_dir(), SUPPORTED_CHECKPOINTS_EXTENSIONS)
        for ckpt_name in cls.default_ckpt_names:
            if ckpt_name not in all_models_names:
                all_models_names += [ckpt_name]
                
        return {
            "required": {
                "model_name": (all_models_names, ),
            },
        }
    
    RETURN_TYPES = (
        "LRM_MODEL",
    )
    RETURN_NAMES = (
        "lrm_model",
    )
    FUNCTION = "load_LRM"
    CATEGORY = "Comfy3D/Import|Export"
    
    def load_LRM(self, model_name):

        from .Gen_3D_Modules.InstantMesh.utils.train_util import instantiate_from_config

        is_flexicubes = True if model_name.startswith('instant_mesh') else False
        
        config_name = model_name.split(".")[0] + ".yaml"
        config_path = os.path.join(self.config_dir(), config_name)
        config = OmegaConf.load(config_path)

        ckpt_path = resume_or_download_model_from_hf(self.ckpt_dir(), self.default_repo_id, model_name, self.__class__.__name__)
        _instantmesh_safetensors(ckpt_path)

        # Emit a recipe, not the model. The LRM holds nvdiffrast state
        # (RasterizeCRStateWrapper) once FlexiCubes is initialised, and that is
        # a C++ object: returning the live model made the worker die at the
        # transport with "cannot pickle '_nvdiffrast_c.RasterizeCRStateWrapper'
        # object". The recipe is a ~100-byte dict; the consumer resolves it
        # inside the worker, where the model already lives.
        cstr(f"[{self.__class__.__name__}] using ckpt {ckpt_path}").msg.print()

        return (model_cache.recipe(
            "instantmesh_lrm",
            model_name=model_name,
            ckpt=ckpt_path,
            config=config_path,
            flexicubes=is_flexicubes,
        ), )
    
class InstantMesh_Reconstruction_Model:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "lrm_model": ("LRM_MODEL", ),
                "multiview_images": ("IMAGE",),
                "orbit_camera_poses": ("ORBIT_CAMPOSES",),    # [orbit radius, elevation, azimuth, orbit center X,  orbit center Y,  orbit center Z]
                "orbit_camera_fovy": ("FLOAT", {"default": 30.0, "min": 0.0, "max": 180.0, "step": 0.1}),
                "texture_resolution": ("INT", {"default": 1024, "min": 128, "max": 8192}),
            }
        }

    RETURN_TYPES = (
        "TRIMESH",
    )
    RETURN_NAMES = (
        "mesh",
    )
    
    FUNCTION = "run_LRM"
    CATEGORY = "Comfy3D/Algorithm"
    
    @torch.no_grad()
    def run_LRM(self, lrm_model, multiview_images, orbit_camera_poses, orbit_camera_fovy, texture_resolution):

        # An LRM_MODEL arrives as a recipe; resolve_lrm_model passes a live
        # model straight through.
        lrm_model = resolve_lrm_model(lrm_model)

        images = multiview_images.permute(0, 3, 1, 2).unsqueeze(0).to(get_device())   # [N, H, W, 3] -> [1, N, 3, H, W]
        images = v2.functional.resize(images, 320, interpolation=3, antialias=True).clamp(0, 1)

        # convert camera format from orbit to lrm inputs
        azimuths, elevations, radius = [], [], []
        for i in range(len(orbit_camera_poses)):
            azimuths.append(orbit_camera_poses[i][2])
            elevations.append(orbit_camera_poses[i][1])
            radius.append(orbit_camera_poses[i][0])
        input_cameras = oribt_camera_poses_to_input_cameras(azimuths, elevations, radius=radius, fov=orbit_camera_fovy).to(get_device())

        # get triplane
        planes = lrm_model.forward_planes(images, input_cameras)

        # get mesh
        mesh_out = lrm_model.extract_mesh(
            planes,
            use_texture_map=True,
            texture_resolution=texture_resolution,
        )

        vertices, faces, uvs, mesh_tex_idx, tex_map = mesh_out
        tex_map = troch_image_dilate(tex_map.permute(1, 2, 0))  # [3, H, W] -> [H, W, 3]
        
        mesh = Mesh(v=vertices, f=faces, vt=uvs, ft=mesh_tex_idx, albedo=tex_map, device=get_device())
        mesh.auto_normal()
        return (_wire_out(mesh),)

class Era3D_MVDiffusion_Model:
    
    config_path = "Era3D_config.yaml"
    @classmethod
    def config_file(cls):
        """Absolute path to this node's config file.

        Derived on demand, not assigned as an INPUT_TYPES side effect:
        INPUT_TYPES need not have run in the process that executes
        this node. See Load_Hunyuan3D_V1.svrm_config().
        """
        return os.path.join(CONFIG_ROOT_PATH, cls.config_path)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "era3d_pipe": ("DIFFUSERS_PIPE",),
                "reference_image": ("IMAGE",),
                "reference_mask": ("MASK",),
                "image_crop_size": ("INT", {"default": 420, "min": 400, "max": 8192}),
                "seed": ("INT", {"default": 600, "min": 0, "max": 0xffffffffffffffff}),
                "guidance_scale": ("FLOAT", {"default": 3.0, "min": 0.0, "step": 0.01}),
                "num_inference_steps": ("INT", {"default": 40, "min": 1}),
                "eta": ("FLOAT", {"default": 1.0, "min": 0.0, "step": 0.01}),
                "radius": ("FLOAT", {"default": 4.0, "min": 0.1, "step": 0.01}),
            },
        }
    
    RETURN_TYPES = (
        "IMAGE",
        "IMAGE",
        "ORBIT_CAMPOSES",   
    )
    RETURN_NAMES = (
        "multiviews",
        "multiview_normals",
        "orbit_camposes",
    )
    FUNCTION = "run_model"
    CATEGORY = "Comfy3D/Algorithm"
    
    @torch.no_grad()
    def run_model(
        self,
        era3d_pipe,
        reference_image,
        reference_mask,
        image_crop_size,
        seed,
        guidance_scale,
        num_inference_steps,
        eta,
        radius,
    ):
        # A DIFFUSERS_PIPE may arrive as a recipe dict or as a live
        # pipeline; resolve_diffusers_pipe passes the latter through.
        era3d_pipe = resolve_diffusers_pipe(era3d_pipe)
        cfg = load_config_era3d(self.config_file())
        
        single_image = torch_imgs_to_pils(reference_image, reference_mask)[0]

        # Get the dataset
        cfg.dataset.prompt_embeds_path = os.path.join(NODES_PATH, cfg.dataset.prompt_embeds_path)
        dataset = Era3DSingleImageDataset(
            single_image=single_image,
            crop_size=image_crop_size,
            dtype=WEIGHT_DTYPE,
            **cfg.dataset
        )

        # Get input data
        img_batch = dataset.__getitem__(0)

        imgs_in = torch.cat([img_batch['imgs_in']]*2, dim=0).to(get_device(), dtype=WEIGHT_DTYPE)    # (B*Nv, 3, H, W) B==1
        #num_views = imgs_in.shape[1]

        normal_prompt_embeddings, clr_prompt_embeddings = img_batch['normal_prompt_embeddings'], img_batch['color_prompt_embeddings'] 
        prompt_embeddings = torch.cat([normal_prompt_embeddings, clr_prompt_embeddings], dim=0).to(get_device(), dtype=WEIGHT_DTYPE)    # (B*Nv, N, C) B==1

        generator = torch.Generator(device=era3d_pipe.unet.device).manual_seed(seed)

        # sampling
        with torch.autocast(get_autocast_device_type()):
            unet_out = era3d_pipe(
                imgs_in, None, prompt_embeds=prompt_embeddings,
                generator=generator, guidance_scale=guidance_scale, output_type='pt', num_images_per_prompt=1, 
                num_inference_steps=num_inference_steps, eta=eta
            )
        
        out = unet_out.images
        bsz = out.shape[0] // 2

        # (1, 3, 512, 512)
        normals_pred = out[:bsz]    
        images_pred = out[bsz:] 
        
        # [N, 3, H, W] -> [N, H, W, 3]
        multiview_images = images_pred.permute(0, 2, 3, 1).to(reference_image.device, dtype=reference_image.dtype)   
        multiview_normals = normals_pred.permute(0, 2, 3, 1).to(reference_image.device, dtype=reference_image.dtype)
        
        azimuths = [0, 45, 90, 180, -90, -45]
        elevations = [0.0] * 6
        radius = [radius] * 6
        center = [0.0] * 6

        orbit_camposes = [azimuths, elevations, radius, center, center, center]

        return (multiview_images, multiview_normals, orbit_camposes)
    
class Instant_NGP:
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reference_image": ("IMAGE",),
                "reference_mask": ("MASK",),
                "reference_orbit_camera_poses": ("ORBIT_CAMPOSES",),    # [orbit radius, elevation, azimuth, orbit center X,  orbit center Y,  orbit center Z]
                "reference_orbit_camera_fovy": ("FLOAT", {"default": 49.1, "min": 0.0, "max": 180.0, "step": 0.1}),
                "training_iterations": ("INT", {"default": 512, "min": 1, "max": 0xffffffffffffffff}),
                "training_resolution": ("INT", {"default": 128, "min": 128, "max": 8192}),
                "marching_cude_grids_resolution": ("INT", {"default": 256, "min": 1, "max": 0xffffffffffffffff}),
                "marching_cude_grids_batch_size": ("INT", {"default": 128, "min": 1, "max": 0xffffffffffffffff}),
                "marching_cude_threshold": ("FLOAT", {"default": 10.0, "min": 0.0, "step": 0.01}),
                "texture_resolution": ("INT", {"default": 1024, "min": 128, "max": 8192}),
                "background_color": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "force_cuda_rast": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = (
        "TRIMESH",
    )
    RETURN_NAMES = (
        "mesh",
    )
    FUNCTION = "run_instant_ngp"
    CATEGORY = "Comfy3D/Algorithm"
    
    def run_instant_ngp(
        self, 
        reference_image, 
        reference_mask, 
        reference_orbit_camera_poses, 
        reference_orbit_camera_fovy,
        training_iterations,
        training_resolution,
        marching_cude_grids_resolution,
        marching_cude_grids_batch_size,
        marching_cude_threshold,
        texture_resolution,
        background_color,
        force_cuda_rast
    ):
        with torch.inference_mode(False):
            
            ngp = InstantNGP(training_resolution).to(get_device())
            ngp.prepare_training(reference_image, reference_mask, reference_orbit_camera_poses, reference_orbit_camera_fovy)
            ngp.fit_nerf(training_iterations, background_color)
            
            vertices, triangles = marching_cubes_density_to_mesh(ngp.get_density, marching_cude_grids_resolution, marching_cude_grids_batch_size, marching_cude_threshold)

            v = torch.from_numpy(vertices).contiguous().float().to(get_device())
            f = torch.from_numpy(triangles).contiguous().int().to(get_device())

            mesh = Mesh(v=v, f=f, device=get_device())
            mesh.auto_normal()
            mesh.auto_uv()
            
            mesh.albedo = color_func_to_albedo(mesh, ngp.get_color, texture_resolution, device=get_device(), force_cuda_rast=force_cuda_rast)
            
        return (_wire_out(mesh), )
        
class FlexiCubes_MVS:
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reference_depth_maps": ("IMAGE",),
                "reference_masks": ("MASK",),
                "reference_orbit_camera_poses": ("ORBIT_CAMPOSES",),    # [orbit radius, elevation, azimuth, orbit center X,  orbit center Y,  orbit center Z]
                "reference_orbit_camera_fovy": ("FLOAT", {"default": 49.1, "min": 0.0, "max": 180.0, "step": 0.1}),
                "training_iterations": ("INT", {"default": 512, "min": 1, "max": 0xffffffffffffffff}),
                "batch_size": ("INT", {"default": 4, "min": 1, "max": 0xffffffffffffffff}),
                "learning_rate": ("FLOAT", {"default": 0.01, "min": 0.001, "step": 0.001}),
                "voxel_grids_resolution": ("INT", {"default": 128, "min": 1, "max": 0xffffffffffffffff}),
                "depth_min_distance": ("FLOAT", {"default": 0.5, "min": 0.0, "step": 0.01}),
                "depth_max_distance": ("FLOAT", {"default": 5.5, "min": 0.0, "step": 0.01}),
                "mask_loss_weight": ("FLOAT", {"default": 1.0, "min": 0.01, "step": 0.01}),
                "depth_loss_weight": ("FLOAT", {"default": 100.0, "min": 0.01, "step": 0.01}),
                "normal_loss_weight": ("FLOAT", {"default": 1.0, "min": 0.01, "step": 0.01}),
                "sdf_regularizer_weight": ("FLOAT", {"default": 0.2, "min": 0.01, "step": 0.01}),
                "remove_floaters_weight": ("FLOAT", {"default": 0.5, "min": 0.01, "step": 0.01}),
                "cube_stabilizer_weight": ("FLOAT", {"default": 0.1, "min": 0.01, "step": 0.01}),
                "force_cuda_rast": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "reference_normal_maps": ("IMAGE",), 
            }
        }

    RETURN_TYPES = (
        "TRIMESH",
    )
    RETURN_NAMES = (
        "mesh",
    )
    FUNCTION = "run_flexicubes"
    CATEGORY = "Comfy3D/Algorithm"
    
    def run_flexicubes(
        self,
        reference_depth_maps,
        reference_masks,
        reference_orbit_camera_poses,
        reference_orbit_camera_fovy,
        training_iterations,
        batch_size,
        learning_rate,
        voxel_grids_resolution,
        depth_min_distance,
        depth_max_distance,
        mask_loss_weight,
        depth_loss_weight,
        normal_loss_weight,
        sdf_regularizer_weight,
        remove_floaters_weight,
        cube_stabilizer_weight,
        force_cuda_rast,
        reference_normal_maps=None
    ):
        
        with torch.inference_mode(False):
            
            fc_trainer = FlexiCubesTrainer(
                training_iterations,
                batch_size,
                learning_rate,
                voxel_grids_resolution,
                depth_min_distance,
                depth_max_distance,
                mask_loss_weight,
                depth_loss_weight,
                normal_loss_weight,
                sdf_regularizer_weight,
                remove_floaters_weight,
                cube_stabilizer_weight,
                force_cuda_rast,
                device=get_device()
            )
            
            fc_trainer.prepare_training(reference_depth_maps, reference_masks, reference_orbit_camera_poses, reference_orbit_camera_fovy, reference_normal_maps)
            
            fc_trainer.training()
            
            mesh = fc_trainer.get_mesh()
            
            return (_wire_out(mesh), )

class Load_Stable_Diffusion_15_Assets:
    """Fetch the stock SD1.5-era models the Unique3D upscale graphs need.

    Unique3D's upscale stages are built from NATIVE ComfyUI nodes --
    CheckpointLoaderSimple, CLIPVisionLoader, UpscaleModelLoader -- each of
    which takes a combo of whatever happens to be in the corresponding models/
    folder. On a fresh install those folders are empty, so the graph fails
    validation before a single node runs, naming a file the user has no obvious
    way to obtain.

    This node downloads them and returns their NAMES, so each output can feed
    the matching native loader's widget. The link is doing real work: it orders
    execution, so the loader cannot run before the download finishes.

    It is a separate node, not extra outputs on a Unique3D loader, precisely so
    it only runs when something is wired to it -- these three files are ~4.7 GB
    and nobody generating a mesh should pay for them.

    All three are safetensors or a plain .pth; nothing here pulls a pickle.
    """

    DISPLAY_NAME = "(Down)Load SD1.5 Assets (checkpoint / CLIP-Vision / upscaler)"

    CATEGORY = "Comfy3D/Import|Export"
    # "*" (ComfyUI's Any type), not "COMBO". A native loader's model_name input
    # is typed as its OPTION LIST, not the string "COMBO", and
    # comfy_execution.validation.validate_node_input compares the two directly:
    #     received_type(COMBO) mismatch input_type(['RealESRGAN_x4plus.pth'])
    # Only "*" short-circuits that check, so it is the one output type that
    # will actually connect to a combo widget.
    RETURN_TYPES = ("*", "*", "*")
    RETURN_NAMES = ("ckpt_name", "clip_name", "upscale_model_name")
    FUNCTION = "download"

    # fp16 rather than the 4.3 GB fp32 original: same weights, half the disk,
    # and the graphs run it in fp16 regardless.
    CKPT_REPO = "Comfy-Org/stable-diffusion-v1-5-archive"
    CKPT_FILE = "v1-5-pruned-emaonly-fp16.safetensors"

    # The OpenCLIP ViT-H image encoder IPAdapter expects. h94/IP-Adapter is the
    # IPAdapter authors' own repo; laion's copy no longer serves this filename.
    CLIP_REPO = "h94/IP-Adapter"
    CLIP_REPO_FILE = "models/image_encoder/model.safetensors"
    CLIP_FILE = "CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors"

    UPSCALER_URL = ("https://github.com/xinntao/Real-ESRGAN/releases/download/"
                    "v0.1.0/RealESRGAN_x4plus.pth")
    UPSCALER_FILE = "RealESRGAN_x4plus.pth"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    def download(self):
        ckpt = _ensure_comfy_model("checkpoints", self.CKPT_FILE,
                                   repo_id=self.CKPT_REPO, repo_file=self.CKPT_FILE)
        clip = _ensure_comfy_model("clip_vision", self.CLIP_FILE,
                                   repo_id=self.CLIP_REPO, repo_file=self.CLIP_REPO_FILE)
        upscaler = _ensure_comfy_model("upscale_models", self.UPSCALER_FILE,
                                       url=self.UPSCALER_URL)

        cstr(f"[{self.__class__.__name__}] ready: {ckpt}, {clip}, {upscaler}").msg.print()
        return (ckpt, clip, upscaler)


class Load_Stable_Zero123_Checkpoint:
    """Fetch stable_zero123.ckpt for the native ImageOnlyCheckpointLoader.

    Using_Stack_Orbit_Camera_Poses_to_conditioning_StableZero123 loads this
    through ComfyUI's own "Load Checkpoint Image Only" node, whose widget is a
    combo over models/checkpoints/ -- empty on a fresh install, so the graph
    fails validation naming a file the user has to go and find.

    Wiring the output orders execution as well as filling the widget, so the
    loader cannot run before the file is there.

    8.6 GB, hence its own node rather than extra outputs on something else:
    nobody who is not running StableZero123 should pay for it.
    """

    DISPLAY_NAME = "(Down)Load Stable Zero123 Checkpoint"

    CATEGORY = "Comfy3D/Import|Export"
    # "*", ComfyUI's Any type: ImageOnlyCheckpointLoader declares ckpt_name as
    # its OPTION LIST, which neither "COMBO" nor "STRING" will connect to.
    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("ckpt_name",)
    FUNCTION = "download"

    # stabilityai/stable-zero123 is NOT gated, unlike stable-fast-3d -- it
    # serves unauthenticated (checked: 200, 8,584,287,851 bytes) -- so this
    # takes the publisher's own weights rather than a mirror.
    REPO_ID = "stabilityai/stable-zero123"
    CKPT_FILE = "stable_zero123.ckpt"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    def download(self):
        name = _ensure_comfy_model("checkpoints", self.CKPT_FILE,
                                   repo_id=self.REPO_ID, repo_file=self.CKPT_FILE)
        cstr(f"[{self.__class__.__name__}] ready: {name}").msg.print()
        return (name, )


class Load_RealESRGAN_x4plus:
    """Fetch RealESRGAN_x4plus.pth and hand its name to UpscaleModelLoader.

    Several shipped graphs load this through the native UpscaleModelLoader,
    whose widget is a combo over models/upscale_models/ -- which ships empty,
    so the graph fails validation naming a file with no hint where to get it.

    One output, nothing else. The bigger asset downloaders in this pack pull
    several GB because their graphs need a checkpoint and a CLIP-Vision model
    too; this is 67 MB and is all you need when the upscaler is the only thing
    missing.

    Wiring the output does real work beyond filling the widget: it ORDERS
    execution, so UpscaleModelLoader cannot run before the file exists.
    """

    DISPLAY_NAME = "(Down)Load RealESRGAN x4plus"

    CATEGORY = "Comfy3D/Import|Export"
    # "*", ComfyUI's Any type. A native loader's model_name is typed as its
    # OPTION LIST, and validate_node_input compares the two directly, so
    # "COMBO" and "STRING" both fail to connect -- only "*" short-circuits it.
    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("upscale_model_name",)
    FUNCTION = "download"

    UPSCALER_URL = Load_Stable_Diffusion_15_Assets.UPSCALER_URL
    UPSCALER_FILE = Load_Stable_Diffusion_15_Assets.UPSCALER_FILE

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    def download(self):
        name = _ensure_upscale_model(self.UPSCALER_URL, self.UPSCALER_FILE)
        cstr(f"[{self.__class__.__name__}] ready: {name}").msg.print()
        return (name, )


class Load_Unique3D_MV_Upscale_Assets:
    """Fetch every stock model the Unique3D MV-RGB upscale graph needs.

    Unique3D_1_MV_RGB_Upscale is built almost entirely from NATIVE ComfyUI
    nodes -- CheckpointLoaderSimple, CLIPVisionLoader, UpscaleModelLoader,
    ControlNetLoader, IPAdapterModelLoader -- and each takes a combo of
    whatever happens to sit in the matching models/ folder. On a fresh install
    all five are empty, so the graph fails validation before a single node
    runs, naming five files with no hint where they come from.

    This returns their NAMES, one output per loader. The links do real work:
    they order execution, so no loader can run before its file has landed.

    Superset of Load_SD1.5_Assets (same first three outputs, same files, so
    they share a download) plus the two that graph does not need: the Unique3D
    tile ControlNet and the SD1.5 IPAdapter.

    ~6.2 GB in total, which is why this is its own node rather than extra
    outputs on a Unique3D loader -- it only runs when something is wired to it.
    Everything here is safetensors or a plain .pth; nothing pulls a pickle.
    """

    DISPLAY_NAME = "(Down)Load Unique3D MV Upscale Assets"

    CATEGORY = "Comfy3D/Import|Export"
    # "*" (ComfyUI's Any type), not "COMBO". A native loader's model_name input
    # is typed as its OPTION LIST, not the string "COMBO", and
    # comfy_execution.validation.validate_node_input compares the two directly:
    #     received_type(COMBO) mismatch input_type(['RealESRGAN_x4plus.pth'])
    # Only "*" short-circuits that check, so it is the one output type that
    # will actually connect to a combo widget.
    RETURN_TYPES = ("*", "*", "*", "*", "*")
    RETURN_NAMES = ("ckpt_name", "clip_name", "upscale_model_name",
                    "control_net_name", "ipadapter_file")
    FUNCTION = "download"

    # Shared with Load_Stable_Diffusion_15_Assets, so wiring both costs one
    # download rather than two.
    CKPT_REPO = Load_Stable_Diffusion_15_Assets.CKPT_REPO
    CKPT_FILE = Load_Stable_Diffusion_15_Assets.CKPT_FILE
    CLIP_REPO = Load_Stable_Diffusion_15_Assets.CLIP_REPO
    CLIP_REPO_FILE = Load_Stable_Diffusion_15_Assets.CLIP_REPO_FILE
    CLIP_FILE = Load_Stable_Diffusion_15_Assets.CLIP_FILE
    UPSCALER_URL = Load_Stable_Diffusion_15_Assets.UPSCALER_URL
    UPSCALER_FILE = Load_Stable_Diffusion_15_Assets.UPSCALER_FILE

    # Unique3D's own tile ControlNet, in safetensors. Kept under the filename
    # the shipped workflow asks for so an untouched graph resolves it.
    CONTROLNET_REPO = "camenduru/Unique3D"
    CONTROLNET_REPO_FILE = "controlnet-tile/diffusion_pytorch_model.safetensors"
    CONTROLNET_FILE = "control_unique3d_sd15_tile.safetensors"

    # The IPAdapter authors' own repo.
    IPADAPTER_REPO = "h94/IP-Adapter"
    IPADAPTER_REPO_FILE = "models/ip-adapter_sd15.safetensors"
    IPADAPTER_FILE = "ip-adapter_sd15.safetensors"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    def download(self):
        ckpt = _ensure_comfy_model("checkpoints", self.CKPT_FILE,
                                   repo_id=self.CKPT_REPO, repo_file=self.CKPT_FILE)
        clip = _ensure_comfy_model("clip_vision", self.CLIP_FILE,
                                   repo_id=self.CLIP_REPO, repo_file=self.CLIP_REPO_FILE)
        upscaler = _ensure_comfy_model("upscale_models", self.UPSCALER_FILE,
                                       url=self.UPSCALER_URL)
        controlnet = _ensure_comfy_model("controlnet", self.CONTROLNET_FILE,
                                         repo_id=self.CONTROLNET_REPO,
                                         repo_file=self.CONTROLNET_REPO_FILE)
        ipadapter = _ensure_comfy_model("ipadapter", self.IPADAPTER_FILE,
                                        repo_id=self.IPADAPTER_REPO,
                                        repo_file=self.IPADAPTER_REPO_FILE)

        cstr(f"[{self.__class__.__name__}] ready: {ckpt}, {clip}, {upscaler}, "
             f"{controlnet}, {ipadapter}").msg.print()
        return (ckpt, clip, upscaler, controlnet, ipadapter)


class Load_Unique3D_Custom_UNet:
    default_repo_id = "MrForExample/Unique3D"
    config_root_dir = "Unique3D_configs"

    # Derived, not INPUT_TYPES side effects. These are read by the module-level
    # _apply_unique3d_unet(), which _build_diffusers_pipe replays on op replay --
    # in a process where this node's INPUT_TYPES may never have been called.
    @classmethod
    def ckpt_dir(cls):
        """Absolute Unique3D checkpoints directory."""
        return os.path.join(CKPT_DIFFUSERS_PATH, cls.default_repo_id)

    @classmethod
    def config_dir(cls):
        """Absolute Unique3D config directory."""
        return os.path.join(CONFIG_ROOT_PATH, cls.config_root_dir)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": ("DIFFUSERS_PIPE",),
                "config_name": (["image2mvimage", "image2normal"],),
            },
        }
    
    RETURN_TYPES = (
        "DIFFUSERS_PIPE",
    )
    RETURN_NAMES = (
        "pipe",
    )
    FUNCTION = "load_diffusers_unet"
    CATEGORY = "Comfy3D/Import|Export"
    
    def load_diffusers_unet(self, pipe, config_name):
        # A mutator, like Set_Diffusers_Pipeline_Scheduler: append to the recipe
        # rather than building the pipeline here just to swap its unet. The op
        # is replayed in _build_diffusers_pipe, in graph order.
        if isinstance(pipe, dict):
            return (model_cache.with_op(pipe, "unique3d_unet",
                                        config_name=config_name), )
        # Live pipeline (a loader that has not been converted yet): mutate now.
        _apply_unique3d_unet(pipe, config_name)
        return (pipe, )


def _apply_unique3d_unet(pipe, config_name):
    """Swap in Unique3D's custom UNet. Shared by the node and the op replay."""
    cls = Load_Unique3D_Custom_UNet
    from .Gen_3D_Modules.Unique3D.custum_3d_diffusion.trainings.config_classes import ExprimentConfig
    from .Gen_3D_Modules.Unique3D.custum_3d_diffusion.custum_modules.unifield_processor import AttnConfig, ConfigurableUNet2DConditionModel
    from .Gen_3D_Modules.Unique3D.custum_3d_diffusion.trainings.utils import load_config

    cfg_path = os.path.join(cls.config_dir(), config_name + ".yaml")
    checkpoint_dir_path = os.path.join(cls.ckpt_dir(), config_name)
    checkpoint_path = os.path.join(checkpoint_dir_path, "unet_state_dict.pth")

    cfg: ExprimentConfig = load_config(ExprimentConfig, cfg_path)
    if cfg.init_config.init_unet_path == "":
        cfg.init_config.init_unet_path = checkpoint_dir_path
    init_config: AttnConfig = load_config(AttnConfig, cfg.init_config)
    configurable_unet = ConfigurableUNet2DConditionModel(init_config, WEIGHT_DTYPE)
    _try_enable_xformers(configurable_unet)

    state_dict = torch.load(checkpoint_path)
    configurable_unet.unet.load_state_dict(state_dict, strict=False)
    configurable_unet.unet.to(get_device(), dtype=WEIGHT_DTYPE)

    pipe.unet = configurable_unet.unet
    cstr(f"[Load_Unique3D_Custom_UNet] loaded unet ckpt from {checkpoint_path}").msg.print()
    return pipe
    
class Unique3D_MVDiffusion_Model:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "unique3d_pipe": ("DIFFUSERS_PIPE",),
                "reference_image": ("IMAGE",),
                "seed": ("INT", {"default": 1145, "min": 0, "max": 0xffffffffffffffff}),
                "guidance_scale": ("FLOAT", {"default": 1.5, "min": 0.0, "step": 0.01}),
                "num_inference_steps": ("INT", {"default": 30, "min": 1}),
                "image_resolution": ([256, 512],),
                "radius": ("FLOAT", {"default": 4.0, "min": 0.1, "step": 0.01}),
                "preprocess_images":  ("BOOLEAN", {"default": True},),
            },
        }
    
    RETURN_TYPES = (
        "IMAGE",
        "ORBIT_CAMPOSES",   
    )
    RETURN_NAMES = (
        "multiviews",
        "orbit_camposes",
    )
    FUNCTION = "run_model"
    CATEGORY = "Comfy3D/Algorithm"
    
    @torch.no_grad()
    def run_model(
        self,
        unique3d_pipe,
        reference_image,    # Need to have white background
        seed,
        guidance_scale,
        num_inference_steps,
        image_resolution,
        radius,
        preprocess_images,
    ):
        # A DIFFUSERS_PIPE may arrive as a recipe dict or as a live
        # pipeline; resolve_diffusers_pipe passes the latter through.
        unique3d_pipe = resolve_diffusers_pipe(unique3d_pipe)
        from .Gen_3D_Modules.Unique3D.scripts.utils import simple_image_preprocess

        pil_image_list = torch_imgs_to_pils(reference_image)
        for i in range(len(pil_image_list)):
            if preprocess_images:
                pil_image_list[i] = simple_image_preprocess(pil_image_list[i])

        pil_image_list = pils_rgba_to_rgb(pil_image_list, bkgd="WHITE")

        generator = torch.Generator(device=unique3d_pipe.unet.device).manual_seed(seed)

        image_pils = unique3d_pipe(
            image=pil_image_list,
            generator=generator,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            width=image_resolution,
            height=image_resolution,
            height_cond=image_resolution,
            width_cond=image_resolution,
        ).images

        # [N, H, W, 3]
        multiview_images = pils_to_torch_imgs(image_pils, reference_image.dtype, reference_image.device)

        orbit_radius = [radius] * 4
        orbit_center = [0.0] * 4
        orbit_elevations, orbit_azimuths = ORBITPOSE_PRESET_DICT["Unique3D(4)"]
        orbit_camposes = compose_orbit_camposes(orbit_radius, orbit_elevations, orbit_azimuths, orbit_center, orbit_center, orbit_center)

        return (multiview_images, orbit_camposes)

class Fast_Normal_Maps_To_Mesh:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "front_side_back_normal_maps": ("IMAGE",),
                "front_side_back_normal_masks": ("MASK",),
            },
        }

    RETURN_TYPES = (
        "TRIMESH",
    )
    RETURN_NAMES = (
        "mesh",
    )
    FUNCTION = "run_fast_recon"
    CATEGORY = "Comfy3D/Algorithm"

    def run_fast_recon(self, front_side_back_normal_maps, front_side_back_normal_masks):
        pil_normal_list = torch_imgs_to_pils(front_side_back_normal_maps, front_side_back_normal_masks)
        meshes = fast_geo(pil_normal_list[0], pil_normal_list[2], pil_normal_list[1])
        vertices, faces, _ = from_py3d_mesh(meshes)

        mesh = Mesh(v=vertices, f=faces, device=get_device())
        return (_wire_out(mesh),)

class ExplicitTarget_Mesh_Optimization:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("TRIMESH",),
                "normal_maps": ("IMAGE",),
                "normal_masks": ("MASK",),
                "reconstruction_steps": ("INT", {"default": 200, "min": 0, "max": 0xffffffffffffffff}),
                "coarse_reconstruct_resolution": ("INT", {"default": 512, "min": 128, "max": 8192}),
                "loss_expansion_weight": ("FLOAT", {"default": 0.1, "min": 0.01, "step": 0.01}),
                "refinement_steps": ("INT", {"default": 100, "min": 0, "max": 0xffffffffffffffff}),
                "target_warmup_update_num": ("INT", {"default": 5, "min": 1, "max": 0xffffffffffffffff}),
                "target_update_interval": ("INT", {"default": 20, "min": 1, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "normal_orbit_camera_poses": ("ORBIT_CAMPOSES",),    # [orbit radius, elevation, azimuth, orbit center X,  orbit center Y,  orbit center Z]
            }
        }

    RETURN_TYPES = (
        "TRIMESH",
    )
    RETURN_NAMES = (
        "mesh",
    )
    FUNCTION = "run_ET_mesh_optimization"
    CATEGORY = "Comfy3D/Algorithm"

    def run_ET_mesh_optimization(
        self, 
        mesh,
        normal_maps, 
        normal_masks,  
        reconstruction_steps,
        coarse_reconstruct_resolution,
        loss_expansion_weight,
        refinement_steps, 
        target_warmup_update_num,
        target_update_interval,
        normal_orbit_camera_poses=None,
    ):
        #TODO For now only support four orthographic view with elevation equals zero
        #azimuths, elevations, radius = normal_orbit_camera_poses[0], normal_orbit_camera_poses[1], normal_orbit_camera_poses[2]
        mesh = _wire_in(mesh)
        pil_normal_list = torch_imgs_to_pils(normal_maps, normal_masks)
        normal_stg1 = [img.resize((coarse_reconstruct_resolution, coarse_reconstruct_resolution)) for img in pil_normal_list]
        with torch.inference_mode(False):
            vertices, faces = mesh.v.detach().clone().to(get_device()), mesh.f.detach().clone().to(get_device()).type(torch.int64)
            if reconstruction_steps > 0:
                vertices, faces = reconstruct_stage1(normal_stg1, steps=reconstruction_steps, vertices=vertices, faces=faces, loss_expansion_weight=loss_expansion_weight)
                
            if refinement_steps > 0:
                vertices, faces = run_mesh_refine(vertices, faces, pil_normal_list, steps=refinement_steps, update_normal_interval=target_update_interval, update_warmup=target_warmup_update_num, )

            mesh = Mesh(v=vertices, f=faces, device=get_device())

        return (_wire_out(mesh),)

class ExplicitTarget_Color_Projection:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("TRIMESH",),
                "reference_images": ("IMAGE",), 
                "reference_masks": ("MASK",),
                "projection_resolution": ("INT", {"default": 1024, "min": 128, "max": 8192}),
                "complete_unseen_rgb":  ("BOOLEAN", {"default": True},),
                "render_orbit_camera_fovy": ("FLOAT", {"default": 47.5, "min": 0.0, "max": 180.0, "step": 0.1}),
                "projection_weights": ("STRING", {"default": "2.0, 0.2, 1.0, 0.2"}),
                "confidence_threshold": ("FLOAT", {"default": 0.02, "min": 0.001, "max": 1.0, "step": 0.001}),
                "texture_projecton":  ("BOOLEAN", {"default": False},),
                "texture_type":  (["Albedo", "Metallic_and_Roughness"],),
            },
            "optional": {
                "reference_orbit_camera_poses": ("ORBIT_CAMPOSES",),    # [orbit radius, elevation, azimuth, orbit center X,  orbit center Y,  orbit center Z]
            }
        }

    RETURN_TYPES = (
        "TRIMESH",
    )
    RETURN_NAMES = (
        "mesh",
    )
    FUNCTION = "run_color_projection"
    CATEGORY = "Comfy3D/Algorithm"

    def run_color_projection(
        self, 
        mesh, 
        reference_images, 
        reference_masks,  
        projection_resolution, 
        complete_unseen_rgb,
        render_orbit_camera_fovy,
        projection_weights,
        confidence_threshold,
        texture_projecton,
        texture_type,
        reference_orbit_camera_poses=None,
    ):
        mesh = _wire_in(mesh)
        pil_image_list = torch_imgs_to_pils(reference_images, reference_masks)

        meshes = to_py3d_mesh(mesh.v, mesh.f)

        #TODO Convert camera format, currently only support elevation equal to zero
        if reference_orbit_camera_poses is None:
            img_num = len(reference_images)
            interval = 360 / img_num
            angle = 0
            azimuths = []
            for i in range(0, img_num):
                azimuths.append(angle)
                angle += interval
                
            cam_list = get_cameras_list(azimuths, get_device(), focal=1)
        else:
            #reference_orbit_camera_poses[0] = [360 + angle if angle < 0 else angle for angle in reference_orbit_camera_poses[0]]
            cam_list = get_orbit_cameras_list(reference_orbit_camera_poses, get_device(), render_orbit_camera_fovy)
        
        weights = projection_weights.split(",")
        if len(weights) == len(cam_list):
            weights = [float(item) for item in weights]
        else:
            weights = None
        
        if texture_projecton:
            target_img = multiview_color_projection_texture(meshes, mesh, pil_image_list, weights=weights, resolution=projection_resolution, device=get_device(), complete_unseen=complete_unseen_rgb, confidence_threshold=confidence_threshold, cameras_list=cam_list)
            target_img = troch_image_dilate(target_img)
            
            if texture_type == "Albedo":
                mesh.albedo = target_img
            elif texture_type == "Metallic_and_Roughness":
                mesh.metallicRoughness = target_img
            else:
                cstr(f"[{self.__class__.__name__}] Unknow texture type: {texture_type}").error.print()
        else:
            new_meshes = multiview_color_projection(meshes, pil_image_list, weights=weights, resolution=projection_resolution, device=get_device(), complete_unseen=complete_unseen_rgb, confidence_threshold=confidence_threshold, cameras_list=cam_list)
            vertices, faces, vertex_colors = from_py3d_mesh(new_meshes)

            mesh = Mesh(v=vertices, f=faces, 
                        vn=None if mesh.vn is None else mesh.vn.clone(), fn=None if mesh.fn is None else mesh.fn.clone(), 
                        vt=None if mesh.vt is None else mesh.vt.clone(), ft=None if mesh.ft is None else mesh.ft.clone(), 
                        vc=vertex_colors, device=get_device())
            if mesh.vn is None:
                mesh.auto_normal()
                
        return (_wire_out(mesh),)
    
class Convert_Vertex_Color_To_Texture:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("TRIMESH",),
                "texture_resolution": ("INT", {"default": 1024, "min": 128, "max": 8192}),
                "batch_size": ("INT", {"default": 128, "min": 1, "max": 0xffffffffffffffff}),
            },
        }

    RETURN_TYPES = (
        "TRIMESH",
    )
    RETURN_NAMES = (
        "mesh",
    )
    FUNCTION = "run_convert_func"
    CATEGORY = "Comfy3D/Algorithm"
    
    def run_convert_func(self, mesh, texture_resolution, batch_size):

        mesh = _wire_in(mesh)
        name = self.__class__.__name__

        if mesh.vc is None:
            cstr(f"[{name}] skip this node since there is no vertex color found in mesh").msg.print()
            return (_wire_out(mesh),)

        # UV unwrap, when the mesh arrives without one, is xatlas -- single
        # threaded, minutes on a dense mesh, and silent. Reconstruction output
        # has no UVs, so this is the common path and worth announcing before it
        # starts rather than explaining afterwards.
        if mesh.vt is None:
            cstr(f"[{name}] mesh has no UVs; unwrapping {len(mesh.f)} faces with xatlas "
                 f"(no progress available, can take minutes)").msg.print()
            t_uv = time.perf_counter()
            mesh.auto_uv()
            cstr(f"[{name}] UV unwrap done in {time.perf_counter() - t_uv:.1f}s").msg.print()

        tiles_per_side = math.ceil((texture_resolution - 1) / batch_size)
        cstr(f"[{name}] baking vertex color into a {texture_resolution}x{texture_resolution} "
             f"texture over {tiles_per_side}x{tiles_per_side} tiles").msg.print()

        pbar = _comfy_progress_bar(tiles_per_side ** 2)
        t_bake = time.perf_counter()
        state = {"next_log": 10}

        def on_tile(done, total):
            if pbar is not None:
                pbar.update_absolute(done, total)
            percent = done * 100 // total
            # Throttled to ten lines: at 8192/128 this loop runs 4096 times and
            # a line per tile would bury the rest of the run's log.
            if percent >= state["next_log"] or done == total:
                elapsed = time.perf_counter() - t_bake
                rate = done / elapsed if elapsed > 0 else 0
                eta = (total - done) / rate if rate > 0 else 0
                cstr(f"[{name}] baking {percent}% ({done}/{total} tiles, "
                     f"{elapsed:.1f}s elapsed, ~{eta:.0f}s left)").msg.print()
                state["next_log"] = percent - percent % 10 + 10

        albedo_img, _ = interpolate_texture_map_attr(
            mesh, texture_resolution, batch_size, interpolate_color=True,
            progress_callback=on_tile,
        )

        cstr(f"[{name}] dilating texture to bleed colour past UV seams").msg.print()
        mesh.albedo = troch_image_dilate(albedo_img)
        cstr(f"[{name}] done in {time.perf_counter() - t_bake:.1f}s").msg.print()

        return (_wire_out(mesh),)
    
class Load_CharacterGen_MVDiffusion_Model:

    # Announces in the UI that this node fetches weights if they are absent.
    DISPLAY_NAME = "(Down)Load CharacterGen MVDiffusion Model"

    checkpoints_dir = "CharacterGen"
    default_repo_id = "zjpshadow/CharacterGen"
    config_path = "CharacterGen_configs/Stage_2D_infer.yaml"
    
    @classmethod
    def ckpt_dir(cls):
        """Absolute checkpoints directory.

        Derived on demand, not assigned as an INPUT_TYPES side effect:
        INPUT_TYPES need not have run in the process that executes
        this node. See Load_Hunyuan3D_V1.svrm_config().
        """
        return os.path.join(CKPT_ROOT_PATH, cls.checkpoints_dir)

    @classmethod
    def config_dir(cls):
        """Absolute config directory.

        Derived on demand, not assigned as an INPUT_TYPES side effect:
        INPUT_TYPES need not have run in the process that executes
        this node. See Load_Hunyuan3D_V1.svrm_config().
        """
        return os.path.join(CONFIG_ROOT_PATH, cls.config_path)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
            },
        }
    
    RETURN_TYPES = (
        "CHARACTER_MV_GEN_PIPE",
    )
    RETURN_NAMES = (
        "character_mv_gen_pipe",
    )
    FUNCTION = "load_model"
    CATEGORY = "Comfy3D/Import|Export"
    
    # CharacterGen builds its 2D stage on top of a full SD 2.1 pipeline -- it
    # pulls tokenizer, text_encoder, vae, unet (twice) and scheduler from
    # `pretrained_model_path` (character_inference.py:106-138).
    #
    # The config shipped by upstream names stabilityai/stable-diffusion-2-1,
    # which Stability has since GATED: an anonymous request now gets 401
    # RepositoryNotFound, and so do -base and -unclip. That is not a transient
    # failure and no retry fixes it, so the node downloaded 14 GB of CharacterGen
    # and then died fetching a tokenizer.
    #
    # nlightcho/stable-diffusion-2-1 is an ungated mirror of the same pipeline,
    # verified against the SD 2.1 fingerprint: unet cross_attention_dim 1024,
    # use_linear_projection true, sample_size 96, block_out_channels
    # [320, 640, 1280, 1280]; text encoder CLIPTextModel 1024x23. It is also
    # safetensors-only, so nothing here pulls a pickle.
    SD21_REPO_ID = "nlightcho/stable-diffusion-2-1"

    @classmethod
    def sd21_dir(cls):
        """Local SD 2.1 base, downloaded on first use."""
        return os.path.join(CKPT_DIFFUSERS_PATH, cls.SD21_REPO_ID)

    def load_model(self):
        # Download checkpoints
        download_repo(self.default_repo_id, self.ckpt_dir(), ignore_patterns=HF_DOWNLOAD_IGNORE)

        # ...and the SD 2.1 base it is built on, to a LOCAL path. Passing a repo
        # id would send diffusers back to the hub on every load; passing a
        # directory keeps this node working offline and off a gated repo.
        download_repo(self.SD21_REPO_ID, self.sd21_dir(), self.SD21_REPO_ID,
                      ignore_patterns=["*.bin", "*.ckpt", "*.png", "*.md"])

        config = OmegaConf.load(self.config_dir())
        config.pretrained_model_path = self.sd21_dir()

        # Load pre-trained models
        character_mv_gen_pipe = Inference2D_API(checkpoint_root_path=self.ckpt_dir(), **config)

        cstr(f"[{self.__class__.__name__}] loaded model ckpt from {self.ckpt_dir()}").msg.print()
        return (character_mv_gen_pipe,)
    
class CharacterGen_MVDiffusion_Model:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "character_mv_gen_pipe": ("CHARACTER_MV_GEN_PIPE",),
                "reference_image": ("IMAGE", ),
                "reference_mask": ("MASK",),
                "target_image_width": ("INT", {"default": 512, "min": 128, "max": 8192}),
                "target_image_height": ("INT", {"default": 768, "min": 128, "max": 8192}),
                "seed": ("INT", {"default": 2333, "min": 0, "max": 0xffffffffffffffff}),
                "guidance_scale": ("FLOAT", {"default": 5.0, "min": 0.0, "step": 0.01}),
                "num_inference_steps": ("INT", {"default": 40, "min": 1}),
                "prompt": ("STRING", {
                    "default": "high quality, best quality",
                    "multiline": True
                }),
                "prompt_neg": ("STRING", {
                    "default": "", 
                    "multiline": True
                }),
                "radius": ("FLOAT", {"default": 1.5, "min": 0.1, "step": 0.01})
            },
        }
    
    RETURN_TYPES = (
        "IMAGE",
        "ORBIT_CAMPOSES",   
    )
    RETURN_NAMES = (
        "multiviews",
        "orbit_camposes",
    )
    FUNCTION = "run_model"
    CATEGORY = "Comfy3D/Algorithm"
    
    @torch.no_grad()
    def run_model(
        self,
        character_mv_gen_pipe,
        reference_image,
        reference_mask,
        target_image_width,
        target_image_height,
        seed,
        guidance_scale,
        num_inference_steps,
        prompt,
        prompt_neg,
        radius
    ):
        single_image = torch_imgs_to_pils(reference_image, reference_mask)[0]

        multiview_images = character_mv_gen_pipe.inference(
            single_image, target_image_width, target_image_height, prompt=prompt, prompt_neg=prompt_neg, 
            guidance_scale=guidance_scale, num_inference_steps=num_inference_steps, seed=seed
        ).to(dtype=reference_image.dtype, device=reference_image.device)
        
        orbit_radius = [radius] * 4
        orbit_center = [0.0] * 4
        orbit_elevations, orbit_azimuths = ORBITPOSE_PRESET_DICT["CharacterGen(4)"]
        orbit_camposes = compose_orbit_camposes(orbit_radius, orbit_elevations, orbit_azimuths, orbit_center, orbit_center, orbit_center)

        return (multiview_images, orbit_camposes)
    
class Load_CharacterGen_Reconstruction_Model:

    # Announces in the UI that this node fetches weights if they are absent.
    DISPLAY_NAME = "(Down)Load CharacterGen Reconstruction Model"

    checkpoints_dir = "CharacterGen"
    default_repo_id = "zjpshadow/CharacterGen"
    config_path = "CharacterGen_configs/Stage_3D_infer.yaml"
    
    @classmethod
    def ckpt_dir(cls):
        """Absolute checkpoints directory.

        Derived on demand, not assigned as an INPUT_TYPES side effect:
        INPUT_TYPES need not have run in the process that executes
        this node. See Load_Hunyuan3D_V1.svrm_config().
        """
        return os.path.join(CKPT_ROOT_PATH, cls.checkpoints_dir)

    @classmethod
    def config_dir(cls):
        """Absolute config directory.

        Derived on demand, not assigned as an INPUT_TYPES side effect:
        INPUT_TYPES need not have run in the process that executes
        this node. See Load_Hunyuan3D_V1.svrm_config().
        """
        return os.path.join(CONFIG_ROOT_PATH, cls.config_path)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
            },
        }
    
    RETURN_TYPES = (
        "CHARACTER_LRM_PIPE",
    )
    RETURN_NAMES = (
        "character_lrm_pipe",
    )
    FUNCTION = "load_model"
    CATEGORY = "Comfy3D/Import|Export"
    
    def load_model(self):
        # Download checkpoints
        download_repo(self.default_repo_id, self.ckpt_dir(), ignore_patterns=HF_DOWNLOAD_IGNORE)
        # Load pre-trained models
        character_lrm_pipe = Inference3D_API(checkpoint_root_path=self.ckpt_dir(), cfg=load_config_cg3d(self.config_dir()))
        
        cstr(f"[{self.__class__.__name__}] loaded model ckpt from {self.ckpt_dir()}").msg.print()
        return (character_lrm_pipe,)
    
class CharacterGen_Reconstruction_Model:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "character_lrm_pipe": ("CHARACTER_LRM_PIPE", ),
                "multiview_images": ("IMAGE",),
                "multiview_masks": ("MASK",),
            }
        }

    RETURN_TYPES = (
        "TRIMESH",
    )
    RETURN_NAMES = (
        "mesh",
    )
    
    FUNCTION = "run_LRM"
    CATEGORY = "Comfy3D/Algorithm"
    
    @torch.no_grad()
    def run_LRM(self, character_lrm_pipe, multiview_images, multiview_masks):
        pil_mv_image_list = torch_imgs_to_pils(multiview_images, multiview_masks, alpha_min=0.2)
        
        vertices, faces = character_lrm_pipe.inference(pil_mv_image_list)

        mesh = Mesh(v=vertices, f=faces.to(torch.int64), device=get_device())
        mesh.auto_normal()
        mesh.auto_uv()
        
        return (_wire_out(mesh),)
    
class Load_Craftsman_Shape_Diffusion_Model:

    # Announces in the UI that this node fetches weights if they are absent.
    DISPLAY_NAME = "(Down)Load Craftsman Shape Diffusion Model"

    checkpoints_dir = "Craftsman"
    default_repo_id = "wyysf/CraftsMan"
    default_ckpt_name = "image-to-shape-diffusion/clip-mvrgb-modln-l256-e64-ne8-nd16-nl6-aligned-vae/model.ckpt"
    config_path = "Craftsman_config.yaml"
    
    @classmethod
    def ckpt_dir(cls):
        """Absolute checkpoints directory.

        Derived on demand, not assigned as an INPUT_TYPES side effect:
        INPUT_TYPES need not have run in the process that executes
        this node. See Load_Hunyuan3D_V1.svrm_config().
        """
        return os.path.join(CKPT_ROOT_PATH, cls.checkpoints_dir)

    @classmethod
    def config_dir(cls):
        """Absolute config directory.

        Derived on demand, not assigned as an INPUT_TYPES side effect:
        INPUT_TYPES need not have run in the process that executes
        this node. See Load_Hunyuan3D_V1.svrm_config().
        """
        return os.path.join(CONFIG_ROOT_PATH, cls.config_path)

    @classmethod
    def INPUT_TYPES(cls):
        all_models_names = get_list_filenames(cls.ckpt_dir(), SUPPORTED_CHECKPOINTS_EXTENSIONS, recursive=True)
        if cls.default_ckpt_name not in all_models_names:
            all_models_names += [cls.default_ckpt_name]

        return {
            "required": {
                "model_name": (all_models_names, ),
            },
        }
    
    RETURN_TYPES = (
        "CRAFTSMAN_MODEL",
    )
    RETURN_NAMES = (
        "craftsman_model",
    )
    FUNCTION = "load_model"
    CATEGORY = "Comfy3D/Import|Export"
    
    def load_model(self, model_name):
        ckpt_path = resume_or_download_model_from_hf(self.ckpt_dir(), self.default_repo_id, model_name, self.__class__.__name__)
        
        cfg: ExperimentConfigCraftsman
        cfg = load_config_craftsman(self.config_dir())

        def build(_config):
            model: BaseSystem = craftsman.find(cfg.system_type)(
                cfg.system, 
            )
            model.load_state_dict(torch.load(ckpt_path, map_location=torch.device('cpu'))['state_dict'])
            return model.eval()

        craftsman_model = model_cache.managed("craftsman", {"ckpt": ckpt_path}, build)

        cstr(f"[{self.__class__.__name__}] loaded model ckpt from {self.ckpt_dir()}").msg.print()
        return (craftsman_model,)
    
class Craftsman_Shape_Diffusion_Model:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "craftsman_model": ("CRAFTSMAN_MODEL", ),
                "multiview_images": ("IMAGE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "guidance_scale": ("FLOAT", {"default": 5.0, "min": 0.0, "step": 0.01}),
                "num_inference_steps": ("INT", {"default": 50, "min": 1}),
                "marching_cude_grids_resolution": ("INT", {"default": 256, "min": 1, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = (
        "TRIMESH",
    )
    RETURN_NAMES = (
        "mesh",
    )
    
    FUNCTION = "run_model"
    CATEGORY = "Comfy3D/Algorithm"
    
    @torch.no_grad()
    def run_model(self, craftsman_model, multiview_images, seed, guidance_scale, num_inference_steps, marching_cude_grids_resolution):
        pil_mv_image_list = torch_imgs_to_pils(multiview_images)
        
        sample_inputs = {"mvimages": [pil_mv_image_list]}   # view order: front, right, back, left
        
        latents = craftsman_model.sample(
            sample_inputs,
            sample_times=1,
            steps=num_inference_steps,
            guidance_scale=guidance_scale,
            return_intermediates=False,
            seed=seed
        )[0]
        
        cstr(f"[{self.__class__.__name__}] Starting to extract mesh...").msg.print()
        # decode the latents to mesh
        box_v = 1.1
        mesh_outputs, _ = craftsman_model.shape_model.extract_geometry(
            latents,
            bounds=[-box_v, -box_v, -box_v, box_v, box_v, box_v],
            grids_resolution=marching_cude_grids_resolution
        )
        vertices, faces = torch.from_numpy(mesh_outputs[0][0]).to(get_device()), torch.from_numpy(mesh_outputs[0][1]).to(torch.int64).to(get_device())

        mesh = Mesh(v=vertices, f=faces, device=get_device())
        mesh.auto_normal()
        mesh.auto_uv()
        
        return (_wire_out(mesh),)

class OrbitPoses_JK:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "orbitpose_preset": (list(ORBITPOSE_PRESET_DICT.keys()),),
                "radius": ("STRING", {"default": "4.0, 4.0, 4.0, 4.0, 4.0, 4.0"}),
                "elevations": ("STRING", {"default": "0.0, 90.0, 0.0, 0.0, -90.0, 0.0"}),
                "azimuths": ("STRING", {"default": "-90.0, 0.0, 180.0, 90.0, 0.0, 0.0"}),
                "centerX": ("STRING", {"default": "0.0, 0.0, 0.0, 0.0, 0.0, 0.0"}),
                "centerY": ("STRING", {"default": "0.0, 0.0, 0.0, 0.0, 0.0, 0.0"}),
                "centerZ": ("STRING", {"default": "0.0, 0.0, 0.0, 0.0, 0.0, 0.0"}),
            },
        }
    
    RETURN_TYPES = ("ORBIT_CAMPOSES",)
    RETURN_NAMES = ("orbit_camposes",)
    
    FUNCTION = "get_orbit_poses"
    CATEGORY = "Comfy3D/Preprocessor"
    
    def get_orbit_poses(self, orbitpose_preset, azimuths, elevations, radius, centerX, centerY, centerZ):
        radius = radius.split(",")
        orbit_radius = [float(item) for item in radius]
        
        centerX = centerX.split(",")
        centerY = centerY.split(",")
        centerZ = centerZ.split(",")
        orbit_center_x = [float(item) for item in centerX]
        orbit_center_y = [float(item) for item in centerY]
        orbit_center_z = [float(item) for item in centerZ]
        
        if orbitpose_preset == "Custom":
            elevations = elevations.split(",")
            azimuths = azimuths.split(",")
            orbit_elevations = [float(item) for item in elevations]
            orbit_azimuths = [float(item) for item in azimuths]
        else:
            orbit_elevations, orbit_azimuths = ORBITPOSE_PRESET_DICT[orbitpose_preset]

        orbit_camposes = compose_orbit_camposes(orbit_radius, orbit_elevations, orbit_azimuths, orbit_center_x, orbit_center_y, orbit_center_z)

        return (orbit_camposes,)
    
class Load_CRM_T2I_V2_Models:

    # Announces in the UI that this node fetches weights if they are absent.
    DISPLAY_NAME = "(Down)Load CRM T2I V2 Models"

    crm_checkpoints_dir = "CRM"
    t2i_v2_checkpoints_dir = "T2I_V2"
    default_crm_ckpt_name = ["pixel-diffusion.pth"]
    default_crm_conf_name = ["sd_v2_base_ipmv_zero_SNR.yaml"]
    default_crm_repo_id = "Zhengyi/CRM"
    config_path = "CRM_T2I_V2_configs"
    
    @classmethod
    def crm_ckpt_dir(cls):
        """Absolute CRM checkpoints directory.

        Derived on demand, not assigned as an INPUT_TYPES side effect:
        INPUT_TYPES need not have run in the process that executes
        this node. See Load_Hunyuan3D_V1.svrm_config().
        """
        return os.path.join(CKPT_ROOT_PATH, cls.crm_checkpoints_dir)

    @classmethod
    def t2i_v2_ckpt_dir(cls):
        """Absolute T2I-V2 checkpoints directory.

        Derived on demand, not assigned as an INPUT_TYPES side effect:
        INPUT_TYPES need not have run in the process that executes
        this node. See Load_Hunyuan3D_V1.svrm_config().
        """
        return os.path.join(CKPT_ROOT_PATH, cls.t2i_v2_checkpoints_dir)

    @classmethod
    def config_dir(cls):
        """Absolute config directory.

        Derived on demand, not assigned as an INPUT_TYPES side effect:
        INPUT_TYPES need not have run in the process that executes
        this node. See Load_Hunyuan3D_V1.svrm_config().
        """
        return os.path.join(CONFIG_ROOT_PATH, cls.config_path)

    @classmethod
    def INPUT_TYPES(cls):
        all_crm_models_names = get_list_filenames(cls.crm_ckpt_dir(), SUPPORTED_CHECKPOINTS_EXTENSIONS)
        for ckpt_name in cls.default_crm_ckpt_name:
            if ckpt_name not in all_crm_models_names:
                all_crm_models_names += [ckpt_name]
                
            
        return {
            "required": {
                "crm_model_name": (all_crm_models_names, ),
                "crm_config_path": (cls.default_crm_conf_name, ),
            },
        }
    
    RETURN_TYPES = (
        "T2IADAPTER_V2",
        "CRM_MVDIFFUSION_SAMPLER_V2",
    )
    RETURN_NAMES = (
        "t2iadapter_v2",
        "crm_mvdiffusion_sampler_v2",
    )
    FUNCTION = "load_CRM"
    CATEGORY = "Comfy3D/Import|Export"
    
    def load_CRM(self, crm_model_name, crm_config_path):
        
        from .Gen_3D_Modules.CRM_T2I_V2.imagedream.ldm.util import instantiate_from_config, get_obj_from_str
        
        t2iadapter_v2 = T2IAdapterV2.from_pretrained(self.t2i_v2_ckpt_dir()).to(get_device(), dtype=WEIGHT_DTYPE)

        crm_config_path = os.path.join(self.config_dir(), crm_config_path)
        
        ckpt_path = resume_or_download_model_from_hf(self.crm_ckpt_dir(), self.default_crm_repo_id, crm_model_name, self.__class__.__name__)
            
        crm_config = OmegaConf.load(crm_config_path)

        crm_mvdiffusion_model = instantiate_from_config(crm_config.model)
        crm_mvdiffusion_model.load_state_dict(torch.load(ckpt_path, map_location="cpu"), strict=False)
        crm_mvdiffusion_model.device = get_device()
        
        crm_mvdiffusion_model.clip_model = crm_mvdiffusion_model.clip_model.to(get_device(), dtype=WEIGHT_DTYPE)
        crm_mvdiffusion_model.vae_model = crm_mvdiffusion_model.vae_model.to(get_device(), dtype=WEIGHT_DTYPE)
        crm_mvdiffusion_model = crm_mvdiffusion_model.to(get_device(), dtype=WEIGHT_DTYPE)
        
        crm_mvdiffusion_sampler_v2 = get_obj_from_str(crm_config.sampler.target)(
            crm_mvdiffusion_model, device=get_device(), dtype=WEIGHT_DTYPE, **crm_config.sampler.params
        )
        
        cstr(f"[{self.__class__.__name__}] loaded model ckpt from {ckpt_path}").msg.print()
        
        return (t2iadapter_v2, crm_mvdiffusion_sampler_v2, )
    
class CRM_T2I_V2_Models:
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "t2iadapter_v2": ("T2IADAPTER_V2",),
                "crm_mvdiffusion_sampler_v2": ("CRM_MVDIFFUSION_SAMPLER_V2",),
                "reference_image": ("IMAGE",),
                "reference_mask": ("MASK",),
                "normal_maps": ("IMAGE",),
                "prompt": ("STRING", {
                    "default": "3D assets",
                    "multiline": True
                }),
                "prompt_neg": ("STRING", {
                    "default": "uniform low no texture ugly, boring, bad anatomy, blurry, pixelated,  obscure, unnatural colors, poor lighting, dull, and unclear.", 
                    "multiline": True
                }),
                "seed": ("INT", {"default": 1234, "min": 0, "max": 0xffffffffffffffff}),
                "mv_guidance_scale": ("FLOAT", {"default": 5.5, "min": 0.0, "step": 0.01}),
                "num_inference_steps": ("INT", {"default": 50, "min": 1}),
                
            },
        }
    
    RETURN_TYPES = (
        "IMAGE",
        "ORBIT_CAMPOSES",   
    )
    RETURN_NAMES = (
        "multiview_images",
        "orbit_camposes",
    )
    FUNCTION = "run_model"
    CATEGORY = "Comfy3D/Algorithm"
    
    def run_model(
        self,
        t2iadapter_v2,
        crm_mvdiffusion_sampler_v2, 
        reference_image, # [N, 256, 256, 3]
        reference_mask,  # [N, 256, 256]
        normal_maps,     # [N * 6, 512, 512, 3]
        prompt, 
        prompt_neg, 
        seed,
        mv_guidance_scale, 
        num_inference_steps, 
    ):  
        # Convert tensores to pil images
        batch_reference_images = [CRMSamplerV2.process_pixel_img(img) for img in torch_imgs_to_pils(reference_image, reference_mask)]
        
        # Adapter conditioning.
        normal_maps = normal_maps.permute(0, 3, 1, 2).to(get_device(), dtype=WEIGHT_DTYPE)    # [N, H, W, 3] -> [N, 3, H, W]
        down_intrablock_additional_residuals = t2iadapter_v2(normal_maps)
        down_intrablock_additional_residuals = [
            sample.to(dtype=WEIGHT_DTYPE).chunk(reference_image.shape[0]) for sample in down_intrablock_additional_residuals
        ]   # List[ List[ feature maps tensor for one down sample block and for one ip image, ... ], ... ]

        # Inference
        multiview_images = CRMSamplerV2.stage1_sample(
            crm_mvdiffusion_sampler_v2,
            batch_reference_images,
            prompt,
            prompt_neg,
            seed,
            mv_guidance_scale, 
            num_inference_steps,
            additional_residuals=down_intrablock_additional_residuals
        ).to(dtype=reference_image.dtype, device=reference_image.device)
            
        gc.collect()
        torch.cuda.empty_cache()

        orbit_radius = [1.63634] * 6
        orbit_center = [0.0] * 6
        orbit_elevations, orbit_azimuths = ORBITPOSE_PRESET_DICT["CRM(6)"]
        orbit_camposes = compose_orbit_camposes(orbit_radius, orbit_elevations, orbit_azimuths, orbit_center, orbit_center, orbit_center)
        
        return (multiview_images, orbit_camposes)
    
class Load_CRM_T2I_V3_Models:

    # Announces in the UI that this node fetches weights if they are absent.
    DISPLAY_NAME = "(Down)Load CRM T2I V3 Models"

    crm_checkpoints_dir = "CRM"
    crm_t2i_v3_checkpoints_dir = "CRM_T2I_V3"
    t2i_v2_checkpoints_dir = "T2I_V2"
    default_crm_t2i_v3_ckpt_name = ["pixel-diffusion_lora_80k_rank_60_Hyper.pth", "pixel-diffusion_dora_90k_rank_128_Hyper.pth"]
    default_crm_ckpt_name = ["pixel-diffusion_Hyper.pth"]
    default_crm_conf_name = ["sd_v2_base_ipmv_zero_SNR_Hyper.yaml"]
    default_crm_repo_id = "Zhengyi/CRM"
    config_path = "CRM_T2I_V3_configs"
    
    @classmethod
    def crm_ckpt_dir(cls):
        """Absolute CRM checkpoints directory.

        Derived on demand, not assigned as an INPUT_TYPES side effect:
        INPUT_TYPES need not have run in the process that executes
        this node. See Load_Hunyuan3D_V1.svrm_config().
        """
        return os.path.join(CKPT_ROOT_PATH, cls.crm_checkpoints_dir)

    @classmethod
    def crm_t2i_v3_ckpt_dir(cls):
        """Absolute CRM T2I-V3 checkpoints directory.

        Derived on demand, not assigned as an INPUT_TYPES side effect:
        INPUT_TYPES need not have run in the process that executes
        this node. See Load_Hunyuan3D_V1.svrm_config().
        """
        return os.path.join(CKPT_ROOT_PATH, cls.crm_t2i_v3_checkpoints_dir)

    @classmethod
    def t2i_v2_ckpt_dir(cls):
        """Absolute T2I-V2 checkpoints directory.

        Derived on demand, not assigned as an INPUT_TYPES side effect:
        INPUT_TYPES need not have run in the process that executes
        this node. See Load_Hunyuan3D_V1.svrm_config().
        """
        return os.path.join(CKPT_ROOT_PATH, cls.t2i_v2_checkpoints_dir)

    @classmethod
    def config_dir(cls):
        """Absolute config directory.

        Derived on demand, not assigned as an INPUT_TYPES side effect:
        INPUT_TYPES need not have run in the process that executes
        this node. See Load_Hunyuan3D_V1.svrm_config().
        """
        return os.path.join(CONFIG_ROOT_PATH, cls.config_path)

    @classmethod
    def INPUT_TYPES(cls):
        all_crm_models_names = get_list_filenames(cls.crm_ckpt_dir(), SUPPORTED_CHECKPOINTS_EXTENSIONS)
        for ckpt_name in cls.default_crm_ckpt_name:
            if ckpt_name not in all_crm_models_names:
                all_crm_models_names += [ckpt_name]
                
        all_crm_t2i_v3_models_names = get_list_filenames(cls.crm_t2i_v3_ckpt_dir(), SUPPORTED_CHECKPOINTS_EXTENSIONS)
        for ckpt_name in cls.default_crm_t2i_v3_ckpt_name:
            if ckpt_name not in all_crm_t2i_v3_models_names:
                all_crm_t2i_v3_models_names += [ckpt_name] 
                
            
        return {
            "required": {
                "crm_model_name": (all_crm_models_names, ),
                "crm_t2i_v3_model_name": (all_crm_t2i_v3_models_names, ),
                "crm_config_path": (cls.default_crm_conf_name, ),
                "rank": ("INT", {"default": 64, "min": 1}),
                "use_dora": ("BOOLEAN", {"default": False}),
            },
        }
    
    # RealESRGAN_x4plus is what the shipped CRM_T2I_V3 texture workflows load
    # through the native UpscaleModelLoader. Same GitHub release MVAdapter
    # already pulls x2plus from (mvadapter_node_utils.py:424).
    UPSCALER_NAME = "RealESRGAN_x4plus.pth"
    UPSCALER_URL = ("https://github.com/xinntao/Real-ESRGAN/releases/download/"
                    "v0.1.0/RealESRGAN_x4plus.pth")

    # upscale_model_name is APPENDED so slots 0-1 keep their indices in saved
    # graphs.
    # "*" (ComfyUI's Any type), not "COMBO". A native loader's model_name input
    # is typed as its OPTION LIST, not the string "COMBO", and
    # comfy_execution.validation.validate_node_input compares the two directly:
    #     received_type(COMBO) mismatch input_type(['RealESRGAN_x4plus.pth'])
    # Only "*" short-circuits that check, so it is the one output type that
    # will actually connect to a combo widget.
    RETURN_TYPES = (
        "T2IADAPTER_V2",
        "CRM_MVDIFFUSION_SAMPLER_V3",
        "*",
    )
    RETURN_NAMES = (
        "t2iadapter_v2",
        "crm_mvdiffusion_sampler_v3",
        "upscale_model_name",
    )
    FUNCTION = "load_CRM"
    CATEGORY = "Comfy3D/Import|Export"
    
    def load_CRM(self, crm_model_name, crm_t2i_v3_model_name, crm_config_path, rank, use_dora):
        
        from .Gen_3D_Modules.CRM_T2I_V3.imagedream.ldm.util import instantiate_from_config, get_obj_from_str
        
        t2iadapter_v2 = T2IAdapterV2.from_pretrained(self.t2i_v2_ckpt_dir()).to(get_device(), dtype=WEIGHT_DTYPE)

        crm_config_path = os.path.join(self.config_dir(), crm_config_path)
        
        ckpt_path = resume_or_download_model_from_hf(self.crm_ckpt_dir(), self.default_crm_repo_id, crm_model_name, self.__class__.__name__)
            
        crm_config = OmegaConf.load(crm_config_path)

        crm_mvdiffusion_model = instantiate_from_config(crm_config.model)
        crm_mvdiffusion_model.load_state_dict(torch.load(ckpt_path, map_location="cpu"), strict=False)
        crm_mvdiffusion_model.device = get_device()
        
        crm_mvdiffusion_model.clip_model = crm_mvdiffusion_model.clip_model.to(get_device(), dtype=WEIGHT_DTYPE)
        crm_mvdiffusion_model.vae_model = crm_mvdiffusion_model.vae_model.to(get_device(), dtype=WEIGHT_DTYPE)
        crm_mvdiffusion_model = crm_mvdiffusion_model.to(get_device(), dtype=WEIGHT_DTYPE)
        
        crm_mvdiffusion_sampler_v3 = get_obj_from_str(crm_config.sampler.target)(
            crm_mvdiffusion_model, device=get_device(), dtype=WEIGHT_DTYPE, **crm_config.sampler.params
        )
        
        unet = crm_mvdiffusion_model.model
        mvdiffusion_model = unet.diffusion_model
        self.inject_lora(mvdiffusion_model, rank, use_dora)
        
        pretrained_lora_model_path = os.path.join(self.crm_t2i_v3_ckpt_dir(), crm_t2i_v3_model_name)
        unet.load_state_dict(torch.load(pretrained_lora_model_path, map_location="cpu"), strict=False)
        
        upscale_model_name = _ensure_upscale_model(self.UPSCALER_URL, self.UPSCALER_NAME)

        cstr(f"[{self.__class__.__name__}] loaded model ckpt from {ckpt_path} and {pretrained_lora_model_path}").msg.print()

        return (t2iadapter_v2, crm_mvdiffusion_sampler_v3, upscale_model_name, )
    
    def inject_lora(self, mvdiffusion_model, rank=64, use_dora=False):
        from peft import LoraConfig, inject_adapter_in_model
        # Add new LoRA weights to the original attention layers
        unet_lora_config = LoraConfig(
            r=rank,
            use_dora=use_dora,
            lora_alpha=rank,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_k_ip", "to_q", "to_v", "to_v_ip", "to_out.0"],
        )
        
        inject_adapter_in_model(unet_lora_config, mvdiffusion_model.input_blocks, "DoRA" if use_dora else "LoRA")
        inject_adapter_in_model(unet_lora_config, mvdiffusion_model.middle_block, "DoRA" if use_dora else "LoRA")
        inject_adapter_in_model(unet_lora_config, mvdiffusion_model.output_blocks, "DoRA" if use_dora else "LoRA")

class CRM_T2I_V3_Models:
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "t2iadapter_v2": ("T2IADAPTER_V2",),
                "crm_mvdiffusion_sampler_v3": ("CRM_MVDIFFUSION_SAMPLER_V3",),
                "reference_image": ("IMAGE",),
                "reference_mask": ("MASK",),
                "normal_maps": ("IMAGE",),
                "prompt": ("STRING", {
                    "default": "3D assets",
                    "multiline": True
                }),
                "prompt_neg": ("STRING", {
                    "default": "uniform low no texture ugly, boring, bad anatomy, blurry, pixelated,  obscure, unnatural colors, poor lighting, dull, and unclear.", 
                    "multiline": True
                }),
                "seed": ("INT", {"default": 1234, "min": 0, "max": 0xffffffffffffffff}),
                "mv_guidance_scale": ("FLOAT", {"default": 5.5, "min": 0.0, "step": 0.01}),
                "num_inference_steps": ("INT", {"default": 50, "min": 1}),
                
            },
        }
    
    RETURN_TYPES = (
        "IMAGE",
        "IMAGE",
        "IMAGE",
        "ORBIT_CAMPOSES",   
    )
    RETURN_NAMES = (
        "multiview_albedos",
        "multiview_metalness",
        "multiview_roughness",
        "orbit_camposes",
    )
    FUNCTION = "run_model"
    CATEGORY = "Comfy3D/Algorithm"
    
    def run_model(
        self,
        t2iadapter_v2,
        crm_mvdiffusion_sampler_v3, 
        reference_image, # [N, 256, 256, 3]
        reference_mask,  # [N, 256, 256]
        normal_maps,     # [N * 6, 512, 512, 3]
        prompt, 
        prompt_neg, 
        seed,
        mv_guidance_scale, 
        num_inference_steps, 
    ):  
        # Convert tensores to pil images
        batch_reference_images = [CRMSamplerV3.process_pixel_img(img) for img in torch_imgs_to_pils(reference_image, reference_mask)]
        
        # Adapter conditioning.
        normal_maps = normal_maps.permute(0, 3, 1, 2).to(get_device(), dtype=WEIGHT_DTYPE)    # [N, H, W, 3] -> [N, 3, H, W]
        down_intrablock_additional_residuals = t2iadapter_v2(normal_maps)
        down_intrablock_additional_residuals = [
            sample.to(dtype=WEIGHT_DTYPE).chunk(reference_image.shape[0]) for sample in down_intrablock_additional_residuals
        ]   # List[ List[ feature maps tensor for one down sample block and for one ip image, ... ], ... ]

        all_multiview_images = [[], [], []] # [list of albedo mvs, list of metalness mvs, list of roughness mvs]

        # Inference
        multiview_images = CRMSamplerV3.stage1_sample(
            crm_mvdiffusion_sampler_v3,
            batch_reference_images,
            prompt,
            prompt_neg,
            seed,
            mv_guidance_scale, 
            num_inference_steps,
            additional_residuals=down_intrablock_additional_residuals
        )
        
        num_mvs = crm_mvdiffusion_sampler_v3.num_frames - 1 # 6
        num_branches = crm_mvdiffusion_sampler_v3.model.model.diffusion_model.num_branches # 3
        ip_batch_size = reference_image.shape[0]
        i_mvs = 0
        for i_branch in range(num_branches):
            for _ in range(ip_batch_size):
                batch_of_mv_imgs = torch.stack(multiview_images[i_mvs:i_mvs+num_mvs], axis=0)
                i_mvs += num_mvs
            
                all_multiview_images[i_branch].append(batch_of_mv_imgs)
              
        output_images = [None] * num_branches
        for i_branch in range(num_branches):
            output_images[i_branch] = torch.cat(all_multiview_images[i_branch], dim=0).to(reference_image.device, dtype=reference_image.dtype)
            
        gc.collect()
        torch.cuda.empty_cache()

        orbit_radius = [1.63634] * 6
        orbit_center = [0.0] * 6
        orbit_elevations, orbit_azimuths = ORBITPOSE_PRESET_DICT["CRM(6)"]
        orbit_camposes = compose_orbit_camposes(orbit_radius, orbit_elevations, orbit_azimuths, orbit_center, orbit_center, orbit_center)
        
        return (output_images[0], output_images[1], output_images[2], orbit_camposes)

class Hunyuan3D_V1_MVDiffusion_Model:
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mvdiffusion_pipe": ("DIFFUSERS_PIPE",),
                "reference_image": ("IMAGE",),
                "reference_mask": ("MASK",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "mv_guidance_scale": ("FLOAT", {"default": 2.0, "min": 0.0, "step": 0.01}),
                "num_inference_steps": ("INT", {"default": 50, "min": 1}),
            }
        }
    
    RETURN_TYPES = (
        "IMAGE",
        "IMAGE",
    )
    RETURN_NAMES = (
        "multiview_image_grid",
        "condition_image",
    )
    FUNCTION = "run_mvdiffusion"
    CATEGORY = "Comfy3D/Algorithm"
    
    @torch.no_grad()
    def run_mvdiffusion(
        self, 
        mvdiffusion_pipe, 
        reference_image, # [1, H, W, 3]
        reference_mask,  # [1, H, W]
        seed,
        mv_guidance_scale, 
        num_inference_steps, 
    ):
        # A DIFFUSERS_PIPE may arrive as a recipe dict or as a live
        # pipeline; resolve_diffusers_pipe passes the latter through.
        mvdiffusion_pipe = resolve_diffusers_pipe(mvdiffusion_pipe)
        single_image = torch_imgs_to_pils(reference_image, reference_mask)[0]

        generator = torch.Generator(device=mvdiffusion_pipe.device).manual_seed(seed)
        views_grid_pil, cond_pil = mvdiffusion_pipe(single_image, 
            num_inference_steps=num_inference_steps,
            guidance_scale=mv_guidance_scale, 
            generat=generator
        ).images
        
        multiview_image_grid = pils_to_torch_imgs(views_grid_pil, reference_image.dtype, reference_image.device)
        condition_image = pils_to_torch_imgs(cond_pil, reference_image.dtype, reference_image.device)

        return (multiview_image_grid, condition_image)
    
class Load_Hunyuan3D_V1:
    """Both halves of Hunyuan3D V1 from one variant choice.

    V1 runs in two stages -- multi-view diffusion turns one image into six
    views, then reconstruction turns those into a mesh -- and they were two
    nodes with two independent widgets that HAD to agree:

      * the diffusion loader picked a subfolder, mvd_lite or mvd_std
      * the reconstruction loader had a `use_lite` boolean whose only effect
        (views_to_mesh.py:46) is choosing the order the six views are read in,
        [0,1,2,3,4,5] for lite against [0,2,4,5,3,1] for std

    Nothing enforced the pairing, and a mismatch silently assigns views to the
    wrong camera angles: a wrong mesh, no error. The shipped Hunyuan3D_V1.json
    had exactly that -- an mvd_std branch with use_lite=True.

    One widget now drives both, so the mismatch cannot be expressed.
    """

    DISPLAY_NAME = "(Down)Load Hunyuan3D V1"

    _REPO_ID = "tencent/Hunyuan3D-1"
    _SVRM = "svrm/svrm.safetensors"
    config_path = "Hunyuan3D_V1_svrm_config.yaml"

    # variant -> (subfolder, pipeline class name, use_lite view order)
    _VARIANTS = {
        "lite": ("mvd_lite", "Hunyuan3DMVDLitePipeline", True),
        "std":  ("mvd_std",  "Hunyuan3DMVDStdPipeline",  False),
    }

    @classmethod
    def svrm_config(cls):
        """Absolute path to the reconstruction config.

        A method, not an attribute set inside INPUT_TYPES. Thirty nodes in this
        file set `cls.<x>_abs` as an INPUT_TYPES side effect, which holds only
        while the same node both defines and uses it -- _build_hunyuan3d_v1_recon
        runs from the *consumer* via resolve_diffusers_pipe, in a process where
        this node's INPUT_TYPES may never have been called.
        """
        return os.path.join(CONFIG_ROOT_PATH, cls.config_path)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "variant": (list(cls._VARIANTS.keys()), {"default": "lite"}),
            },
        }

    RETURN_TYPES = ("DIFFUSERS_PIPE", "HUNYUAN3D_V1_RECONSTRUCTION_MODEL")
    RETURN_NAMES = ("mvdiffusion_pipe", "reconstruction_model")
    FUNCTION = "load"
    CATEGORY = "Comfy3D/Import|Export"

    @classmethod
    def _ensure_weights(cls, variant):
        """Fetch one variant plus the shared reconstruction weights.

        Scoped: the repo is mvd_lite 5.58 GB + mvd_std 9.60 GB + svrm 0.92 GB,
        and the old loader took all of it -- ~16 GB to use 6.5 GB, with the
        unused variant unusable anyway.
        """
        subfolder, _cls_name, _lite = cls._VARIANTS[variant]
        return download_repo(
            cls._REPO_ID,
            CKPT_DIFFUSERS_PATH, cls._REPO_ID,
            ignore_patterns=HF_DOWNLOAD_IGNORE,
            allow_patterns=[f"{subfolder}/**", "svrm/**"],
            requires=[f"{subfolder}/model_index.json", cls._SVRM],
        )

    def load(self, variant):
        self._ensure_weights(variant)
        subfolder, pipeline_name, _lite = self._VARIANTS[variant]
        return (
            model_cache.recipe(
                "diffusers_pipe",
                pipeline_name=pipeline_name,
                repo_id=self._REPO_ID,
                checkpoint_sub_dir=subfolder,
            ),
            model_cache.recipe("hunyuan3d_v1_recon", variant=variant),
        )


@_pipe_builder("hunyuan3d_v1_recon")
def _build_hunyuan3d_v1_recon(config):
    """Materialize a hunyuan3d_v1_recon recipe. Only on a cache miss."""
    cls = Load_Hunyuan3D_V1
    variant = config["variant"]
    _subfolder, _name, use_lite = cls._VARIANTS[variant]
    base = cls._ensure_weights(variant)
    ckpt = os.path.join(base, cls._SVRM)
    model = Views2Mesh(cls.svrm_config(), ckpt, get_device(), use_lite=use_lite)
    cstr(f"[Load_Hunyuan3D_V1] loaded {variant} reconstruction ckpt from {ckpt}").msg.print()
    return model


class Hunyuan3D_V1_Reconstruction_Model:
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "hunyuan3d_v1_reconstruction_model": ("HUNYUAN3D_V1_RECONSTRUCTION_MODEL",),
                "multiview_image_grid": ("IMAGE",),
                "condition_image": ("IMAGE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "target_face_count": ("INT", {"default": 90000, "min": 1}),
            }
        }

    RETURN_TYPES = (
        "TRIMESH",
    )
    RETURN_NAMES = (
        "mesh",
    )
    
    FUNCTION = "run_model"
    CATEGORY = "Comfy3D/Algorithm"
    
    @torch.no_grad()
    def run_model(self, hunyuan3d_v1_reconstruction_model, multiview_image_grid, condition_image, seed, target_face_count):
        # May arrive as a recipe (Load_Hunyuan3D_V1) or, from an older graph, as
        # a live Views2Mesh; the resolver passes the latter through untouched.
        hunyuan3d_v1_reconstruction_model = resolve_diffusers_pipe(
            hunyuan3d_v1_reconstruction_model)
        mv_grid_pil = torch_imgs_to_pils(multiview_image_grid)[0]
        condition_pil = torch_imgs_to_pils(condition_image)[0]
        
        vertices, faces, vtx_colors = hunyuan3d_v1_reconstruction_model(
            mv_grid_pil,
            condition_pil,
            seed=seed,
            target_face_count=target_face_count
        )
        vertices, faces, vtx_colors = torch.from_numpy(vertices).to(get_device()), torch.from_numpy(faces).to(torch.int64).to(get_device()), torch.from_numpy(vtx_colors).to(get_device())
        mesh = Mesh(v=vertices, f=faces.to(torch.int64), vc=vtx_colors, device=get_device())
        mesh.auto_normal()
        
        return (_wire_out(mesh),)
    
# deprecated
class Hunyuan3D_V2_DiT_Flow_Matching_Model:
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "hunyuan3d_v2_i23d_pipe": ("DIFFUSERS_PIPE",),
                "reference_image": ("IMAGE",),
                "reference_mask": ("MASK",),
                "seed": ("INT", {"default": 1234, "min": 0, "max": 0xffffffffffffffff}),
                "guidance_scale": ("FLOAT", {"default": 5.5, "min": 0.0, "step": 0.01}),
                "num_inference_steps": ("INT", {"default": 30, "min": 1}),
                "octree_resolution": ("INT", {"default": 256, "min": 1}),
            }
        }
    
    RETURN_TYPES = (
        "TRIMESH",
    )
    RETURN_NAMES = (
        "mesh",
    )
    FUNCTION = "run_model"
    CATEGORY = "Comfy3D/Algorithm"
    
    @torch.no_grad()
    def run_model(
        self, 
        hunyuan3d_v2_i23d_pipe, 
        reference_image, # [1, H, W, 3]
        reference_mask,  # [1, H, W]
        seed,
        guidance_scale, 
        num_inference_steps,
        octree_resolution,
    ):
        # A DIFFUSERS_PIPE may arrive as a recipe dict or as a live
        # pipeline; resolve_diffusers_pipe passes the latter through.
        hunyuan3d_v2_i23d_pipe = resolve_diffusers_pipe(hunyuan3d_v2_i23d_pipe)
        single_image = torch_imgs_to_pils(reference_image, reference_mask)[0]

        generator = torch.Generator(device=hunyuan3d_v2_i23d_pipe.device).manual_seed(seed)
        mesh = hunyuan3d_v2_i23d_pipe(
            image=single_image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            octree_resolution=octree_resolution
        )[0]

        mesh = FloaterRemover()(mesh)
        mesh = DegenerateFaceRemover()(mesh)
        mesh = FaceReducer()(mesh)

        mesh = Mesh.load_trimesh(given_mesh=mesh)

        return (_wire_out(mesh),)

# deprecated
class Hunyuan3D_V2_Paint_Model:
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "hunyuan3d_v2_texgen_pipe": ("DIFFUSERS_PIPE",),
                "reference_image": ("IMAGE",),
                "reference_mask": ("MASK",),
                "mesh": ("TRIMESH",),
            }
        }
    
    RETURN_TYPES = (
        "TRIMESH",
    )
    RETURN_NAMES = (
        "mesh",
    )
    FUNCTION = "run_model"
    CATEGORY = "Comfy3D/Algorithm"
    
    @torch.no_grad()
    def run_model(
        self, 
        hunyuan3d_v2_texgen_pipe, 
        reference_image, # [1, H, W, 3]
        reference_mask,  # [1, H, W]
        mesh,
    ):
        mesh = _wire_in(mesh)
        single_image = torch_imgs_to_pils(reference_image, reference_mask)[0]

        v_np = mesh.v.detach().cpu().numpy()
        f_np = mesh.f.detach().cpu().numpy()

        mesh = trimesh.Trimesh(vertices=v_np, faces=f_np, process=False)
        hunyuan3d_v2_texgen_pipe = resolve_hunyuan3d_v2_texgen(hunyuan3d_v2_texgen_pipe)
        mesh = hunyuan3d_v2_texgen_pipe(mesh, single_image)

        mesh = Mesh.load_trimesh(given_mesh=mesh)
        mesh.auto_normal()

        return (_wire_out(mesh),)
    
class Load_Trellis_Structured_3D_Latents_Models:
    default_repo_id = "jetx/TRELLIS-image-large"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "repo_id": ("STRING", {"default": cls.default_repo_id, "multiline": False}),
            },
        }
    
    RETURN_TYPES = (
        "TRELLIS_PIPE",
    )
    RETURN_NAMES = (
        "trellis_pipe",
    )
    FUNCTION = "load_pipe"
    CATEGORY = "Comfy3D/Import|Export"
    
    def load_pipe(self, repo_id):
        
        pipe = TrellisImageTo3DPipeline.from_pretrained(repo_id)
        pipe.to(get_device())
        
        return (pipe,)
    
    
class Trellis_Structured_3D_Latents_Models:
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trellis_pipe": ("TRELLIS_PIPE",),
                "reference_image": ("IMAGE",),
                "reference_mask": ("MASK",),
                "seed": ("INT", {"default": 1, "min": 0, "max": 0xffffffffffffffff}),
                "sparse_structure_guidance_scale": ("FLOAT", {"default": 7.5, "min": 0.0, "step": 0.01}),
                "sparse_structure_sample_steps": ("INT", {"default": 12, "min": 1}),
                "structured_latent_guidance_scale": ("FLOAT", {"default": 3.0, "min": 0.0, "step": 0.01}),
                "structured_latent_sample_steps": ("INT", {"default": 12, "min": 1}),
            }
        }
    
    RETURN_TYPES = (
        "TRIMESH",
    )
    RETURN_NAMES = (
        "mesh",
    )
    FUNCTION = "run_model"
    CATEGORY = "Comfy3D/Algorithm"
    
    @torch.no_grad()
    def run_model(
        self, 
        trellis_pipe, 
        reference_image, # [1, H, W, 3]
        reference_mask,  # [1, H, W]
        seed,
        sparse_structure_guidance_scale,
        sparse_structure_sample_steps,
        structured_latent_guidance_scale,
        structured_latent_sample_steps,
    ):
        single_image = torch_imgs_to_pils(reference_image, reference_mask)[0]

        outputs = trellis_pipe.run(
            single_image,
            # Optional parameters
            seed=seed,
            formats=["gaussian", "mesh"],
            sparse_structure_sampler_params={
                "cfg_strength": sparse_structure_guidance_scale,
                "steps": sparse_structure_sample_steps,
            },
            slat_sampler_params={
                "cfg_strength": structured_latent_guidance_scale,
                "steps": structured_latent_sample_steps,
            },
        )

        # GLB files can be extracted from the outputs
        vertices, faces, uvs, texture = postprocessing_utils.finalize_mesh(
            outputs['gaussian'][0],
            outputs['mesh'][0],
            # Optional parameters
            simplify=0.95,          # Ratio of triangles to remove in the simplification process
            texture_size=1024,      # Size of the texture used for the GLB
        )

        vertices, faces, uvs, texture = torch.from_numpy(vertices).to(get_device()), torch.from_numpy(faces).to(torch.int64).to(get_device()), torch.from_numpy(uvs).to(get_device()), torch.from_numpy(texture).to(get_device())
        mesh = Mesh(v=vertices, f=faces, vt=uvs, ft=faces, albedo=texture, device=get_device())
        mesh.auto_normal()

        return (_wire_out(mesh),)

class TripoSG_I23D_Model:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tsg_pipe": ("DIFFUSERS_PIPE",),
                "reference_image": ("IMAGE",),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
                "guidance_scale": ("FLOAT", {"default": 7.0, "min": 0.0, "step": 0.01}),
                "num_inference_steps": ("INT", {"default": 50, "min": 1}),
                "use_flash_decoder": ("BOOLEAN", {"default": True}),
                "flash_octree_depth": ("INT", {"default": 9, "min": 1}),
                "hierarchical_octree_depth": ("INT", {"default": 9, "min": 1}),
                "dense_octree_depth": ("INT", {"default": 8, "min": 1}),
            }
        }
    
    RETURN_TYPES = (
        "TRIMESH",
    )
    RETURN_NAMES = (
        "mesh",
    )
    FUNCTION = "run_model"
    CATEGORY = "Comfy3D/Algorithm"
    
    @torch.no_grad()
    def run_model(
        self, 
        tsg_pipe, 
        reference_image, # [1, H, W, 3]
        seed,
        guidance_scale,
        num_inference_steps,
        use_flash_decoder,
        flash_octree_depth,
        hierarchical_octree_depth,
        dense_octree_depth,
    ):
        
        # A DIFFUSERS_PIPE may arrive as a recipe dict or as a live
        # pipeline; resolve_diffusers_pipe passes the latter through.
        tsg_pipe = resolve_diffusers_pipe(tsg_pipe)
        single_image = torch_imgs_to_pils(reference_image)[0]
        
        with torch.inference_mode(False):
            outputs = tsg_pipe(
                image=single_image,
                generator=torch.Generator(device=get_device()).manual_seed(seed),
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                use_flash_decoder=use_flash_decoder,
                flash_octree_depth=flash_octree_depth,
                hierarchical_octree_depth=hierarchical_octree_depth,
                dense_octree_depth=dense_octree_depth,
            ).samples[0]

            vertices, faces = torch.from_numpy(outputs[0].astype(np.float32)).to(get_device()), torch.from_numpy(np.ascontiguousarray(outputs[1])).to(torch.int64).to(get_device()) 
            mesh = Mesh(v=vertices, f=faces, device=get_device())
            mesh.auto_normal()

        return (_wire_out(mesh),)
    
class TripoSG_Scribble_Model:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tsg_scribble_pipe": ("DIFFUSERS_PIPE",),
                "scribble_image": ("IMAGE",),
                "prompt": ("STRING", {
                    "default": "3D assets",
                    "multiline": True
                }),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
                "num_inference_steps": ("INT", {"default": 16, "min": 1}),
                "scribble_confidence": ("FLOAT", {"default": 0.4, "min": 0.0, "step": 0.01}),
                "prompt_confidence": ("FLOAT", {"default": 1.0, "min": 0.0, "step": 0.01}),
                "use_flash_decoder": ("BOOLEAN", {"default": False}),
                "flash_octree_depth": ("INT", {"default": 8, "min": 1}),
                "hierarchical_octree_depth": ("INT", {"default": 8, "min": 1}),
                "dense_octree_depth": ("INT", {"default": 8, "min": 1}),
            }
        }
    
    RETURN_TYPES = (
        "TRIMESH",
    )
    RETURN_NAMES = (
        "mesh",
    )
    FUNCTION = "run_model"
    CATEGORY = "Comfy3D/Algorithm"
    
    @torch.no_grad()
    def run_model(
        self, 
        tsg_scribble_pipe, 
        scribble_image, # [1, H, W, 3]
        prompt,
        seed,
        num_inference_steps,
        scribble_confidence,
        prompt_confidence,
        use_flash_decoder,
        flash_octree_depth,
        hierarchical_octree_depth,
        dense_octree_depth,
    ):
        
        # A DIFFUSERS_PIPE may arrive as a recipe dict or as a live
        # pipeline; resolve_diffusers_pipe passes the latter through.
        tsg_scribble_pipe = resolve_diffusers_pipe(tsg_scribble_pipe)
        single_image = torch_imgs_to_pils(scribble_image)[0]
        
        outputs = tsg_scribble_pipe(
            image=single_image,
            prompt=prompt,
            generator=torch.Generator(device=get_device()).manual_seed(seed),
            num_inference_steps=num_inference_steps,
            guidance_scale=0, # this is a CFG-distilled model
            attention_kwargs={"cross_attention_scale": prompt_confidence, "cross_attention_2_scale": scribble_confidence},
            use_flash_decoder=use_flash_decoder,
            flash_octree_depth=flash_octree_depth, # there're some boundary problems when using flash decoder with this model
            hierarchical_octree_depth=hierarchical_octree_depth,
            dense_octree_depth=dense_octree_depth,
        ).samples[0]

        vertices, faces = torch.from_numpy(outputs[0].astype(np.float32)).to(get_device()), torch.from_numpy(np.ascontiguousarray(outputs[1])).to(torch.int64).to(get_device())
        mesh = Mesh(v=vertices, f=faces.to(torch.int64), device=get_device())
        mesh.auto_normal()

        return (_wire_out(mesh),)

class Load_Hunyuan3D_V2_ShapeGen_Pipeline:

    # Announces in the UI that this node fetches weights if they are absent.
    DISPLAY_NAME = "(Down)Load Hunyuan3D V2 ShapeGen Pipeline"

    CATEGORY      = "Comfy3D/Algorithm"
    RETURN_TYPES  = ("DIFFUSERS_PIPE",)
    RETURN_NAMES  = ("shapegen_pipe",)
    FUNCTION      = "load"

    _REPO_ID_BASE = "tencent"

    _MODES = {
        "Hunyuan3D-2":             ("Hunyuan3D-2",     "hunyuan3d-dit-v2-0",         30),
        "Hunyuan3D-2-Fast":        ("Hunyuan3D-2",     "hunyuan3d-dit-v2-0-fast",    20),
        "Hunyuan3D-2-Turbo":       ("Hunyuan3D-2",     "hunyuan3d-dit-v2-0-turbo",    5),
        "Hunyuan3D-2mini":         ("Hunyuan3D-2mini", "hunyuan3d-dit-v2-mini",       30),
        "Hunyuan3D-2mini-Fast":    ("Hunyuan3D-2mini", "hunyuan3d-dit-v2-mini-fast",   20),
        "Hunyuan3D-2mini-Turbo":   ("Hunyuan3D-2mini", "hunyuan3d-dit-v2-mini-turbo",  5),
        "Hunyuan3D-2mv":           ("Hunyuan3D-2mv",   "hunyuan3d-dit-v2-mv",   30),
        "Hunyuan3D-2mv-Fast":      ("Hunyuan3D-2mv",   "hunyuan3d-dit-v2-mv-fast",    20),
        "Hunyuan3D-2mv-Turbo":     ("Hunyuan3D-2mv",   "hunyuan3d-dit-v2-mv-turbo",   5),
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "generation_mode": (list(cls._MODES.keys()),),
                # No weights_format widget: safetensors, always. Every shipped
                # workflow already chose it, the .ckpt is the same tensors in a
                # pickle, and offering the choice only invites loading one.
                "flash_vdm"      : ("BOOLEAN", {"default": True}),
            }
        }

    @staticmethod
    def _ensure_weights(repo: str, subfolder: str):
        cls = Load_Hunyuan3D_V2_ShapeGen_Pipeline
        ckpt_file = "model.fp16.safetensors"
        # One named file, not a repo snapshot. _build_pipe below calls
        # from_single_file, and config.yaml is pinned in git in this same
        # directory -- so nothing else from the repo is ever read.
        #
        # This used to be download_repo() with only `requires`, and `requires`
        # decides *whether* to fetch, not *what*: it took all of
        # tencent/Hunyuan3D-2, which is 74.89 GB across nine model variants,
        # to use 4.93 GB of it. download_files skips the repo listing walk too.
        download_files(
            f"{cls._REPO_ID_BASE}/{repo}",
            [f"{subfolder}/{ckpt_file}"],
            CKPT_DIFFUSERS_PATH, f"{cls._REPO_ID_BASE}/{repo}",
        )

    @staticmethod
    def _build_pipe(repo: str, subfolder: str, flash_vdm: bool):
        Load_Hunyuan3D_V2_ShapeGen_Pipeline._ensure_weights(repo, subfolder)

        model_dir = os.path.join(CKPT_DIFFUSERS_PATH,
                                 f"{Load_Hunyuan3D_V2_ShapeGen_Pipeline._REPO_ID_BASE}/{repo}",
                                 subfolder)
        ckpt = os.path.join(model_dir, "model.fp16.safetensors")
        cfg  = os.path.join(model_dir, "config.yaml")

        pipe = Hunyuan3DDiTFlowMatchingPipeline.from_single_file(
            ckpt_path=ckpt,
            config_path=cfg,
            device=get_device(),
            dtype=WEIGHT_DTYPE,
            use_safetensors=True,
            from_pretrained_kwargs={
                "model_path": f"{Load_Hunyuan3D_V2_ShapeGen_Pipeline._REPO_ID_BASE}/{repo}",
                "subfolder": subfolder,
                "use_safetensors": True,
            },
        )

        if flash_vdm and any(tag in subfolder for tag in ("turbo", "fast")):
            pipe.enable_flashvdm(replace_vae=False)

        # Cast dtype, do NOT place on device: comfy/sd.py does the same
        # (`first_stage_model.to(self.vae_dtype)`, then the module goes to a
        # ModelPatcher). model_cache.managed() wraps this so load_models_gpu()
        # chooses placement -- that is what makes eviction and partial loading
        # possible. Moving here spends the VRAM before ComfyUI is asked if
        # there is room, which is where every OOM in the 20:05 log was raised.
        #
        # Keyword, not positional: this pipeline defines `to(self, device=None,
        # dtype=None)`, so a positional dtype is taken as the device. It looks
        # right because nn.Module.to() does accept a dtype positionally.
        return pipe.to(dtype=WEIGHT_DTYPE)

    def load(self, generation_mode, flash_vdm):
        repo, subfolder, _def_steps = self._MODES[generation_mode]
        # Download now, build later: see _build_hunyuan3d_v2_shapegen.
        self._ensure_weights(repo, subfolder)
        return (model_cache.recipe(
            "hunyuan3d_v2_shapegen",
            generation_mode=generation_mode,
            flash_vdm=flash_vdm,
        ),)


@_pipe_builder("instantmesh_lrm")
def _build_instantmesh_lrm(config):
    """Materialize an instantmesh_lrm recipe. Only on a cache miss."""
    from .Gen_3D_Modules.InstantMesh.utils.train_util import instantiate_from_config

    model = instantiate_from_config(OmegaConf.load(config["config"]).model_config)
    model.load_state_dict(_instantmesh_state_dict(config["ckpt"]), strict=True)
    return model.eval()


def resolve_lrm_model(config):
    """LRM_MODEL recipe -> live model, cached across runs.

    Non-dicts pass through, as with resolve_diffusers_pipe: this is called
    unconditionally by consumers, and a live model must not be frozen into a
    cache key.
    """
    if not isinstance(config, dict):
        return config

    def post_load(model, device):
        # FlexiCubes allocates its tetrahedral grid on the target device, so it
        # has to wait until the weights are actually there.
        if config.get("flexicubes"):
            model.init_flexicubes_geometry(device, fovy=30.0)

    return model_cache.managed(
        "instantmesh_lrm", config, _build_instantmesh_lrm,
        post_load=post_load, reloadable=True)


@_pipe_builder("hunyuan3d_v2_shapegen")
def _build_hunyuan3d_v2_shapegen(config):
    """Materialize a hunyuan3d_v2_shapegen recipe. Only on a cache miss."""
    cls = Load_Hunyuan3D_V2_ShapeGen_Pipeline
    repo, subfolder, def_steps = cls._MODES[config["generation_mode"]]
    pipe = cls._build_pipe(repo, subfolder, config["flash_vdm"])
    pipe.num_inference_steps = def_steps
    return pipe
    
class Load_Hunyuan3D_V2_TexGen_Pipeline: 

    # Announces in the UI that this node fetches weights if they are absent.
    DISPLAY_NAME = "(Down)Load Hunyuan3D V2 TexGen Pipeline"

    CATEGORY     = "Comfy3D/Algorithm"
    RETURN_TYPES = ("DIFFUSERS_PIPE",)
    RETURN_NAMES = ("texgen_pipe",)
    FUNCTION     = "load"

    MODEL2REPO = {
        "Standard": ("tencent/Hunyuan3D-2", "hunyuan3d-paint-v2-0"),
        "Turbo":    ("tencent/Hunyuan3D-2", "hunyuan3d-paint-v2-0-turbo"),
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "generation_mode": (list(cls.MODEL2REPO.keys()),),
        }}

    def _download_required_weights(self, repo_id, subfolder):
        # The loop this replaces iterated over the two folder names but never
        # used the loop variable and passed no allow_patterns, so it fetched the
        # whole repo -- twice -- despite the comment saying otherwise. One call,
        # scoped to the two directories that are actually wanted.
        # No `requires` here on purpose: the exact weight filenames vary by
        # submodel, and a directory-existence check would skip a partially
        # downloaded folder. snapshot_download is cheap once everything is
        # cached, which is what the original relied on.
        folders = ["hunyuan3d-delight-v2-0", subfolder]
        return download_repo(
            repo_id,
            CKPT_DIFFUSERS_PATH, repo_id,
            ignore_patterns=HF_DOWNLOAD_IGNORE,
            allow_patterns=[f"{f}/**" for f in folders],
        )

    def load(self, generation_mode):
        repo_id, subfolder = self.MODEL2REPO[generation_mode]

        # Fetch here so this node is what reports a network failure, but do NOT
        # build the pipeline: a live Hunyuan3DPaintPipeline holds module
        # objects and cannot cross comfy-env's worker boundary (pickle fails on
        # 'module'). Emit the JSON-safe recipe instead and let the consumer
        # materialize it in-process, the same way Load_Diffusers_Pipeline does.
        self._download_required_weights(repo_id, subfolder)

        return (model_cache.recipe(
            "hunyuan3d_v2_texgen",
            repo_id=repo_id,
            subfolder=subfolder,
        ),)


@_pipe_builder("hunyuan3d_v2_texgen")
def _build_hunyuan3d_v2_texgen(config):
    """Materialize a hunyuan3d_v2_texgen recipe. Called only on a cache miss."""
    local_repo_dir = os.path.join(CKPT_DIFFUSERS_PATH, config["repo_id"])
    pipe = Hunyuan3DPaintPipeline.from_pretrained(
        model_path=local_repo_dir,
        subfolder=config["subfolder"],
    )
    # Cast dtype, do NOT place on device: comfy/sd.py does the same
    # (`first_stage_model.to(self.vae_dtype)`, then the module goes to a
    # ModelPatcher). model_cache.managed() wraps this so load_models_gpu()
    # chooses placement -- that is what makes eviction and partial loading
    # possible. Moving here spends the VRAM before ComfyUI is asked if
    # there is room, which is where every OOM in the 20:05 log was raised.
    #
    # Keyword, not positional: this pipeline defines `to(self, device=None,
    # dtype=None)`, so a positional dtype is taken as the device. It looks
    # right because nn.Module.to() does accept a dtype positionally.
    return pipe.to(dtype=WEIGHT_DTYPE)


def resolve_hunyuan3d_v2_texgen(config):
    """hunyuan3d_v2_texgen recipe -> live pipeline, cached across runs.

    Accepts an already-built pipeline unchanged, so a graph saved before this
    node emitted recipes keeps working.
    """
    return resolve_diffusers_pipe(config)

class Hunyuan3D_V2_Paint_Model_Turbo_MV:
    """
    Texture-painting pipeline using a list of PIL images.
    If list contains 1 image -> single-view; if >1 -> multi-view mode.
    """

    CATEGORY = "Comfy3D/Algorithm"
    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("mesh",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "hunyuan3d_v2_texgen_pipe": ("DIFFUSERS_PIPE",),
                "mesh": ("TRIMESH",),
                "images": ("LIST",),
            }
        }

    @torch.no_grad()
    def run(self, hunyuan3d_v2_texgen_pipe, mesh, images):
        mesh = _wire_in(mesh)
        if not isinstance(images, list) or len(images) == 0:
            raise Exception("[Hunyuan3D_V2_Paint_Model_Turbo_MV] 'images' must be a non-empty list of PIL images")

        v_np = mesh.v.detach().cpu().numpy()
        f_np = mesh.f.detach().cpu().numpy()
        tri = trimesh.Trimesh(vertices=v_np, faces=f_np, process=False)

        hunyuan3d_v2_texgen_pipe = resolve_hunyuan3d_v2_texgen(hunyuan3d_v2_texgen_pipe)
        try:
            textured = hunyuan3d_v2_texgen_pipe(tri, images)
        except Exception as e:
            raise Exception(f"[Hunyuan3D_V2_Paint_Model_Turbo_MV] Texture generation failed: {str(e)}")

        m_out = Mesh.load_trimesh(given_mesh=textured)
        m_out.auto_normal()
        return (_wire_out(m_out),)

class Remove_Image_Background:
    """rembg background removal, matching the widget set the shipped workflows use.

    Reimplemented, not copied: five workflows call WAS Node Suite's
    `Image Rembg (Remove Background)`, and every pack these graphs depend on is
    GPL-3.0 while this one is MIT. rembg is already in this env (the Hunyuan3D
    families use it), so this is a wrapper over the same library WAS wraps.

    Widget order deliberately mirrors WAS's so a swapped node keeps its
    positional widgets_values from an existing graph.
    """

    CATEGORY = "Comfy3D/Preprocessors"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "remove_background"

    _MODELS = ["u2net", "u2netp", "u2net_human_seg", "silueta",
               "isnet-general-use", "isnet-anime"]
    _BG_COLORS = {
        "none":    None,
        "black":   (0, 0, 0, 255),
        "white":   (255, 255, 255, 255),
        "magenta": (255, 0, 255, 255),
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "transparency": ("BOOLEAN", {"default": True}),
                "model": (cls._MODELS,),
                "post_processing": ("BOOLEAN", {"default": False}),
                "only_mask": ("BOOLEAN", {"default": False}),
                "alpha_matting": ("BOOLEAN", {"default": False}),
                "alpha_matting_foreground_threshold": ("INT", {"default": 240, "min": 0, "max": 255}),
                "alpha_matting_background_threshold": ("INT", {"default": 10, "min": 0, "max": 255}),
                "alpha_matting_erode_size": ("INT", {"default": 10, "min": 0, "max": 64}),
                "background_color": (list(cls._BG_COLORS),),
            },
        }

    @torch.no_grad()
    def remove_background(self, images, transparency, model, post_processing, only_mask,
                          alpha_matting, alpha_matting_foreground_threshold,
                          alpha_matting_background_threshold, alpha_matting_erode_size,
                          background_color):
        from rembg import new_session, remove

        session = new_session(model)
        bgcolor = self._BG_COLORS[background_color]
        # A background colour and transparency are mutually exclusive: compositing
        # onto a colour is what removes the alpha. WAS lets both be set; the
        # colour wins there too, so keep that rather than surprising a graph.
        keep_alpha = transparency and bgcolor is None

        out = []
        for pil_img in torch_imgs_to_pils(images):
            result = remove(
                pil_img.convert("RGBA" if keep_alpha else "RGB"),
                session=session,
                alpha_matting=alpha_matting,
                alpha_matting_foreground_threshold=alpha_matting_foreground_threshold,
                alpha_matting_background_threshold=alpha_matting_background_threshold,
                alpha_matting_erode_size=alpha_matting_erode_size,
                only_mask=only_mask,
                post_process_mask=post_processing,
                bgcolor=bgcolor,
            )
            if only_mask:
                result = result.convert("L").convert("RGB")
            elif not keep_alpha:
                result = result.convert("RGB")
            out.append(result)

        return (pils_to_torch_imgs(out),)


class Pad_Image:
    """Pad one side of an image with a border matching that edge's mean colour.

    Reimplemented behaviour of Eden.art's `Eden_IMG_padder`, used by two
    CharacterGen workflows. That repo declares no license, so its code cannot be
    copied into this MIT pack -- but "extend one edge by a fraction of the image
    and fill with the edge's average colour" is a description, not an
    implementation.

    The fill is a single scalar averaged over the edge strip AND all channels,
    which is what upstream does -- so the border is grey-ish rather than tinted.
    A per-channel mean would look better; it is deliberately not done here so a
    shipped workflow renders as its author intended.
    """

    CATEGORY = "Comfy3D/Preprocessors"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "pad"

    _EDGE_PIXELS = 4  # strip depth the fill colour is averaged over

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "pad_fraction": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "pad_location": (["bottom", "top", "left", "right"],),
            },
        }

    @torch.no_grad()
    def pad(self, image, pad_fraction, pad_location):
        bs, h, w, c = image.shape
        n = self._EDGE_PIXELS

        if pad_location in ("top", "bottom"):
            pad_px = int(h * pad_fraction)
            strip = image[:, :n, :, :] if pad_location == "top" else image[:, -n:, :, :]
            shape, dim = (bs, pad_px, w, c), 1
        else:
            pad_px = int(w * pad_fraction)
            strip = image[:, :, :n, :] if pad_location == "left" else image[:, :, -n:, :]
            shape, dim = (bs, h, pad_px, c), 2

        if pad_px <= 0:
            return (image,)

        border = torch.full(shape, float(strip.mean()),
                            dtype=image.dtype, device=image.device)
        parts = (border, image) if pad_location in ("top", "left") else (image, border)
        return (torch.cat(parts, dim=dim),)


class Convert_Image_To_Grayscale:
    """RGB/RGBA -> grayscale, keeping 3 channels.

    Vendored in behaviour, NOT in code. Two shipped CRM_T2I_V3 texture
    workflows call Eden.art's `ConvertToGrayscale`, and that repo declares no
    license -- so its source cannot be copied into this MIT pack. The operation
    itself is ITU-R BT.601 luma plus alpha compositing, which is a standard
    formula and reimplemented here from scratch.

    Node type differs from Eden's, so a graph referencing theirs must be
    repointed at this one; installing both is harmless.
    """

    CATEGORY = "Comfy3D/Preprocessors"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "convert"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                # Luma to composite transparent pixels onto: 0 = black, 1 = white.
                "alpha_channel_convert_to": ("FLOAT", {"default": 0.0, "min": 0.0,
                                                       "max": 1.0, "step": 0.01}),
            },
        }

    @torch.no_grad()
    def convert(self, image, alpha_channel_convert_to):
        channels = image.shape[-1]
        if channels == 1:
            return (image,)
        if channels not in (3, 4):
            raise ValueError(
                f"[{self.__class__.__name__}] expected 1, 3 or 4 channels, got "
                f"{channels} (shape {tuple(image.shape)})")

        rgb = image[..., :3]
        # BT.601 luma. Not a plain channel mean: the eye is far more sensitive
        # to green, and averaging makes saturated reds and blues read as the
        # same grey.
        weights = torch.tensor([0.2989, 0.5870, 0.1140],
                               dtype=rgb.dtype, device=rgb.device)
        luma = (rgb * weights).sum(dim=-1, keepdim=True)
        gray = luma.repeat(1, 1, 1, 3)

        if channels == 4:
            alpha = image[..., 3:]
            gray = gray * alpha + alpha_channel_convert_to * (1.0 - alpha)

        return (gray,)


class Multi_Background_Remover:
    """
    Converts 1 to 4 image inputs (front/back/left/right) to a list of processed PIL images.
    Applies RGBA conversion and background removal.
    Suitable for feeding directly into ShapeGen or Paint models.
    """

    CATEGORY = "Comfy3D/Preprocessors"
    RETURN_TYPES = ("LIST",)  # List of PIL images
    RETURN_NAMES = ("images",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_front": ("IMAGE",),
            },
            "optional": {
                "image_back": ("IMAGE",),
                "image_left": ("IMAGE",),
            }
        }

    @torch.no_grad()
    def run(
        self,
        image_front,
        image_back=None,
        image_left=None,
        image_right=None
    ):
        rmbg = BackgroundRemover()

        mv_inputs = {
            k: v for k, v in {
                "front": image_front,
                "back": image_back,
                "left": image_left,
                "right": image_right
            }.items() if v is not None
        }

        images = []
        for key, tensor_img in mv_inputs.items():
            pil_img = torch_imgs_to_pils(tensor_img)[0]
            
            if pil_img.mode != "RGBA":
                pil_img = rmbg(pil_img.convert("RGB"))
            else:
                alpha = pil_img.getchannel('A')
                if alpha.getextrema()[0] == 255:
                    rgb_img = Image.new('RGB', pil_img.size, (255, 255, 255))
                    rgb_img.paste(pil_img, mask=pil_img.split()[-1] if len(pil_img.split()) == 4 else None)
                    pil_img = rmbg(rgb_img)
                    
            images.append(pil_img)

        return (images,) 

class Hunyuan3D_V2_ShapeGen_MV:
    """
    Shape generation pipeline using a list of processed PIL images.
    If len(images) == 1 -> single-view; if >1 -> multi-view dict.
    """

    CATEGORY = "Comfy3D/Algorithm"
    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("mesh",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "shapegen_pipe": ("DIFFUSERS_PIPE",),
                "images": ("LIST",),
                "seed": ("INT", {"default": 1234, "min": 0, "max": 0xffffffffffffffff}),
                "guidance_scale": ("FLOAT", {"default": 5.0, "min": 0.0, "step": 0.1}),
                "num_inference_steps": ("INT", {"default": 5, "min": 0, "tooltip": "Turbo: 5; Fast/Standard: 30–40"}),
                "octree_resolution": ("INT", {"default": 256, "min": 64}),
            }
        }

    @torch.no_grad()
    def run(
        self,
        shapegen_pipe,
        images,
        seed=1234,
        guidance_scale=5.0,
        num_inference_steps=5,
        octree_resolution=256
    ):
        # A DIFFUSERS_PIPE may arrive as a recipe dict or as a live
        # pipeline; resolve_diffusers_pipe passes the latter through.
        shapegen_pipe = resolve_diffusers_pipe(shapegen_pipe)
        if not isinstance(images, list) or len(images) == 0:
            raise Exception("[Hunyuan3D_V2_ShapeGen_MV] 'images' must be a non-empty list of PIL images")

        if len(images) == 1:
            image = images[0]
        else:
            directions = ["front", "back", "left", "right"]
            image = {k: v for k, v in zip(directions, images)}

        steps = shapegen_pipe.default_steps if num_inference_steps == 0 else num_inference_steps
        gen = torch.Generator(device=shapegen_pipe.device).manual_seed(seed)

        try:
            mesh = shapegen_pipe(
                image=image,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=gen,
                octree_resolution=octree_resolution,
                output_type="trimesh",
            )[0]
        except Exception as e:
            raise Exception(f"[Hunyuan3D_V2_ShapeGen_MV] Shape generation failed: {str(e)}")

        for fn in (FloaterRemover(), DegenerateFaceRemover(), FaceReducer()):
            mesh = fn(mesh)

        return (_wire_out(Mesh.load_trimesh(given_mesh=mesh)),)

#--------------------------------
class Load_StableGen_Trellis_Pipeline:
    CATEGORY      = "Comfy3D/Algorithm"
    RETURN_TYPES  = ("DIFFUSERS_PIPE",)
    RETURN_NAMES  = ("trellis_pipe",)
    FUNCTION      = "load"

    _REPO_ID_BASE = "Stable-X"
    CKPT_STABLEGEN_PATH = os.path.join(CKPT_DIFFUSERS_PATH, "Stable3DGen")

    _MODES = {
        "trellis-normal-v0-1": ("trellis-normal-v0-1", 12, 12),  # (repo, ss_steps, slat_steps)
    }

    @classmethod
    def INPUT_TYPES(cls):
        available_attn, available_sparse = StableGenPipelineBuilder.get_available_backends()

        return {
            "required": {
                "model_name": (list(cls._MODES.keys()),),
                "dinov2_model": (["dinov2_vitl14_reg"],),
                "use_fp16": ("BOOLEAN", {"default": True}),
                "attn_backend": (available_attn,),
                "sparse_backend": (available_sparse,),
                "spconv_algo": (["implicit_gemm", "native", "auto"],),
                "smooth_k": ("BOOLEAN", {"default": True}),
            }
        }

    @classmethod
    def _build_pipe(cls, repo: str, dinov2_model: str, use_fp16: bool, attn_backend: str, 
                   sparse_backend: str, spconv_algo: str, smooth_k: bool):
        
        return StableGenPipelineBuilder.build_trellis_pipeline(
            repo=repo,
            dinov2_model=dinov2_model,
            use_fp16=use_fp16,
            attn_backend=attn_backend,
            sparse_backend=sparse_backend,
            spconv_algo=spconv_algo,
            smooth_k=smooth_k,
            ckpt_path=cls.CKPT_STABLEGEN_PATH
        )

    def load(self, model_name, dinov2_model, use_fp16, attn_backend, sparse_backend, spconv_algo, smooth_k):
        # Recipe, not a live pipeline -- see Load_PartCrafter_Pipeline.load.
        return (model_cache.recipe(
            "stablegen_trellis",
            model_name=model_name, dinov2_model=dinov2_model, use_fp16=use_fp16,
            attn_backend=attn_backend, sparse_backend=sparse_backend,
            spconv_algo=spconv_algo, smooth_k=smooth_k),)


@_pipe_builder("stablegen_trellis")
def _build_stablegen_trellis(config):
    """Materialize a stablegen_trellis recipe. Only on a cache miss."""
    cls = Load_StableGen_Trellis_Pipeline
    repo, ss_steps, slat_steps = cls._MODES[config["model_name"]]
    pipe = cls._build_pipe(
        repo, config["dinov2_model"], config["use_fp16"], config["attn_backend"],
        config["sparse_backend"], config["spconv_algo"], config["smooth_k"])
    # Defaults the consumer reads off the pipeline.
    pipe.default_ss_steps = ss_steps
    pipe.default_slat_steps = slat_steps
    return pipe


class Load_StableGen_StableX_Pipeline:
    CATEGORY     = "Comfy3D/Algorithm"
    RETURN_TYPES = ("DIFFUSERS_PIPE",)
    RETURN_NAMES = ("stablex_pipe",)
    FUNCTION     = "load"

    _REPO_ID_BASE = "Stable-X"
    CKPT_STABLEGEN_PATH = os.path.join(CKPT_DIFFUSERS_PATH, "Stable3DGen")

    _MODES = {
        "yoso-normal-v1-8-1": "yoso-normal-v1-8-1",
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (list(cls._MODES.keys()),),
                "use_fp16": ("BOOLEAN", {"default": True}),
            }
        }

    @classmethod
    def _build_pipe(cls, repo: str, use_fp16: bool):
        return StableGenPipelineBuilder.build_stablex_pipeline(
            repo=repo,
            use_fp16=use_fp16,
            ckpt_path=cls.CKPT_STABLEGEN_PATH
        )

    def load(self, model_name, use_fp16):
        # Recipe, not a live pipeline -- see Load_PartCrafter_Pipeline.load.
        return (model_cache.recipe(
            "stablegen_stablex", model_name=model_name, use_fp16=use_fp16),)


@_pipe_builder("stablegen_stablex")
def _build_stablegen_stablex(config):
    """Materialize a stablegen_stablex recipe. Only on a cache miss."""
    cls = Load_StableGen_StableX_Pipeline
    repo = cls._MODES[config["model_name"]]
    return cls._build_pipe(repo, config["use_fp16"])


class StableGen_Trellis_Image_To_3D:
    """
    3D generation pipeline using Trellis model.
    """

    CATEGORY = "Comfy3D/Algorithm"
    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("mesh",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trellis_pipe": ("DIFFUSERS_PIPE",),
                "images": ("IMAGE", {"list": True}),
                "mode": (["single", "multi"], {"default": "single"}),
                "seed": ("INT", {"default": 1234, "min": 0, "max": 0xffffffffffffffff}),
                "ss_guidance_strength": ("FLOAT", {"default": 7.5, "min": 0.0, "step": 0.1}),
                "ss_sampling_steps": ("INT", {"default": 12, "min": 1}),
                "slat_guidance_strength": ("FLOAT", {"default": 3.0, "min": 0.0, "step": 0.1}),
                "slat_sampling_steps": ("INT", {"default": 12, "min": 1}),
                "mesh_simplify": ("FLOAT", {"default": 0.95, "min": 0.9, "max": 1.0, "step": 0.01}),
            }
        }

    @torch.no_grad()
    def run(
        self,
        trellis_pipe,
        images,
        mode="single",
        seed=1234,
        ss_guidance_strength=7.5,
        ss_sampling_steps=12,
        slat_guidance_strength=3.0,
        slat_sampling_steps=12,
        mesh_simplify=0.95
    ):
        # A DIFFUSERS_PIPE may arrive as a recipe dict or as a live
        # pipeline; resolve_diffusers_pipe passes the latter through.
        trellis_pipe = resolve_diffusers_pipe(trellis_pipe)
        if isinstance(images, torch.Tensor):
            images = torch_imgs_to_pils(images)
        
        if not isinstance(images, list) or len(images) == 0:
            raise Exception(f"[StableGen_Trellis_Image_To_3D] 'images' must be a non-empty list of PIL images. Got type: {type(images)}, len: {len(images) if hasattr(images, '__len__') else 'no len'}")

        with trellis_pipe.inference_context():
            if mode == "single":
                if len(images) > 1:
                    print(f"Warning: Single mode selected but {len(images)} images provided. Using first image.")
                image_input = images[0]
                
                pipeline_params = StableGenPipelineBuilder.create_trellis_pipeline_params(
                    seed=seed,
                    ss_sampling_steps=ss_sampling_steps,
                    ss_guidance_strength=ss_guidance_strength,
                    slat_sampling_steps=slat_sampling_steps,
                    slat_guidance_strength=slat_guidance_strength,
                    formats=["mesh"]
                )
                
                outputs = trellis_pipe.run(
                    image_input,
                    **pipeline_params
                )
            else:  
                pipeline_params = StableGenPipelineBuilder.create_trellis_pipeline_params(
                    seed=seed,
                    ss_sampling_steps=ss_sampling_steps,
                    ss_guidance_strength=ss_guidance_strength,
                    slat_sampling_steps=slat_sampling_steps,
                    slat_guidance_strength=slat_guidance_strength,
                    formats=["mesh"]
                )
                
                outputs = trellis_pipe.run_multi_image(
                    images,
                    **pipeline_params
                )
            
        try:
            mesh_output = outputs['mesh'][0]
            
            vertices = mesh_output.vertices.cpu().numpy()
            faces = mesh_output.faces.cpu().numpy()
            
            transformation_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
            vertices = vertices @ transformation_matrix
            
            tri_mesh = trimesh.Trimesh(vertices, faces, process=False)
            
            if mesh_simplify < 1.0:
                try:
                    target_faces = int(len(faces) * mesh_simplify)
                    tri_mesh = tri_mesh.simplify_quadric_decimation(target_faces)
                except Exception as e:
                    print(f"Warning: Mesh simplification failed: {e}. Using original mesh.")
            
            mesh = Mesh.load_trimesh(given_mesh=tri_mesh)
            mesh.auto_normal()
            
            return (_wire_out(mesh),)
            
        except Exception as e:
            raise Exception(f"[StableGen_Trellis_Image_To_3D] 3D generation failed: {str(e)}")


class StableGen_StableX_Process_Image:
    """
    Image processing pipeline using StableX model.
    """

    CATEGORY = "Comfy3D/Algorithm"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("processed_image",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "stablex_pipe": ("DIFFUSERS_PIPE",),
                "image": ("IMAGE",),
                "processing_resolution": ("INT", {"default": 2048, "min": 64, "max": 4096, "step": 16}),
                "controlnet_strength": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 10.0, "step": 0.01}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    @torch.no_grad()
    def run(self, stablex_pipe, image, processing_resolution=2048, controlnet_strength=1.0, seed=42):
        # A DIFFUSERS_PIPE may arrive as a recipe dict or as a live
        # pipeline; resolve_diffusers_pipe passes the latter through.
        stablex_pipe = resolve_diffusers_pipe(stablex_pipe)
        if image.dim() == 4:
            image = image.squeeze(0)
        
        image_tensor = image.permute(2, 0, 1).unsqueeze(0).to(get_device()).to(torch.float16)

        generator = torch.Generator(device=get_device()).manual_seed(seed)

        try:
            pipe_out = stablex_pipe(
                image_tensor,
                controlnet_conditioning_scale=controlnet_strength,
                processing_resolution=processing_resolution,
                generator=generator,
                output_type="pt",
            )
            
            processed = (pipe_out.prediction.clip(-1, 1) + 1) / 2
            out_tensor = processed.permute(0, 2, 3, 1).cpu().float()
            
            return (out_tensor,)
            
        except Exception as e:
            raise Exception(f"[StableGen_StableX_Process_Image] Image processing failed: {str(e)}")
# --- MV START ----
class Load_MVAdapter_IG2MV_Pipeline:
    """Loader pipeline for MV-Adapter (Image to Multi-View)"""
    CATEGORY = "Comfy3D/Algorithm"
    RETURN_TYPES = ("DIFFUSERS_PIPE",)
    RETURN_NAMES = ("mvadapter_pipe",)
    FUNCTION = "load"

    CKPT_MVADAPTER_PATH = os.path.join(CKPT_DIFFUSERS_PATH, "huanngzh", "MV-Adapter")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_model": (["stabilityai/stable-diffusion-xl-base-1.0"], 
                             {"default": "stabilityai/stable-diffusion-xl-base-1.0"}),
                "vae_model": (["madebyollin/sdxl-vae-fp16-fix", "None"], 
                             {"default": "madebyollin/sdxl-vae-fp16-fix"}),
                "adapter_path": (["huanngzh/mv-adapter"], {"default": "huanngzh/mv-adapter"}),
                "scheduler": (["ddpm"], {"default": "ddpm"}),
                "num_views": ("INT", {"default": 6, "min": 1, "max": 16}),
                "use_fp16": ("BOOLEAN", {"default": True}),
                "use_mmgp": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "lora_model": ("STRING", {"default": ""}),
            }
        }
        
    @classmethod
    def load(cls, base_model, vae_model, adapter_path, scheduler, num_views,
            use_fp16, use_mmgp, lora_model=""):
        # Recipe, not a live pipeline -- see Load_PartCrafter_Pipeline.load.
        return (model_cache.recipe(
            "mvadapter_ig2mv",
            base_model=base_model, vae_model=vae_model, lora_model=lora_model,
            adapter_path=adapter_path, scheduler=scheduler, num_views=num_views,
            use_fp16=use_fp16, use_mmgp=use_mmgp),)


def _mvadapter_kwargs(config, cls):
    """Recipe fields -> mvadapter_prepare_*_pipeline kwargs.

    dtype and the None-for-"None" fixups live here rather than in the node, so
    the recipe stays JSON-safe: torch.float16 is not serialisable, use_fp16 is.
    """
    return {
        "base_model": config["base_model"],
        "vae_model": None if config["vae_model"] == "None" else config["vae_model"],
        "lora_model": config["lora_model"] or None,
        "adapter_path": config["adapter_path"],
        "scheduler": config["scheduler"],
        "num_views": config["num_views"],
        "device": get_device_str(),
        "dtype": torch.float16 if config["use_fp16"] else torch.float32,
        "use_mmgp": config["use_mmgp"],
        "adapter_local_path": cls.CKPT_MVADAPTER_PATH,
    }


@_pipe_builder("mvadapter_ig2mv")
def _build_mvadapter_ig2mv(config):
    """Materialize an mvadapter_ig2mv recipe. Only on a cache miss."""
    kwargs = _mvadapter_kwargs(config, Load_MVAdapter_IG2MV_Pipeline)
    pipe = mvadapter_prepare_pipeline(**kwargs)
    print("MV-Adapter IG2MV pipeline loaded successfully")
    return pipe

class MVAdapter_IG2MV:
    """Generate multi-view images from single image and 3D mesh"""
    CATEGORY = "Comfy3D/Algorithm"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("multiview_images",)
    FUNCTION = "run"

    @classmethod  
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mvadapter_pipe": ("DIFFUSERS_PIPE",),
                "mesh_file": _mesh_input_combo(),
                "reference_image": ("IMAGE",),
                "prompt": ("STRING", {"default": "high quality", "multiline": True}),
                "negative_prompt": ("STRING", {"default": "watermark, ugly, deformed, noisy, blurry, low contrast", "multiline": True}),
                "num_inference_steps": ("INT", {"default": 50, "min": 1, "max": 200}),
                "guidance_scale": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "reference_conditioning_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "height": ("INT", {"default": 768, "min": 256, "max": 2048, "step": 8}),
                "width": ("INT", {"default": 768, "min": 256, "max": 2048, "step": 8}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff}),
                "remove_background": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "lora_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                # Wins when set; see _resolve_input_mesh.
                "mesh_path": ("STRING", {"default": ""}),
            }
        }

    def run(self, mvadapter_pipe, mesh_file, reference_image, prompt, negative_prompt, 
            num_inference_steps, guidance_scale, reference_conditioning_scale,
            height, width, seed, remove_background, lora_scale=1.0, mesh_path=""):
        
        # A DIFFUSERS_PIPE may arrive as a recipe dict or as a live
        # pipeline; resolve_diffusers_pipe passes the latter through.
        mvadapter_pipe = resolve_diffusers_pipe(mvadapter_pipe)
        if isinstance(reference_image, torch.Tensor):
            reference_images = torch_imgs_to_pils(reference_image)
            reference_image = reference_images[0]
        
        mesh_path = _resolve_input_mesh(mesh_path, mesh_file)
        
        num_views = 6 
        images, pos_images, normal_images, processed_ref_image = mvadapter_run_pipeline(
            pipe=mvadapter_pipe,
            mesh_path=mesh_path,
            num_views=num_views,
            text=prompt,
            image=reference_image,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            reference_conditioning_scale=reference_conditioning_scale,
            negative_prompt=negative_prompt,
            lora_scale=lora_scale,
            device=get_device_str(),
        )
        
        return_images = pils_to_torch_imgs(images, device=get_device_str())
        return (return_images,)

class Load_MVAdapter_TG2MV_Pipeline:
    """Loader pipeline for MV-Adapter Text-Guided to Multi-View (TG2MV)"""
    CATEGORY = "Comfy3D/Algorithm"
    RETURN_TYPES = ("DIFFUSERS_PIPE",)
    RETURN_NAMES = ("mvadapter_tg2mv_pipe",)
    FUNCTION = "load"

    CKPT_MVADAPTER_PATH = os.path.join(CKPT_DIFFUSERS_PATH, "huanngzh", "MV-Adapter")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_model": (["stabilityai/stable-diffusion-xl-base-1.0"], 
                             {"default": "stabilityai/stable-diffusion-xl-base-1.0"}),
                "vae_model": (["madebyollin/sdxl-vae-fp16-fix", "None"], 
                             {"default": "madebyollin/sdxl-vae-fp16-fix"}),
                "adapter_path": (["huanngzh/mv-adapter"], {"default": "huanngzh/mv-adapter"}),
                "scheduler": (["ddpm"], {"default": "ddpm"}),
                "num_views": ("INT", {"default": 6, "min": 1, "max": 16}),
                "use_fp16": ("BOOLEAN", {"default": True}),
                "use_mmgp": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "lora_model": ("STRING", {"default": ""}),
            }
        }

    @classmethod
    def load(cls, base_model, vae_model, adapter_path, scheduler, num_views,
             use_fp16, use_mmgp, lora_model=""):
        # Recipe, not a live pipeline -- see Load_PartCrafter_Pipeline.load.
        return (model_cache.recipe(
            "mvadapter_tg2mv",
            base_model=base_model, vae_model=vae_model, lora_model=lora_model,
            adapter_path=adapter_path, scheduler=scheduler, num_views=num_views,
            use_fp16=use_fp16, use_mmgp=use_mmgp),)


@_pipe_builder("mvadapter_tg2mv")
def _build_mvadapter_tg2mv(config):
    """Materialize an mvadapter_tg2mv recipe. Only on a cache miss."""
    # Was `if torch.cuda.is_available(): torch.cuda.empty_cache()`. comfy's
    # version also covers MPS/XPU and syncs first. Imported here, not at module
    # scope, to keep this module importable outside ComfyUI.
    import comfy.model_management
    comfy.model_management.soft_empty_cache()
    kwargs = _mvadapter_kwargs(config, Load_MVAdapter_TG2MV_Pipeline)
    pipe = mvadapter_prepare_tg2mv_pipeline(**kwargs)
    print("MV-Adapter TG2MV pipeline loaded successfully")
    return pipe


class MVAdapter_TG2MV:
    """Generate multi-view images from text prompt using 3D mesh guidance"""
    CATEGORY = "Comfy3D/Algorithm"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("multiview_images",)
    FUNCTION = "run"

    @classmethod  
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mvadapter_tg2mv_pipe": ("DIFFUSERS_PIPE",),
                "mesh_path": ("STRING", {"default": ""}),
                "prompt": ("STRING", {"default": "a high quality 3D model", "multiline": True}),
                "negative_prompt": ("STRING", {"default": "watermark, ugly, deformed, noisy, blurry, low contrast", "multiline": True}),
                "num_views": ("INT", {"default": 6, "min": 1, "max": 16}),
                "num_inference_steps": ("INT", {"default": 50, "min": 1, "max": 200}),
                "guidance_scale": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "height": ("INT", {"default": 768, "min": 256, "max": 2048, "step": 8}),
                "width": ("INT", {"default": 768, "min": 256, "max": 2048, "step": 8}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "lora_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
            }
        }

    def run(self, mvadapter_tg2mv_pipe, mesh_path, prompt, negative_prompt, num_views,
            num_inference_steps, guidance_scale, height, width, seed, lora_scale=1.0):
        
        # A DIFFUSERS_PIPE may arrive as a recipe dict or as a live
        # pipeline; resolve_diffusers_pipe passes the latter through.
        mvadapter_tg2mv_pipe = resolve_diffusers_pipe(mvadapter_tg2mv_pipe)
        if not mesh_path or not os.path.exists(mesh_path):
            raise ValueError(f"Mesh path does not exist: {mesh_path}")
        
        images, pos_images, normal_images = mvadapter_run_tg2mv_pipeline(
            pipe=mvadapter_tg2mv_pipe,
            mesh_path=mesh_path,
            num_views=num_views,
            text=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            negative_prompt=negative_prompt,
            lora_scale=lora_scale,
            device=get_device_str(),
        )
        
        
        return_images = pils_to_torch_imgs(images, device=get_device_str())
        return (return_images,)
            
class Load_MVAdapter_Texture_Pipeline:
    """Load texture projection pipeline for MVAdapter"""
    CATEGORY = "Comfy3D/Algorithm"
    RETURN_TYPES = ("TEXTURE_PIPE",)
    RETURN_NAMES = ("texture_pipeline",)
    FUNCTION = "load"

    TEXTURE_CKPT_DIR = os.path.join(CKPT_DIFFUSERS_PATH, "huanngzh/MV-Adapter")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "upscaler_ckpt_name": ("STRING", {"default": "RealESRGAN_x2plus.pth"}),
                "inpaint_ckpt_name": ("STRING", {"default": "big-lama.pt"}),
                "use_mmgp": ("BOOLEAN", {"default": False}),
                "auto_download": ("BOOLEAN", {"default": True}),
            }
        }

    @classmethod
    def load(cls, upscaler_ckpt_name, inpaint_ckpt_name, use_mmgp, auto_download):
        
        upscaler_ckpt_path = os.path.join(cls.TEXTURE_CKPT_DIR, upscaler_ckpt_name) if upscaler_ckpt_name.strip() else None
        inpaint_ckpt_path = os.path.join(cls.TEXTURE_CKPT_DIR, inpaint_ckpt_name) if inpaint_ckpt_name.strip() else None
        
        if auto_download:
            download_texture_checkpoints(cls.TEXTURE_CKPT_DIR, upscaler_ckpt_path, inpaint_ckpt_path)
        
        texture_pipe = mvadapter_prepare_texture_pipeline(
            upscaler_ckpt_path=upscaler_ckpt_path,
            inpaint_ckpt_path=inpaint_ckpt_path,
            device=get_device_str(),
            use_mmgp=use_mmgp,
        )
        
        print("Texture pipeline loaded successfully")
        return (texture_pipe,)
    

class MVAdapter_Texture_Projection:
    """Project grid image onto 3D mesh using pre-loaded texture pipeline"""
    CATEGORY = "Comfy3D/Algorithm"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("shaded_model_path", "pbr_model_path") 
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "texture_pipeline": ("TEXTURE_PIPE",),
                "grid_image": ("IMAGE",),
                "mesh_path": ("STRING", {"default": ""}),
                "save_dir": ("STRING", {"default": "./output"}),
                "save_name": ("STRING", {"default": "textured_model"}),
                "uv_size": ("INT", {"default": 4096, "min": 512, "max": 8192, "step": 256}),
                "view_upscale": ("BOOLEAN", {"default": True}),
                "inpaint_mode": (["none", "uv", "view"], {"default": "view"}),
                "uv_unwarp": ("BOOLEAN", {"default": True}),
                "preprocess_mesh": ("BOOLEAN", {"default": False}),
                "move_to_center": ("BOOLEAN", {"default": False}),
                "front_x": ("BOOLEAN", {"default": True}),
                "create_pbr_model": ("BOOLEAN", {"default": True}),
                "apply_dilate": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "camera_azimuth_deg": ("STRING", {"default": "0,90,180,270,180,180"}),
                "camera_elevation_deg": ("STRING", {"default": "0,0,0,0,89.99,-89.99"}),
                "camera_distance": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "camera_ortho_scale": ("FLOAT", {"default": 1.1, "min": 0.1, "max": 5.0, "step": 0.1}),
                "debug_mode": ("BOOLEAN", {"default": False}),
            }
        }

    def run(self, texture_pipeline, grid_image, mesh_path, save_dir, save_name, 
            uv_size, view_upscale, inpaint_mode, uv_unwarp, preprocess_mesh, 
            move_to_center, front_x, create_pbr_model, apply_dilate,
            camera_azimuth_deg="0,90,180,270,180,180",
            camera_elevation_deg="0,0,0,0,89.99,-89.99", 
            camera_distance=1.0, camera_ortho_scale=1.1, debug_mode=False):
        
        if isinstance(grid_image, torch.Tensor):
            pil_grid = torch_imgs_to_pils(grid_image)[0]  # Get first (and only) image
        else:
            raise ValueError("grid_image must be torch.Tensor")
        
        if not mesh_path or not os.path.exists(mesh_path):
            raise ValueError(f"Mesh file does not exist: {mesh_path}")
        
        save_dir = os.path.abspath(save_dir)
        os.makedirs(save_dir, exist_ok=True)
        
        try:
            azimuth_deg = [float(x.strip()) for x in camera_azimuth_deg.split(",")]
            elevation_deg = [float(x.strip()) for x in camera_elevation_deg.split(",")]
        except:
            azimuth_deg = [0, 90, 180, 270, 180, 180]
            elevation_deg = [0, 0, 0, 0, 89.99, -89.99]
        
        azimuth_deg_corrected = [x - 90 for x in azimuth_deg]
        
        temp_grid_path = os.path.join(save_dir, f"{save_name}_temp_grid.png")
        pil_grid.save(temp_grid_path)
        
        try:
            from .Gen_3D_Modules.MV_Adapter.mvadapter.pipelines.pipeline_texture import ModProcessConfig
            
            rgb_process_config = ModProcessConfig(
                view_upscale=view_upscale,
                inpaint_mode=inpaint_mode
            )
            
            base_color_process_config = ModProcessConfig(
                view_upscale=view_upscale,
                inpaint_mode=inpaint_mode
            )
            
            texture_args = {
                "mesh_path": mesh_path,
                "save_dir": save_dir,
                "save_name": save_name,
                "move_to_center": move_to_center,
                "front_x": front_x,
                "uv_unwarp": uv_unwarp,
                "preprocess_mesh": preprocess_mesh,
                "uv_size": uv_size,
                "rgb_path": temp_grid_path,
                "rgb_process_config": rgb_process_config,
                "camera_elevation_deg": elevation_deg,
                "camera_azimuth_deg": azimuth_deg_corrected,
                "camera_distance": camera_distance,
                "camera_ortho_scale": camera_ortho_scale,
                "debug_mode": debug_mode,
                "apply_dilate": apply_dilate,
            }
            
            if create_pbr_model:
                texture_args.update({
                    "base_color_path": temp_grid_path,
                    "base_color_process_config": base_color_process_config,
                })
            
            output = texture_pipeline(**texture_args)
            
            if os.path.exists(temp_grid_path):
                os.remove(temp_grid_path)
            
            shaded_path = os.path.abspath(output.shaded_model_save_path) if output.shaded_model_save_path else ""
            pbr_path = os.path.abspath(output.pbr_model_save_path) if output.pbr_model_save_path else ""
            
            print(f"Texturing from grid completed:")
            print(f"  Shaded model: {shaded_path}")
            print(f"  PBR model: {pbr_path}")
                
            return (shaded_path, pbr_path)
            
        except Exception as e:
            if os.path.exists(temp_grid_path):
                os.remove(temp_grid_path)
            raise e

class Load_Hunyuan3D_21_ShapeGen_Pipeline:

    # Announces in the UI that this node fetches weights if they are absent.
    DISPLAY_NAME = "(Down)Load Hunyuan3D 21 ShapeGen Pipeline"

    """Load Hunyuan3D-2.1 Shape Generation Pipeline"""
    
    CATEGORY = "Comfy3D/Algorithm/Hunyuan3D-2.1"
    RETURN_TYPES = ("DIFFUSERS_PIPE",)
    RETURN_NAMES = ("shapegen_pipe",)
    FUNCTION = "load"

    # tencent/Hunyuan3D-2.1 ships this checkpoint only as a .ckpt pickle.
    # niknah/Hunyuan3D-2.1-safetensors is a straight conversion of it, in the
    # same directory layout, and publishes the ckpt2safetensors.py that made it.
    # Verified against the safetensors header: 1601 tensors, all F16, 7.37 GB --
    # byte-identical in size to tencent's .ckpt, with the same key names.
    #
    # It is a single file holding all three sub-models, split by key prefix and
    # reassembled in pipelines.py: model 6.10 GB, vae 0.66 GB, conditioner
    # 0.61 GB. The separate hunyuan3d-vae-v2-1/model.fp16.ckpt this used to
    # fetch was never read -- 0.66 GB downloaded and ignored.
    #
    # Third-party: an individual's repo, not tencent's. Mirror it under an
    # account you control if that matters more than the pickle does.
    _REPO_ID = "niknah/Hunyuan3D-2.1-safetensors"
    _CKPT = "hunyuan3d-dit-v2-1/model.fp16.safetensors"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "subfolder": (["hunyuan3d-dit-v2-1"], {"default": "hunyuan3d-dit-v2-1"}),
            }
        }

    @staticmethod
    def _ensure_weights(subfolder: str):
        cls = Load_Hunyuan3D_21_ShapeGen_Pipeline
        return download_files(
            cls._REPO_ID,
            [cls._CKPT],
            CKPT_DIFFUSERS_PATH, "tencent/Hunyuan3D-2_1",
        )

    def load(self, subfolder):
        # Recipe, not a live pipeline -- see Load_PartCrafter_Pipeline.load.
        self._ensure_weights(subfolder)
        return (model_cache.recipe("hunyuan3d_21_shapegen", subfolder=subfolder),)


@_pipe_builder("hunyuan3d_21_shapegen")
def _build_hunyuan3d_21_shapegen(config):
    """Materialize a hunyuan3d_21_shapegen recipe. Only on a cache miss."""
    subfolder = config["subfolder"]
    base_dir = Load_Hunyuan3D_21_ShapeGen_Pipeline._ensure_weights(subfolder)
    return Hunyuan3DDiTFlowMatchingPipeline_2_1.from_pretrained(
        base_dir,
        subfolder=subfolder,
        use_safetensors=True,
        device=get_device(),
    )

class Load_Hunyuan3D_21_TexGen_Pipeline:

    # Announces in the UI that this node fetches weights if they are absent.
    DISPLAY_NAME = "(Down)Load Hunyuan3D 21 TexGen Pipeline"

    """Load Hunyuan3D-2.1 Texture Generation Pipeline"""
    
    CATEGORY = "Comfy3D/Algorithm/Hunyuan3D-2.1"
    RETURN_TYPES = ("DIFFUSERS_PIPE",)
    RETURN_NAMES = ("texgen_pipe",)
    FUNCTION = "load"

    _REPO_ID_BASE = "tencent"
    _REPO_NAME = "Hunyuan3D-2.1"

    # Pipeline cache: { (max_view, res, mmgp) : pipeline }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "max_num_view": ("INT", {"default": 8, "min": 4, "max": 12}),
                "resolution": ("INT", {"default": 768, "min": 512, "max": 1024, "step": 256}),
                "enable_mmgp": ("BOOLEAN", {"default": True}),
            }
        }

    @staticmethod 
    def _ensure_weights():
        cls = Load_Hunyuan3D_21_TexGen_Pipeline
        safe_repo_name = cls._REPO_NAME.replace(".", "_")
        target_folder = "hunyuan3d-paintpbr-v2-1"
        return download_repo(
            f"{cls._REPO_ID_BASE}/{cls._REPO_NAME}",
            CKPT_DIFFUSERS_PATH, f"{cls._REPO_ID_BASE}/{safe_repo_name}",
            ignore_patterns=HF_DOWNLOAD_IGNORE,
            allow_patterns=[f"{target_folder}/**"],
            requires=[
                f"{target_folder}/image_encoder/model.safetensors",
                f"{target_folder}/text_encoder/pytorch_model.bin",
                f"{target_folder}/unet/diffusion_pytorch_model.bin",
                f"{target_folder}/vae/diffusion_pytorch_model.bin",
            ],
        )


    def load(self, max_num_view, resolution, enable_mmgp):
        # No private cache: model_cache dedupes on the frozen recipe, which is
        # what the old (max_num_view, resolution, enable_mmgp) key was doing by
        # hand -- and its dict was invisible to ComfyUI's memory manager.
        self._ensure_weights()
        return (model_cache.recipe(
            "hunyuan3d_21_texgen",
            max_num_view=max_num_view, resolution=resolution,
            enable_mmgp=enable_mmgp),)


@_pipe_builder("hunyuan3d_21_texgen")
def _build_hunyuan3d_21_texgen(config):
    """Materialize a hunyuan3d_21_texgen recipe. Only on a cache miss."""
    Load_Hunyuan3D_21_TexGen_Pipeline._ensure_weights()

    conf = Hunyuan3DPaintConfig_2_1(max_num_view=config["max_num_view"],
                                    resolution=config["resolution"])
    conf.multiview_cfg_path = os.path.join(NODES_PATH, "Gen_3D_Modules/Hunyuan3D_2_1/hy3dpaint/cfgs/hunyuan-paint-pbr.yaml")
    conf.custom_pipeline = os.path.join(NODES_PATH, "Gen_3D_Modules/Hunyuan3D_2_1/hy3dpaint/hunyuanpaintpbr")

    pipeline = Hunyuan3DPaintPipeline_2_1(conf)

    if config["enable_mmgp"]:
        try:
            # Imported here, not at module scope: importing mmgp replaces
            # safetensors.safe_open process-wide, which breaks every
            # transformers safetensors load. See shared_utils/mmgp_compat.
            from mmgp import offload, profile_type
            from .shared_utils.mmgp_compat import keep_safe_open_compatible
            keep_safe_open_compatible()

            core_pipe = pipeline.models["multiview_model"].pipeline
            offload.profile(core_pipe, profile_type.LowRAM_LowVRAM)
            print("mmgp optimization enabled for texture pipeline")
        except Exception as e:
            print(f"[mmgp] Failed to apply optimization for texture: {e}")
    else:
        print("mmgp optimization disabled for texture pipeline")

    return pipeline

class Hunyuan3D_21_ShapeGen:
    """Hunyuan3D-2.1 Shape Generation"""
    
    CATEGORY = "Comfy3D/Algorithm/Hunyuan3D-2.1"
    RETURN_TYPES = ("TRIMESH", "IMAGE")
    RETURN_NAMES = ("mesh", "processed_image")
    FUNCTION = "generate"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "shapegen_pipe": ("DIFFUSERS_PIPE",),
                "image": ("IMAGE",),
                "seed": ("INT", {"default": 1234, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 30, "min": 1, "max": 100}),
                "guidance_scale": ("FLOAT", {"default": 7.5, "min": 0.0, "step": 0.1}),
                "octree_resolution": ("INT", {"default": 256, "min": 64, "max": 512}),
                "remove_background": ("BOOLEAN", {"default": True}),
            }
        }

    @torch.no_grad()
    def generate(self, shapegen_pipe, image, seed, steps, guidance_scale, octree_resolution, remove_background):
        # A DIFFUSERS_PIPE may arrive as a recipe dict or as a live
        # pipeline; resolve_diffusers_pipe passes the latter through.
        shapegen_pipe = resolve_diffusers_pipe(shapegen_pipe)
        pil_image = torch_imgs_to_pils(image)[0].convert("RGBA")
        
        if remove_background or pil_image.mode == "RGB":
            rmbg_worker = BackgroundRemover_2_1()
            pil_image = rmbg_worker(pil_image.convert('RGB'))
            del rmbg_worker

        generator = torch.Generator(device=shapegen_pipe.device)
        generator = generator.manual_seed(int(seed))
        
        outputs = shapegen_pipe(
            image=pil_image,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
            octree_resolution=octree_resolution,
            num_chunks=200000,
            output_type='mesh'
        )
        
        mesh = export_to_trimesh_2_1(outputs)[0]
        
        face_reduce_worker = FaceReducer_2_1()
        mesh = face_reduce_worker(mesh)
        del face_reduce_worker
        
        # No hand-rolled cleanup here. This used to tear `unet`, `vae` and
        # `scheduler` off the pipeline after every run -- but that object is the
        # one model_cache holds, so the next execution got a cache hit on a
        # gutted pipeline and died at pipelines.py:741 reaching for
        # self.scheduler. It worked exactly once per ComfyUI start.
        #
        # ComfyUI owns this VRAM now: the pipeline is registered with a
        # ModelPatcher (see model_cache.managed), so unload_all_models() and
        # ordinary eviction reclaim it without destroying the cached object.
        del outputs

        mesh_out = Mesh.load_trimesh(given_mesh=mesh)
        mesh_out.auto_normal()
        
        processed_image_tensor = pils_to_torch_imgs([pil_image])
        
        return (_wire_out(mesh_out), processed_image_tensor)

class Hunyuan3D_21_TexGen:
    """Hunyuan3D-2.1 Texture Generation"""
    
    CATEGORY = "Comfy3D/Algorithm/Hunyuan3D-2.1"
    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("textured_mesh",)
    FUNCTION = "generate"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "texgen_pipe": ("DIFFUSERS_PIPE",),
                "mesh_file": _mesh_input_combo(),
                "image": ("IMAGE",),
                "create_pbr": ("BOOLEAN", {"default": True}),
                "use_remesh": ("BOOLEAN", {"default": False}),
            },
            # Optional, and it wins when set: a COMBO cannot receive a STRING
            # link, so this is how a graph wires in a path from another node.
            "optional": {
                "mesh_path": ("STRING", {"default": ""}),
            },
        }

    @torch.no_grad()
    def generate(self, texgen_pipe, mesh_file, image, create_pbr, use_remesh, mesh_path=""):
        # A DIFFUSERS_PIPE may arrive as a recipe dict or as a live
        # pipeline; resolve_diffusers_pipe passes the latter through.
        texgen_pipe = resolve_diffusers_pipe(texgen_pipe)
        mesh_path = _resolve_input_mesh(mesh_path, mesh_file)

        pil_image = torch_imgs_to_pils(image)[0]
        
        # Save files to output/Hun2-1 directory
        output_dir = "output/Hun2-1"
        os.makedirs(output_dir, exist_ok=True)
        
        image_path = os.path.join(output_dir, "hunyuan_input.png")
        output_path = os.path.join(output_dir, "hunyuan_output.obj")
        
        try:
            pil_image.save(image_path)
            
            result_path = texgen_pipe(
                mesh_path=mesh_path,
                image_path=image_path,
                output_mesh_path=output_path,
                save_glb=True,  # Always create GLB
                use_remesh=use_remesh
            )
            
            mesh_out = None
            
            if create_pbr:
                glb_path = result_path.replace(".obj", ".glb")
                
                if not os.path.exists(glb_path):
                    base_path = os.path.splitext(result_path)[0]
                    textures_dict = {
                        'albedo': f"{base_path}.jpg",
                    }
                    
                    metallic_path = f"{base_path}_metallic.jpg"
                    roughness_path = f"{base_path}_roughness.jpg"
                    normal_path = f"{base_path}_normal.jpg"
                    
                    if os.path.exists(metallic_path):
                        textures_dict['metallic'] = metallic_path
                    if os.path.exists(roughness_path):
                        textures_dict['roughness'] = roughness_path
                    if os.path.exists(normal_path):
                        textures_dict['normal'] = normal_path
                    
                    try:
                        create_glb_with_pbr_materials_2_1(result_path, textures_dict, glb_path)
                        print(f"Created GLB with full PBR materials: {glb_path}")
                    except Exception as e:
                        print(f"Warning: Failed to create GLB with PBR materials: {e}")
                        # Fallback to basic conversion
                        from .Gen_3D_Modules.Hunyuan3D_2_1.hy3dpaint.DifferentiableRenderer.mesh_utils import convert_obj_to_glb
                        convert_obj_to_glb(result_path, glb_path)
                        print(f"Created GLB with basic conversion: {glb_path}")
                
                if os.path.exists(glb_path):
                    try:
                        glb_scene = trimesh.load(glb_path)
                        
                        if hasattr(glb_scene, 'geometry') and glb_scene.geometry:
                            mesh_name = list(glb_scene.geometry.keys())[0]
                            glb_mesh = glb_scene.geometry[mesh_name]
                        else:
                            glb_mesh = glb_scene
                        
                        mesh_out = Mesh.load_trimesh(given_mesh=glb_mesh)
                        mesh_out.auto_normal()
                        print(f"Loaded GLB mesh with PBR materials from: {glb_path}")
                    except Exception as e:
                        print(f"Warning: Failed to load GLB mesh: {e}")
                        print(f"GLB path was: {glb_path}")
            
            # If PBR failed or not requested, load regular textured mesh
            if mesh_out is None:
                textured_mesh = trimesh.load(result_path)
                mesh_out = Mesh.load_trimesh(given_mesh=textured_mesh)
                mesh_out.auto_normal()
                if create_pbr:
                    print("Warning: PBR creation failed, loaded regular textured mesh")
                else:
                    print("Loaded regular textured mesh")
            
            return (_wire_out(mesh_out),)
            
        finally:
            # Clean up files
            torch.cuda.empty_cache()
            gc.collect()
            
            for file_path in [image_path, output_path]:
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                except:
                    pass

# --------------------- PARTCRAFTER ---------------------

class Load_PartCrafter_Pipeline:

    # Announces in the UI that this node fetches weights if they are absent.
    DISPLAY_NAME = "(Down)Load PartCrafter Pipeline"

    """Load PartCrafter Pipeline"""
    
    CATEGORY = "Comfy3D/Algorithm/PartCrafter"
    RETURN_TYPES = ("DIFFUSERS_PIPE",)
    RETURN_NAMES = ("partcrafter_pipe",)
    FUNCTION = "load"

    _REPO_ID = "wgsxm/PartCrafter"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
        }

    @staticmethod
    def _ensure_weights():
        # Weights only; configs and any custom_pipeline come from git, which is
        # what HF_DOWNLOAD_IGNORE excludes from the snapshot.
        return download_repo(
            Load_PartCrafter_Pipeline._REPO_ID,
            CKPT_DIFFUSERS_PATH, Load_PartCrafter_Pipeline._REPO_ID,
            ignore_patterns=HF_DOWNLOAD_IGNORE,
            requires=[
                "diffusion_pytorch_model.safetensors",
                "diffusion_pytorch_model.bin",
            ],
        )

    def load(self):
        # Fetch here so this node reports a network failure, but emit a recipe:
        # a live pipeline holds module objects and cannot cross comfy-env's
        # worker boundary. resolve_diffusers_pipe builds it in-process.
        self._ensure_weights()
        return (model_cache.recipe("partcrafter"),)


@_pipe_builder("partcrafter")
def _build_partcrafter(config):
    """Materialize a partcrafter recipe. Called only on a cache miss."""
    base_dir = Load_PartCrafter_Pipeline._ensure_weights()
    _alias_model_index_libraries(base_dir)
    # Cast dtype, do NOT place on device: comfy/sd.py does the same
    # (`first_stage_model.to(self.vae_dtype)`, then the module goes to a
    # ModelPatcher). model_cache.managed() wraps this so load_models_gpu()
    # chooses placement -- that is what makes eviction and partial loading
    # possible. Moving here spends the VRAM before ComfyUI is asked if
    # there is room, which is where every OOM in the 20:05 log was raised.
    pipeline = PartCrafterPipeline.from_pretrained(base_dir).to(WEIGHT_DTYPE)
    print("PartCrafter pipeline loaded")
    return pipeline


class Load_PartCrafter_Scene_Pipeline:

    # Announces in the UI that this node fetches weights if they are absent.
    DISPLAY_NAME = "(Down)Load PartCrafter Scene Pipeline"

    """Load PartCrafter Scene Pipeline"""
    
    CATEGORY = "Comfy3D/Algorithm/PartCrafter"
    RETURN_TYPES = ("DIFFUSERS_PIPE",)
    RETURN_NAMES = ("partcrafter_scene_pipe",)
    FUNCTION = "load"

    _REPO_ID = "wgsxm/PartCrafter-Scene"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
        }

    @staticmethod
    def _ensure_weights():
        # Weights only; configs and any custom_pipeline come from git, which is
        # what HF_DOWNLOAD_IGNORE excludes from the snapshot.
        return download_repo(
            Load_PartCrafter_Scene_Pipeline._REPO_ID,
            CKPT_DIFFUSERS_PATH, Load_PartCrafter_Scene_Pipeline._REPO_ID,
            ignore_patterns=HF_DOWNLOAD_IGNORE,
            requires=[
                "model_index.json",
                "transformer/diffusion_pytorch_model.safetensors",
                "vae/diffusion_pytorch_model.safetensors",
            ],
        )

    def load(self):
        # Recipe, not a live pipeline -- see Load_PartCrafter_Pipeline.load.
        self._ensure_weights()
        return (model_cache.recipe("partcrafter_scene"),)


@_pipe_builder("partcrafter_scene")
def _build_partcrafter_scene(config):
    """Materialize a partcrafter_scene recipe. Only on a cache miss."""
    base_dir = Load_PartCrafter_Scene_Pipeline._ensure_weights()
    _alias_model_index_libraries(base_dir)
    # Cast dtype, do NOT place on device: comfy/sd.py does the same
    # (`first_stage_model.to(self.vae_dtype)`, then the module goes to a
    # ModelPatcher). model_cache.managed() wraps this so load_models_gpu()
    # chooses placement -- that is what makes eviction and partial loading
    # possible. Moving here spends the VRAM before ComfyUI is asked if
    # there is room, which is where every OOM in the 20:05 log was raised.
    pipeline = PartCrafterPipeline.from_pretrained(base_dir).to(WEIGHT_DTYPE)
    print("PartCrafter-Scene pipeline loaded")
    return pipeline


class PartCrafter_Generate:
    """PartCrafter Generation - Creates multi-part 3D scenes with colored components"""
    
    CATEGORY = "Comfy3D/Algorithm/PartCrafter"
    # `mesh` is APPENDED, never inserted: slots 0-2 keep their indices, so the
    # shipped workflow and every saved graph keep their links.
    RETURN_TYPES = ("STRING", "STRING", "IMAGE", "TRIMESH")
    RETURN_NAMES = ("parts_zip_path", "glb_mesh_path", "processed_image", "mesh")
    FUNCTION = "generate"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "partcrafter_pipe": ("DIFFUSERS_PIPE",),
                "image": ("IMAGE",),
                "num_parts": ("INT", {"default": 4, "min": 1, "max": 16}),
                "seed": ("INT", {"default": 1234, "min": 0, "max": 0xffffffffffffffff}),
                "num_tokens": ("INT", {"default": 1024, "min": 256, "max": 2048}),
                "num_inference_steps": ("INT", {"default": 50, "min": 1, "max": 100}),
                "guidance_scale": ("FLOAT", {"default": 7.0, "min": 0.0, "step": 0.1}),
                "max_num_expanded_coords": ("INT", {"default": 1000000000, "min": 1000, "max": 10000000000}),
                "use_flash_decoder": ("BOOLEAN", {"default": False}),
                "remove_background": ("BOOLEAN", {"default": True}),
                "sampling_version": ("INT", {"default": 1, "min": 1, "max": 2}),
            }
        }

    @torch.no_grad()
    def generate(self, partcrafter_pipe, image, num_parts, seed, num_tokens, num_inference_steps, 
                 guidance_scale, max_num_expanded_coords, use_flash_decoder, remove_background, sampling_version):
        
        # Convert image
        # A DIFFUSERS_PIPE may arrive as a recipe dict or as a live
        # pipeline; resolve_diffusers_pipe passes the latter through.
        partcrafter_pipe = resolve_diffusers_pipe(partcrafter_pipe)
        pil_image = torch_imgs_to_pils(image)[0]
        
        # Remove background if needed
        if remove_background:
            rmbg_worker = BackgroundRemover_2_1()
            if pil_image.mode == "RGBA":
                pil_image = pil_image.convert('RGB')
            pil_image = rmbg_worker(pil_image, background_color=(255, 255, 255))
            del rmbg_worker
        
        # Set sampling version
        if hasattr(partcrafter_pipe.vae, 'set_sampling_version'):
            partcrafter_pipe.vae.set_sampling_version(sampling_version)
        
        # Generation
        generator = torch.Generator(device=partcrafter_pipe.device)
        generator = generator.manual_seed(int(seed))
        
        outputs = partcrafter_pipe(
            image=[pil_image] * num_parts,
            attention_kwargs={"num_parts": num_parts},
            num_tokens=num_tokens,
            generator=generator,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            max_num_expanded_coords=max_num_expanded_coords,
            use_flash_decoder=use_flash_decoder,
        ).meshes
        
        # Ensure no None outputs 
        for i, mesh in enumerate(outputs):
            if mesh is None:
                outputs[i] = trimesh.Trimesh(vertices=[[0,0,0]], faces=[[0,0,0]])
                print(f"Replaced None mesh at index {i} with dummy mesh")
        
        # is_random=False: part k always gets RGB[k], so the colour IS a stable
        # part id across runs. The default re-rolls random colours every run,
        # which makes the one thing identifying a part unreproducible.
        merged_mesh_trimesh = get_colored_mesh_composition(outputs, is_random=False)

        # One mesh with disconnected components rather than a batch: parts are
        # separate bodies of a single Trimesh, and the per-part colour survives
        # concatenate. Ordering comes from dump() rather than `outputs` so the
        # ids match the geometry actually concatenated.
        parts_in_order = merged_mesh_trimesh.dump()
        merged_mesh = trimesh.util.concatenate(parts_in_order)
        # Exact membership alongside the colour. The palette holds 15 entries and
        # num_parts goes to 16, so at the top setting two parts share a colour --
        # part_id stays unique whatever the palette does.
        merged_mesh.metadata["part_id"] = np.concatenate(
            [np.full(len(part.faces), i, dtype=np.int32)
             for i, part in enumerate(parts_in_order)]
        ) if parts_in_order else np.zeros(0, dtype=np.int32)

        # The palette rides in metadata too, so `part_id` + `part_colors`
        # reconstructs the colouring exactly. Needed because serialization.py
        # carries TextureVisuals only -- ColorVisuals vertex colours are a
        # documented TODO there and are dropped crossing the worker boundary,
        # which every node output does. metadata is mesh-lived, so it is not
        # exposed to the id-reuse hazard that keeps the colour accessor out.
        merged_mesh.metadata["part_colors"] = np.array(
            [np.asarray(part.visual.vertex_colors[0][:3], dtype=np.uint8)
             for part in parts_in_order], dtype=np.uint8
        ) if parts_in_order else np.zeros((0, 3), dtype=np.uint8)

        # Put the colours on the wire the way the pack expects: `vc` inside
        # comfy3d_extras, which _wire_in feeds straight into Mesh.vc, and which
        # write_glb now emits as COLOR_0. Going through visual.vertex_colors
        # instead loses them -- serialization.py carries TextureVisuals only.
        # .astype() copies, so no torch tensor ever aliases a TrackedArray.
        vertex_colors = np.asarray(merged_mesh.visual.vertex_colors)
        merged_mesh.metadata[_MESH_EXTRAS_KEY] = {
            "vc": torch.from_numpy(
                vertex_colors[:, :3].astype(np.float32) / 255.0
            )
        }
        
        # Debug logging
        # Create ZIP with individual parts
        parts_output_dir = "output/partcrafter_parts"
        os.makedirs(parts_output_dir, exist_ok=True)
        
        zip_path = os.path.join(parts_output_dir, f"parts.zip")
        parts = []
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for idx, mesh in enumerate(outputs):
                if mesh is not None:
                    # Save part as GLB file
                    part_path = os.path.join(parts_output_dir, f"part_{idx:02d}.glb")
                    mesh.export(part_path)
                    parts.append(part_path)
                    
                    # Add to ZIP
                    zipf.write(part_path, f"part_{idx:02d}.glb")
                    
                    # Clean up temporary file
                    try:
                        os.remove(part_path)
                    except:
                        pass
        
        print(f"Created parts ZIP: {zip_path} with {len(parts)} parts")
        
        # Save merged mesh 
        scene_output_dir = "output/partcrafter_scenes"
        os.makedirs(scene_output_dir, exist_ok=True)
        scene_file_path = os.path.join(scene_output_dir, f"scene.glb")
        
        # For Preview_3DMesh, return relative path from ComfyUI output directory
        relative_scene_path = "partcrafter_scenes/scene.glb"
        
        # Export the merged mesh (same as app.py: merged.export(merged_path))
        merged_mesh_trimesh.export(scene_file_path)
        cstr(f"[PartCrafter_Generate] wrote {scene_file_path}").msg.print()

        # Processed image
        processed_image_tensor = pils_to_torch_imgs([pil_image])
        
        cstr(f"[PartCrafter_Generate] {len(parts_in_order)} parts -> one mesh, "
             f"{merged_mesh.body_count} disconnected bodies, "
             f"{len(merged_mesh.faces)} faces").msg.print()

        return (zip_path, relative_scene_path, processed_image_tensor,
                _wire_out(merged_mesh))

