import numpy as np
import pymeshlab
import torch
import trimesh
import xatlas
from pymeshlab import PercentageValue


### Mesh Utils ###
##### read mesh
def read_mesh_from_path(mesh_path):
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(mesh_path)
    return ms


def mesh_to_meshlab(vertices, faces):
    mesh = pymeshlab.Mesh(vertex_matrix=vertices, face_matrix=faces)
    ms = pymeshlab.MeshSet()
    ms.add_mesh(mesh)
    return ms


def meshlab_to_mesh(ms):
    m = ms.current_mesh()
    return m.vertex_matrix(), m.face_matrix(), m.vertex_normal_matrix()


##### decimation
def decimate_quadric_edge_collapse_with_texture(
    ms, targetfacenum=None, preservenormal=True, verbose=False
):
    # targetfacenum: int, Target number of faces.
    # preservenormal: bool, Preserve the normals of the original mesh.
    if verbose:
        print("Starting decimation ...  ")
    m = ms.current_mesh()
    if targetfacenum is None:
        targetfacenum = int(m.face_number() * 0.5)
    if verbose:
        print("... Initial face number is %d ... " % m.face_number())
    ms.meshing_decimation_quadric_edge_collapse_with_texture(
        targetfacenum=targetfacenum, preservenormal=preservenormal
    )
    if verbose:
        print("... Decimated face number is %d ... " % m.face_number())
        print("Decimation done!\n ")


def decimate_quadric_edge_collapse(
    ms, targetfacenum=None, preservenormal=True, verbose=False
):
    # targetfacenum: int, Target number of faces.
    # preservenormal: bool, Preserve the normals of the original mesh.
    if verbose:
        print("Starting decimation ...  ")
    m = ms.current_mesh()
    if targetfacenum is None:
        targetfacenum = int(m.face_number() * 0.5)
    if verbose:
        print("... Initial face number is %d ... " % m.face_number())
    ms.meshing_decimation_quadric_edge_collapse(
        targetfacenum=targetfacenum, preservenormal=preservenormal
    )
    if verbose:
        print("... Decimated face number is %d ... " % m.face_number())
        print("Decimation done!\n ")


##### vertex merge
def merge_close_vertices(ms, threshold=0.0001, verbose=False):
    # threshold: float, Merge together all the vertices that are nearer than the specified threshold.
    if verbose:
        print("Starting merge vertices ...  ")
    m = ms.current_mesh()
    if verbose:
        print("... Initial vertex number is %d ... " % m.vertex_number())
    ms.meshing_merge_close_vertices(threshold=PercentageValue(threshold * 100))
    if verbose:
        print("... Merged vertex number is %d ... " % m.vertex_number())
        print("Merge vertices done!\n ")


##### Island Removal
def remove_isolated_pieces(ms, mincomponentsize=25, diameter=None, verbose=False):
    # mincomponentsize: Delete isolated connected components composed by a limited number of triangles
    # diameter: Delete isolated connected components whose diameter is smaller than the specified constant
    if verbose:
        print("Starting remove isolated pieces ...  ")
    m = ms.current_mesh()
    if verbose:
        print("... Initial face number is %d ... " % m.face_number())
    if diameter is None:
        ms.meshing_remove_connected_component_by_face_number(
            mincomponentsize=mincomponentsize, removeunref=True
        )
    else:
        ms.meshing_remove_connected_component_by_diameter(
            mincomponentdiag=PercentageValue(diameter), removeunref=True
        )
    if verbose:
        print("... Isolated removed face number is %d ... " % m.face_number())
        print("Remove isolated pieces done!\n ")


##### hole filling
def fix_hole(ms, maxholesize=30, verbose=False):
    # maxholesize: int, Maximum size of the hole to be filled.
    if verbose:
        print("Starting fix holes ...  ")
    m = ms.current_mesh()
    if verbose:
        print("... Initial face number is %d ... " % m.face_number())
    ms.meshing_close_holes(maxholesize=maxholesize)
    if verbose:
        print("... Fixed hole face number is %d ... " % m.face_number())
        print("Fix holes done!\n ")


##### repair non manifold edges
def repair_non_manifold(ms, verbose=False):
    if verbose:
        print("Starting repair non manifold edges ...  ")
    m = ms.current_mesh()
    if verbose:
        print("... Initial face number is %d ... " % m.face_number())
    ms.meshing_repair_non_manifold_edges()
    ms.meshing_repair_non_manifold_vertices(vertdispratio=0.1)
    ms.meshing_remove_duplicate_faces()
    if verbose:
        print("... Fixed non manifold edges face number is %d ... " % m.face_number())
        print("Repair non manifold edges done!\n ")


##### laplacian_smooth
def laplacian_smooth(ms, stepsmoothnum=3, verbose=False):
    # stepsmoothnum: int, Number of smoothing steps to be performed
    if verbose:
        print("Starting laplacian smooth ...  ")
    m = ms.current_mesh()
    ms.apply_coord_laplacian_smoothing(stepsmoothnum=stepsmoothnum)
    if verbose:
        print("Laplacian smooth done!\n ")


##### taubin_smooth
def taubin_smooth(ms, stepsmoothnum=3, verbose=False):
    if verbose:
        print("Starting Taubin smooth ...  ")
    m = ms.current_mesh()
    ms.apply_coord_taubin_smoothing(stepsmoothnum=stepsmoothnum)
    if verbose:
        print("Taubin smooth done!\n ")


##### compute_normal
def compute_normal(ms, weightmode="Simple Average", verbose=False):
    if verbose:
        print("Starting compute_normal_per_vertex ...  ")
    m = ms.current_mesh()
    ms.compute_normal_per_vertex(weightmode=weightmode)
    if verbose:
        print("compute_normal_per_vertex done!\n ")


### Pre-process Mesh ###
def process_mesh(
    vertices,
    faces,
    threshold=0.0001,
    mincomponentRatio=0.02,
    targetfacenum=50000,
    maxholesize=30,
    stepsmoothnum=10,
    verbose=False,
):
    ms = mesh_to_meshlab(vertices, faces)

    ### Vertex Merge
    merge_close_vertices(ms, threshold=threshold, verbose=verbose)

    ### Island Removal
    faces = ms.current_mesh().face_matrix()
    remove_isolated_pieces(
        ms, mincomponentsize=int(len(faces) * mincomponentRatio), verbose=verbose
    )

    ### Hole Filling
    repair_non_manifold(ms)  # repair before fix hole
    fix_hole(ms, maxholesize=maxholesize, verbose=verbose)

    ### Taubin Smoothing
    taubin_smooth(ms, stepsmoothnum=stepsmoothnum, verbose=verbose)

    ### Mesh Simplification/Decimation
    if ms.current_mesh().face_number() > targetfacenum:
        decimate_quadric_edge_collapse(
            ms, targetfacenum=targetfacenum, verbose=verbose
        )

    taubin_smooth(ms, stepsmoothnum=stepsmoothnum, verbose=verbose)
    repair_non_manifold(ms, verbose=verbose)
    compute_normal(ms, verbose=verbose)
    return meshlab_to_mesh(ms)


### UV Un-Warp ###
def uv_parameterize_uvatlas(
    vertices,
    faces,
    size=1024,
    gutter=2.5,
    max_stretch=0.1666666716337204,
    parallel_partitions=16,
    nthreads=0,
):
    # xatlas rather than open3d's UVAtlas: open3d ships no cp313 wheel and no
    # sdist, so it pins the whole env to <=3.12. xatlas is already a dependency
    # and is what the rest of the pack unwraps with. Different packer, so the
    # island layout differs from UVAtlas -- the contract kept here is the one
    # the caller relies on: per-face UVs, (#F, 3, 2), normalized to [0, 1].
    # parallel_partitions/nthreads have no xatlas equivalent and are ignored.
    atlas = xatlas.Atlas()
    atlas.add_mesh(vertices.astype(np.float32), faces.astype(np.uint32))

    chart_options = xatlas.ChartOptions()
    chart_options.max_cost = max_stretch

    pack_options = xatlas.PackOptions()
    pack_options.resolution = int(size)
    pack_options.padding = int(round(gutter))  # open3d's gutter, in texels

    atlas.generate(chart_options, pack_options)
    _vmapping, indices, uvs = atlas[0]

    if indices.shape[0] != faces.shape[0]:
        # The caller derives t_tex_idx as arange(#F * 3), so a face count that
        # shifts under xatlas would silently misalign every UV.
        raise RuntimeError(
            f"xatlas returned {indices.shape[0]} faces for an input of "
            f"{faces.shape[0]}; UV indices would not line up with the mesh."
        )

    return uvs[indices].astype(np.float32)  # (#F, 3, 2)


### Pack All ###
def process_raw(mesh_path, save_path, preprocess=True, device="cpu"):
    scene = trimesh.load(mesh_path, force="mesh", process=False)
    if isinstance(scene, trimesh.Trimesh):
        mesh = scene
    elif isinstance(scene, trimesh.scene.Scene):
        mesh = trimesh.Trimesh()
        for obj in scene.geometry.values():
            mesh = trimesh.util.concatenate([mesh, obj])
    else:
        raise ValueError(f"Unknown mesh type at {mesh_path}.")

    vertices = mesh.vertices
    faces = mesh.faces

    mesh_post_process_options = {
        "mincomponentRatio": 0.02,
        "targetfacenum": 50000,
        "maxholesize": 100,
        "stepsmoothnum": 10,
        "verbose": False,
    }

    if preprocess:
        v_pos, t_pos_idx, normals = process_mesh(
            vertices=vertices,
            faces=faces,
            **mesh_post_process_options,
        )
    else:
        v_pos, t_pos_idx, normals = vertices, faces, mesh.vertex_normals

    v_tex_np = (
        uv_parameterize_uvatlas(v_pos, t_pos_idx).reshape(-1, 2).astype(np.float32)
    )

    v_pos = torch.from_numpy(v_pos).to(device=device, dtype=torch.float32)
    t_pos_idx = torch.from_numpy(t_pos_idx).to(device=device, dtype=torch.long)
    v_tex = torch.from_numpy(v_tex_np).to(device=device, dtype=torch.float32)
    normals = torch.from_numpy(normals).to(device=device, dtype=torch.float32)
    assert v_tex.shape[0] == t_pos_idx.shape[0] * 3
    t_tex_idx = torch.arange(
        t_pos_idx.shape[0] * 3,
        device=device,
        dtype=torch.long,
    ).reshape(-1, 3)
    # uv, index = torch.unique(v_tex, dim=0, return_inverse=True) # 这样实现是2毫秒
    # super efficient de-duplication
    v_tex_u_uint32 = v_tex_np[..., 0].view(np.uint32)
    v_tex_v_uint32 = v_tex_np[..., 1].view(np.uint32)
    v_hashed = (v_tex_u_uint32.astype(np.uint64) << 32) | v_tex_v_uint32
    v_hashed = torch.from_numpy(v_hashed.view(np.int64)).to(v_pos.device)

    t_pos_idx_f3 = torch.arange(
        t_pos_idx.shape[0] * 3, device=t_pos_idx.device, dtype=torch.long
    ).reshape(-1, 3)
    v_pos_f3 = v_pos[t_pos_idx].reshape(-1, 3)
    normals_f3 = normals[t_pos_idx].reshape(-1, 3)

    v_hashed_dedup, inverse_indices = torch.unique(v_hashed, return_inverse=True)
    dedup_size, full_size = v_hashed_dedup.shape[0], inverse_indices.shape[0]
    indices = torch.scatter_reduce(
        torch.full(
            [dedup_size],
            fill_value=full_size,
            device=inverse_indices.device,
            dtype=torch.long,
        ),
        index=inverse_indices,
        src=torch.arange(full_size, device=inverse_indices.device, dtype=torch.int64),
        dim=0,
        reduce="amin",
    )
    v_tex = v_tex[indices]
    t_tex_idx = inverse_indices.reshape(-1, 3)

    v_pos = v_pos_f3[indices]
    normals = normals_f3[indices]

    normals = normals.to(dtype=torch.float32, device=device)

    # either flip uv or flip texture
    # here we flip uv
    uv_to_save = v_tex.clone()
    uv_to_save[:, 1] = 1.0 - uv_to_save[:, 1]

    visual = trimesh.visual.TextureVisuals(uv=uv_to_save.cpu().numpy())
    tmesh = trimesh.Trimesh(
        vertices=v_pos.cpu().numpy(),
        faces=t_tex_idx.cpu().numpy(),
        vertex_normals=normals.cpu().numpy(),
        visual=visual,
        process=False,
    )
    tmesh.export(save_path)
