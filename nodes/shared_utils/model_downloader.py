"""Model weights: where they live, and how they get there.

Weights belong in ComfyUI's models/ tree, not inside custom_nodes/. Registering
the directory with folder_paths means it honours --models-directory and
extra_model_paths.yaml, shows up in the UI's model lists, survives a reinstall
of the pack, and can be shared with other packs instead of being downloaded a
second time into a private folder.

This mirrors what ComfyUI-TRELLIS2 does (nodes/nodes_loader.py:15-17 and
nodes/stages.py:165-230); the pieces that matter are the same four:

  1. models live under folder_paths.models_dir and are registered
  2. downloads report progress into the ComfyUI UI, not just the console
  3. every download is existence-checked first, so re-running is free
  4. loader nodes say "(Down)Load ..." because that is what they do

Everything here is import-safe without ComfyUI present -- `folder_paths` is
resolved lazily and falls back to a path relative to this file, so the module
can be imported by tooling outside a running server.
"""

import os

_SUBDIR = "3d-pack"
_registered = False


def _models_root():
    """ComfyUI's models/ dir, or a sensible fallback outside a server."""
    try:
        import folder_paths

        return folder_paths.models_dir
    except Exception:
        # nodes/shared_utils/ -> nodes/ -> pack -> custom_nodes -> ComfyUI
        here = os.path.dirname(os.path.abspath(__file__))
        return os.path.abspath(os.path.join(here, "..", "..", "..", "..", "models"))


def get_model_dir(*parts):
    """Absolute path under models/3d-pack/, created, and registered once.

    Registration is what puts the directory in front of ComfyUI's own model
    resolution, so a user who already has e.g. a TRELLIS checkout does not
    download it again.
    """
    global _registered

    # Callers that already hold an absolute directory (the legacy
    # resume_or_download_model_from_hf signature takes one) pass it straight
    # through rather than having it silently reinterpreted as a subdirectory.
    if parts and os.path.isabs(str(parts[0])):
        path = os.path.join(*[str(p) for p in parts])
        os.makedirs(path, exist_ok=True)
        return path

    root = os.path.join(_models_root(), _SUBDIR)
    os.makedirs(root, exist_ok=True)

    if not _registered:
        try:
            import folder_paths

            folder_paths.add_model_folder_path(_SUBDIR, root)
        except Exception:
            pass  # no server; the path still works, it just isn't advertised
        _registered = True

    path = os.path.join(root, *parts) if parts else root
    os.makedirs(path, exist_ok=True)
    return path


def _comfy_tqdm():
    """tqdm subclass that mirrors download progress into ComfyUI's progress bar.

    huggingface_hub takes a `tqdm_class`; without this the node just looks
    frozen for however long a multi-gigabyte download takes.
    """
    try:
        import comfy.utils
        import tqdm as _tqdm_mod
    except ImportError:
        return None

    holder = {"pbar": None, "total": 0, "done": 0}

    class _T(_tqdm_mod.tqdm):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            if self.total and self.total > 0 and holder["pbar"] is None:
                holder["total"] = self.total
                holder["done"] = 0
                holder["pbar"] = comfy.utils.ProgressBar(self.total)

        def update(self, n=1):
            ret = super().update(n)
            if n and holder["pbar"] and holder["total"] > 0:
                holder["done"] = min(holder["done"] + n, holder["total"])
                holder["pbar"].update_absolute(holder["done"], holder["total"])
            return ret

    return _T


def download_file(repo_id, filename, *parts, repo_type="model", force=False):
    """One file from a HF repo -> absolute local path.

    Skips the network entirely when the file is already present, which is what
    makes it safe to call on every node execution.
    """
    local_dir = get_model_dir(*(parts or (repo_id.replace("/", os.sep),)))
    dest = os.path.join(local_dir, filename)
    if os.path.isfile(dest) and not force:
        return dest

    from huggingface_hub import hf_hub_download

    print(f"[Comfy3D] downloading {filename} from {repo_id} ...", flush=True)
    hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=local_dir,
        repo_type=repo_type,
        tqdm_class=_comfy_tqdm(),
    )
    return dest


def download_repo(repo_id, *parts, repo_type="model", allow_patterns=None,
                  ignore_patterns=None, force=False, requires=()):
    """A whole HF repo (or a subset) -> absolute local directory.

    `requires` is the list of repo-relative files whose presence means "already
    fetched". Without it every call re-walks the repo listing over the network
    even when nothing needs downloading -- which is what the hand-rolled
    _ensure_weights helpers each reimplemented, slightly differently.
    """
    local_dir = get_model_dir(*(parts or (repo_id.replace("/", os.sep),)))

    if requires and not force:
        missing = [f for f in requires
                   if not os.path.exists(os.path.join(local_dir, f))]
        if not missing:
            return local_dir
        print(f"[Comfy3D] {repo_id}: {len(missing)} of {len(requires)} "
              f"required file(s) missing", flush=True)

    from huggingface_hub import snapshot_download

    print(f"[Comfy3D] downloading repo {repo_id} ...", flush=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        repo_type=repo_type,
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
        force_download=force,
        tqdm_class=_comfy_tqdm(),
    )
    return local_dir


def download_files(repo_id, filenames, *parts, repo_type="model", force=False):
    """Named files from a HF repo -> absolute local directory.

    Cheaper than download_repo when the needed files are known: no repo listing
    walk, and only the missing ones are fetched.
    """
    local_dir = get_model_dir(*(parts or (repo_id.replace("/", os.sep),)))
    for name in filenames:
        dest = os.path.join(local_dir, name)
        if os.path.isfile(dest) and not force:
            continue
        from huggingface_hub import hf_hub_download

        print(f"[Comfy3D] downloading {name} from {repo_id} ...", flush=True)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        hf_hub_download(
            repo_id=repo_id,
            filename=name,
            local_dir=local_dir,
            repo_type=repo_type,
            tqdm_class=_comfy_tqdm(),
        )
    return local_dir


def download_url(url, filename, *parts, force=False):
    """A plain HTTP asset (GitHub release, etc.) -> absolute local path."""
    local_dir = get_model_dir(*parts)
    dest = os.path.join(local_dir, filename)
    if os.path.isfile(dest) and not force:
        return dest

    import urllib.request

    print(f"[Comfy3D] downloading {filename} from {url} ...", flush=True)
    urllib.request.urlretrieve(url, dest)
    return dest
