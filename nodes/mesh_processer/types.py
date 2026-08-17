"""Small container types shared across the pack.

Kept apart from mesh.py deliberately: mesh.py pulls torch, cv2, trimesh and
kiui, and anything importing it pays for all four. PointCloud is three numpy
arrays -- consumers of it should not have to load a mesh stack, and gutting
mesh.py should not be able to break their imports.
"""

from typing import NamedTuple

import numpy as np


class PointCloud(NamedTuple):
    points: np.array
    colors: np.array
    normals: np.array
