"""
Mesh processing utilities.

This module provides functions for loading meshes, sampling points,
and handling mesh-related operations.
"""

import logging
import numpy as np
import trimesh
from typing import Optional

logger = logging.getLogger(__name__)

def load_mesh_and_sample(mesh_file: str, num_points: int = 2048, scale: float = 1.0) -> Optional[np.ndarray]:
    """Load a mesh file and sample points uniformly from its surface.

    Multi-geometry scenes are concatenated into a single mesh.

    Args:
        mesh_file: Path to the mesh file (.obj, .stl, .ply).
        num_points: Number of points to sample.
        scale: Scale factor to apply to the mesh.

    Returns:
        ``float32`` point cloud of shape ``(num_points, 3)``, or ``None`` if
        loading/sampling fails or the scene has no geometry.
    """
    try:
        mesh = trimesh.load(mesh_file)
        if isinstance(mesh, trimesh.Scene):
                if len(mesh.geometry) == 0:
                    logger.warning(f"Loaded mesh scene {mesh_file} has no geometry")
                    return None
                mesh = mesh.dump(concatenate=True)
        
        mesh.apply_scale(scale)
        
        # Sample points from surface
        pcd, _ = trimesh.sample.sample_surface(mesh, num_points)
        return np.array(pcd).astype(np.float32)
            
    except Exception as e:
        logger.error(f"Failed to load/sample mesh {mesh_file}: {e}")
        return None
