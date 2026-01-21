"""
Mesh processing utilities.

This module provides functions for loading meshes, sampling points,
and handling mesh-related operations.
"""

import logging
import numpy as np
import trimesh
import open3d as o3d
from typing import Optional, Tuple, Union

logger = logging.getLogger(__name__)

def load_mesh_and_sample(mesh_file: str, num_points: int = 2048, scale: float = 1.0) -> Optional[np.ndarray]:
    """
    Load a mesh file and sample points from its surface.
    
    Args:
        mesh_file: Path to the mesh file (.obj, .stl, .ply)
        num_points: Number of points to sample
        scale: Scale factor to apply to the mesh
        
    Returns:
        Sampled point cloud (N, 3) or None if loading fails
    """
    try:
        if mesh_file.endswith(".ply"):
            pcd_o3d = o3d.io.read_point_cloud(mesh_file)
            pcd = np.array(pcd_o3d.points).astype(np.float32)
            # Simple downsampling/upsampling if needed
            if len(pcd) > num_points:
                indices = np.random.choice(len(pcd), num_points, replace=False)
                pcd = pcd[indices]
            # If less points, we might want to return as is or handle it. 
            # For now, just return what we have if it's not empty
            return pcd
        else:
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
