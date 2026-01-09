"""
Point cloud processing utilities.

This module provides common point cloud operations used across perception
and grasp prediction modules.
"""

import numpy as np
from typing import Tuple, Optional
from scipy.spatial.transform import Rotation as R


def parse_camera_intrinsics(intrinsics: np.ndarray) -> Tuple[float, float, float, float]:
    """Parse camera intrinsics into (fx, fy, cx, cy).

    Supports:
    - 1D array: [fx, fy, cx, cy] (robosuite default)
    - 2D matrix: standard pinhole K (3x3, 4x4, etc)
    """
    if intrinsics is None:
        raise ValueError("Camera intrinsics not available")

    I = np.asarray(intrinsics)

    if I.ndim == 1:
        if I.shape[0] != 4:
            raise ValueError(
                f"1D intrinsics array has invalid length {I.shape[0]}, expected 4 [fx, fy, cx, cy]"
            )
        fx, fy, cx, cy = float(I[0]), float(I[1]), float(I[2]), float(I[3])
    elif I.ndim == 2:
        if I.shape[0] < 2 or I.shape[1] < 3:
            raise ValueError(f"Invalid intrinsics matrix shape {I.shape}, expected at least (2, 3)")
        fx, fy, cx, cy = float(I[0, 0]), float(I[1, 1]), float(I[0, 2]), float(I[1, 2])
    else:
        raise ValueError(f"Invalid intrinsics shape {I.shape}, expected 1D or 2D array")

    return fx, fy, cx, cy


def ensure_depth_2d(depth_image: np.ndarray, *, height: Optional[int] = None, width: Optional[int] = None) -> np.ndarray:
    """Ensure depth image is a 2D array.

    - Accepts flat buffers if height/width are provided.
    - Squeezes trailing singleton channel (H, W, 1) -> (H, W).
    """
    if depth_image is None:
        raise ValueError("Depth image is None")

    depth = np.asarray(depth_image)

    if depth.ndim == 1:
        if height is None or width is None:
            raise ValueError(
                f"Flat depth buffer provided with shape {depth.shape} but height/width not specified"
            )
        expected = int(height) * int(width)
        if depth.size != expected:
            raise ValueError(f"Flat depth buffer size {depth.size} != expected {expected}")
        depth = depth.reshape(int(height), int(width))

    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]

    if depth.ndim != 2:
        raise ValueError(f"Depth has invalid shape {depth.shape}, expected 2D")

    return depth


def downsample_point_cloud(pcd: np.ndarray, target_points: int = 2048, 
                           method: str = 'random') -> np.ndarray:
    """
    Downsample point cloud to target number of points.
    
    Args:
        pcd: Input point cloud (N, 3)
        target_points: Target number of points
        method: Downsampling method ('random' or 'uniform')
        
    Returns:
        Downsampled point cloud (target_points, 3)
    """
    if len(pcd) <= target_points:
        return pcd
    
    if method == 'random':
        indices = np.random.choice(len(pcd), target_points, replace=False)
        return pcd[indices]
    elif method == 'uniform':
        # Uniform stride sampling
        stride = len(pcd) // target_points
        indices = np.arange(0, len(pcd), stride)[:target_points]
        return pcd[indices]
    else:
        raise ValueError(f"Unknown downsampling method: {method}")


def estimate_pose_from_point_cloud(pcd: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate object pose from point cloud using PCA for orientation.
    
    Uses Principal Component Analysis to determine the principal axes
    of the point cloud, which gives an estimate of the object's orientation.
    
    Args:
        pcd: Point cloud (N, 3) in world frame
        
    Returns:
        Tuple of (position, quaternion) where:
        - position: centroid [x, y, z]
        - quaternion: [x, y, z, w] (scipy convention)
    """
    if len(pcd) == 0:
        return np.array([0, 0, 0]), np.array([0, 0, 0, 1])

    # Centroid
    centroid = np.mean(pcd, axis=0)

    # Center the point cloud
    pcd_centered = pcd - centroid

    # Compute covariance matrix
    cov_matrix = np.cov(pcd_centered.T)

    # Eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Sort eigenvectors by eigenvalues (largest first)
    sort_indices = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, sort_indices]

    # Create rotation matrix from eigenvectors
    # X axis: direction of maximum variance
    # Y axis: direction of second maximum variance  
    # Z axis: up direction (minimum variance, assuming objects rest on surfaces)
    rot_matrix = eigenvectors.copy()
    
    # Ensure right-handed coordinate system
    if np.linalg.det(rot_matrix) < 0:
        rot_matrix[:, 2] = -rot_matrix[:, 2]

    # Convert to quaternion
    quat = R.from_matrix(rot_matrix).as_quat()

    return centroid, quat


def transform_point_cloud(pcd: np.ndarray, transform_matrix: np.ndarray) -> np.ndarray:
    """
    Transform point cloud using a 4x4 transformation matrix.
    
    Args:
        pcd: Point cloud (N, 3)
        transform_matrix: 4x4 homogeneous transformation matrix
        
    Returns:
        Transformed point cloud (N, 3)
    """
    if len(pcd) == 0:
        return pcd
    
    # Convert to homogeneous coordinates
    ones = np.ones((len(pcd), 1))
    pcd_homogeneous = np.hstack([pcd, ones])
    
    # Apply transform
    pcd_transformed = (transform_matrix @ pcd_homogeneous.T).T[:, :3]
    
    return pcd_transformed


def compute_point_cloud_bounds(pcd: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute axis-aligned bounding box of point cloud.
    
    Args:
        pcd: Point cloud (N, 3)
        
    Returns:
        Tuple of (min_bounds, max_bounds) each as [x, y, z]
    """
    if len(pcd) == 0:
        return np.zeros(3), np.zeros(3)
    
    min_bounds = np.min(pcd, axis=0)
    max_bounds = np.max(pcd, axis=0)
    
    return min_bounds, max_bounds


def filter_point_cloud_by_range(pcd: np.ndarray, 
                                 min_range: float = 0.0,
                                 max_range: float = 10.0) -> np.ndarray:
    """
    Filter point cloud by distance from origin.
    
    Args:
        pcd: Point cloud (N, 3)
        min_range: Minimum distance to keep
        max_range: Maximum distance to keep
        
    Returns:
        Filtered point cloud
    """
    if len(pcd) == 0:
        return pcd
    
    distances = np.linalg.norm(pcd, axis=1)
    mask = (distances >= min_range) & (distances <= max_range)
    
    return pcd[mask]


def validate_point_cloud(pcd: np.ndarray, 
                         max_range: float = 10.0,
                         min_points: int = 1) -> bool:
    """
    Validate point cloud data.
    
    Args:
        pcd: Point cloud to validate
        max_range: Maximum valid coordinate range
        min_points: Minimum number of points required
        
    Returns:
        True if valid, False otherwise
    """
    if not isinstance(pcd, np.ndarray):
        return False
    
    if pcd.size == 0 or len(pcd) < min_points:
        return False
    
    if pcd.ndim != 2 or pcd.shape[1] != 3:
        return False
    
    # Check for reasonable coordinate ranges
    if np.any(np.abs(pcd) > max_range):
        return False
    
    # Check for NaN or Inf values
    if np.any(~np.isfinite(pcd)):
        return False
    
    return True


def deproject_depth_to_point_cloud(depth: np.ndarray,
                                    mask: Optional[np.ndarray],
                                    fx: float, fy: float,
                                    cx: float, cy: float,
                                    max_depth: float = 10.0,
                                    *,
                                    mujoco_convention: bool = True) -> np.ndarray:
    """
    Deproject depth image to 3D point cloud using camera intrinsics.
    
    Args:
        depth: 2D depth image (H, W)
        mask: Optional binary mask (H, W) to select pixels
        fx, fy: Focal lengths
        cx, cy: Principal point
        max_depth: Maximum valid depth value
        
    Returns:
        Point cloud (N, 3) in camera frame
    """
    if mask is not None:
        # Get pixel coordinates where mask is positive
        y_coords, x_coords = np.where(mask > 0)
        if len(y_coords) == 0:
            return np.array([]).reshape(0, 3)
        depth_values = depth[y_coords, x_coords]
    else:
        # Use all pixels
        H, W = depth.shape
        y_coords, x_coords = np.mgrid[0:H, 0:W]
        y_coords = y_coords.flatten()
        x_coords = x_coords.flatten()
        depth_values = depth.flatten()
    
    # Filter valid depth values
    valid_mask = (depth_values > 0) & (depth_values < max_depth) & np.isfinite(depth_values)
    x_coords = x_coords[valid_mask]
    y_coords = y_coords[valid_mask]
    depth_values = depth_values[valid_mask]
    
    if len(depth_values) == 0:
        return np.array([]).reshape(0, 3)
    
    # Deproject to 3D
    # Start from standard pinhole where Z is positive “forward”.
    Z = depth_values
    x = (x_coords - cx) * Z / fx
    y = (y_coords - cy) * Z / fy

    if mujoco_convention:
        # MuJoCo camera frames (and OpenGL-style rendering) typically use:
        # - image v increases downward, but camera +Y is up -> flip y
        # - camera forward direction is -Z -> negate z
        y = -y
        z = -Z
    else:
        z = Z

    return np.column_stack([x, y, z]).astype(np.float32)
