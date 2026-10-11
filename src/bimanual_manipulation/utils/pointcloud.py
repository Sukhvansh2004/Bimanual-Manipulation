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

    Args:
        intrinsics: Intrinsics array as described above.

    Returns:
        ``(fx, fy, cx, cy)`` as floats.

    Raises:
        ValueError: If ``intrinsics`` is ``None`` or has an unsupported shape.
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

    Args:
        depth_image: Depth array of shape ``(H*W,)``, ``(H, W)`` or ``(H, W, 1)``.
        height: Image height, required for flat buffers.
        width: Image width, required for flat buffers.

    Returns:
        Depth array of shape ``(H, W)``.

    Raises:
        ValueError: If ``depth_image`` is ``None``, a flat buffer lacks or
            mismatches ``height``/``width``, or the result is not 2D.
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
    """Downsample a point cloud to a target number of points.

    Args:
        pcd: Input point cloud ``(N, 3)``.
        target_points: Target number of points.
        method: ``'random'`` (without replacement) or ``'uniform'``
            (fixed index stride).

    Returns:
        Point cloud of shape ``(target_points, 3)``, or ``pcd`` unchanged if
        ``N <= target_points``.

    Raises:
        ValueError: If ``method`` is unknown.
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
    """Estimate object pose from a point cloud using PCA for orientation.

    The rotation's columns are the principal axes sorted by decreasing
    variance (X = largest, Z = smallest), with Z flipped if needed to make
    the frame right-handed. Axis signs are otherwise arbitrary.

    Args:
        pcd: Point cloud ``(N, 3)`` in world frame.

    Returns:
        ``(position, quaternion)``: centroid ``[x, y, z]`` and quaternion
        ``[x, y, z, w]`` (scipy convention). Identity at the origin if
        ``pcd`` is empty.
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
    """Transform a point cloud using a 4x4 transformation matrix.

    Args:
        pcd: Point cloud ``(N, 3)``.
        transform_matrix: Homogeneous transformation matrix ``(4, 4)``.

    Returns:
        Transformed point cloud ``(N, 3)`` (``pcd`` itself if empty).
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
    """Compute the axis-aligned bounding box of a point cloud.

    Args:
        pcd: Point cloud ``(N, 3)``.

    Returns:
        ``(min_bounds, max_bounds)``, each ``(3,)``; zeros if ``pcd`` is empty.
    """
    if len(pcd) == 0:
        return np.zeros(3), np.zeros(3)
    
    min_bounds = np.min(pcd, axis=0)
    max_bounds = np.max(pcd, axis=0)
    
    return min_bounds, max_bounds


def filter_point_cloud_by_range(pcd: np.ndarray, 
                                 min_range: float = 0.0,
                                 max_range: float = 10.0) -> np.ndarray:
    """Filter a point cloud by Euclidean distance from the origin.

    Args:
        pcd: Point cloud ``(N, 3)``.
        min_range: Minimum distance to keep (inclusive).
        max_range: Maximum distance to keep (inclusive).

    Returns:
        Filtered point cloud ``(M, 3)``.
    """
    if len(pcd) == 0:
        return pcd
    
    distances = np.linalg.norm(pcd, axis=1)
    mask = (distances >= min_range) & (distances <= max_range)
    
    return pcd[mask]


def validate_point_cloud(pcd: np.ndarray, 
                         max_range: float = 10.0,
                         min_points: int = 1) -> bool:
    """Validate point cloud data.

    Args:
        pcd: Point cloud to validate.
        max_range: Maximum allowed absolute value of any coordinate.
        min_points: Minimum number of points required.

    Returns:
        True if ``pcd`` is an ``(N, 3)`` ndarray with ``N >= min_points``,
        all coordinates finite and within ``max_range``; False otherwise.
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
                                    mujoco_convention: bool = True,
                                    image_flipped: bool = False) -> np.ndarray:
    """Deproject a depth image to a 3D point cloud using camera intrinsics.

    Pixels with depth ``<= 0``, ``>= max_depth`` or non-finite are dropped.

    Args:
        depth: 2D metric depth image ``(H, W)``.
        mask: Optional mask ``(H, W)``; pixels with ``mask > 0`` are used.
        fx: Focal length in x (pixels).
        fy: Focal length in y (pixels).
        cx: Principal point x (pixels).
        cy: Principal point y (pixels).
        max_depth: Exclusive upper bound on valid depth.
        mujoco_convention: If True, negate Y and Z for MuJoCo camera
            convention (+Y up, looking along -Z); otherwise the standard
            optical convention (+Z forward).
        image_flipped: If True, the depth image has been vertically flipped
            from OpenGL origin (bottom-left) to standard image origin (top-left).
            The pinhole formula will use corrected v coordinates to match
            robosuite's camera extrinsic matrix.

    Returns:
        ``float32`` point cloud ``(N, 3)`` in the camera frame (``(0, 3)`` if
        no valid pixels).
    """
    H, W = depth.shape
    
    if mask is not None:
        # Get pixel coordinates where mask is positive
        y_coords, x_coords = np.where(mask > 0)
        if len(y_coords) == 0:
            return np.array([]).reshape(0, 3)
        depth_values = depth[y_coords, x_coords]
    else:
        # Use all pixels
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
    
    # Convert v coordinates back to original OpenGL space if image was flipped.
    # This is necessary because robosuite's extrinsic matrix is computed for
    # the original OpenGL image coordinate system (origin at bottom-left).
    if image_flipped:
        # v_original = H - 1 - v_flipped
        y_coords_for_projection = (H - 1) - y_coords
    else:
        y_coords_for_projection = y_coords
    
    # Deproject to 3D
    # Standard pinhole where Z is positive "forward".
    Z = depth_values
    x = (x_coords - cx) * Z / fx
    y = (y_coords_for_projection - cy) * Z / fy

    if mujoco_convention:
        # MuJoCo camera frames (and OpenGL-style rendering) typically use:
        # - image v increases downward, but camera +Y is up -> flip y
        # - camera forward direction is -Z -> negate z
        y = -y
        z = -Z
    else:
        z = Z

    return np.column_stack([x, y, z]).astype(np.float32)

