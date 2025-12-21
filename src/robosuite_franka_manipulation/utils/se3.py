"""
SE(3) Lie Algebra Utilities for Trajectory Planning.

This module provides functions for working with rigid body transformations
using the SE(3) Lie group and se(3) Lie algebra. These are essential for:
- Proper coupled rotation-translation interpolation
- Geodesic motion on the SE(3) manifold
- Trajectory planning with smooth, natural robot motions

Mathematical Background:
- SE(3): Special Euclidean group in 3D (rigid body transformations)
- se(3): Lie algebra of SE(3) (infinitesimal transformations/twists)
- exp: Maps se(3) -> SE(3) (Lie algebra to Lie group)
- log: Maps SE(3) -> se(3) (Lie group to Lie algebra)

Twist Convention:
- 6D vector [omega, v] where omega is angular velocity (3D) and v is linear velocity (3D)
- omega represents the axis-angle of rotation
- v represents translation (in body frame for body-fixed, spatial frame for spatial)

References:
- Murray, Li, Sastry: "A Mathematical Introduction to Robotic Manipulation"
- Lynch, Park: "Modern Robotics: Mechanics, Planning, and Control"
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Tuple


def skew(v: np.ndarray) -> np.ndarray:
    """
    Create skew-symmetric matrix from 3D vector.
    
    The skew-symmetric matrix [v]× satisfies: [v]× @ u = v × u (cross product)
    
    For vector v = [v1, v2, v3], returns:
        [[ 0, -v3,  v2],
         [ v3,  0, -v1],
         [-v2,  v1,  0]]
    
    Args:
        v: 3D vector
        
    Returns:
        3x3 skew-symmetric matrix
    """
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])


def unskew(S: np.ndarray) -> np.ndarray:
    """
    Extract 3D vector from skew-symmetric matrix.
    
    Inverse of skew(): unskew(skew(v)) = v
    
    Args:
        S: 3x3 skew-symmetric matrix
        
    Returns:
        3D vector
    """
    return np.array([S[2, 1], S[0, 2], S[1, 0]])


def se3_to_matrix(twist: np.ndarray) -> np.ndarray:
    """
    Convert 6D twist vector to 4x4 se(3) matrix representation.
    
    Twist format: [omega (3), v (3)] where omega is angular velocity
    and v is linear velocity.
    
    The se(3) matrix has the form:
        [[skew(omega), v],
         [0, 0, 0,     0]]
    
    Args:
        twist: 6D twist vector [omega, v]
        
    Returns:
        4x4 se(3) matrix
    """
    omega = twist[:3]
    v = twist[3:]
    mat = np.zeros((4, 4))
    mat[:3, :3] = skew(omega)
    mat[:3, 3] = v
    return mat


def matrix_to_se3(mat: np.ndarray) -> np.ndarray:
    """
    Convert 4x4 se(3) matrix to 6D twist vector.
    
    Inverse of se3_to_matrix(): matrix_to_se3(se3_to_matrix(twist)) = twist
    
    Args:
        mat: 4x4 se(3) matrix
        
    Returns:
        6D twist vector [omega, v]
    """
    omega = unskew(mat[:3, :3])
    v = mat[:3, 3]
    return np.concatenate([omega, v])


def SE3_log(T: np.ndarray) -> np.ndarray:
    """
    Compute the matrix logarithm of an SE(3) transformation matrix.
    
    Maps SE(3) -> se(3) (Lie group to Lie algebra).
    Uses closed-form solution for numerical stability.
    
    This is the inverse of SE3_exp(): SE3_exp(SE3_log(T)) ≈ T
    
    The logarithm gives the "twist" that, when integrated for unit time,
    produces the transformation T.
    
    Args:
        T: 4x4 homogeneous transformation matrix in SE(3)
        
    Returns:
        6D twist vector [omega, v] in se(3)
        
    Note:
        Handles three cases:
        - Near identity (theta ≈ 0): First-order approximation
        - Near 180° rotation (theta ≈ π): Special axis extraction
        - General case: Closed-form Rodriguez logarithm
    """
    R_mat = T[:3, :3]
    p = T[:3, 3]
    
    # Compute rotation angle
    cos_theta = (np.trace(R_mat) - 1) / 2
    cos_theta = np.clip(cos_theta, -1, 1)
    theta = np.arccos(cos_theta)
    
    if theta < 1e-8:
        # Near identity: use first-order approximation
        omega = np.zeros(3)
        v = p
    elif np.abs(theta - np.pi) < 1e-8:
        # Near 180 degrees: special handling
        # Find axis from diagonal elements
        diag = np.diag(R_mat)
        idx = np.argmax(diag)
        
        # Reconstruct axis more carefully
        r_vec = np.sqrt((diag + 1) / 2)
        r_vec[idx] = np.sqrt((diag[idx] + 1) / 2)
        
        # Determine signs from off-diagonal elements
        if idx == 0:
            r_vec[1] = np.sign(R_mat[0, 1]) * r_vec[1] if R_mat[0, 1] != 0 else r_vec[1]
            r_vec[2] = np.sign(R_mat[0, 2]) * r_vec[2] if R_mat[0, 2] != 0 else r_vec[2]
        elif idx == 1:
            r_vec[0] = np.sign(R_mat[0, 1]) * r_vec[0] if R_mat[0, 1] != 0 else r_vec[0]
            r_vec[2] = np.sign(R_mat[1, 2]) * r_vec[2] if R_mat[1, 2] != 0 else r_vec[2]
        else:
            r_vec[0] = np.sign(R_mat[0, 2]) * r_vec[0] if R_mat[0, 2] != 0 else r_vec[0]
            r_vec[1] = np.sign(R_mat[1, 2]) * r_vec[1] if R_mat[1, 2] != 0 else r_vec[1]
        
        omega = theta * r_vec / (np.linalg.norm(r_vec) + 1e-10)
        
        # V_inv for theta = pi
        omega_hat = skew(omega)
        omega_norm = np.linalg.norm(omega)
        if omega_norm > 1e-10:
            V_inv = np.eye(3) - 0.5 * omega_hat + (1 / omega_norm**2) * (1 - omega_norm * np.sin(omega_norm) / (2 * (1 - np.cos(omega_norm)))) * omega_hat @ omega_hat
        else:
            V_inv = np.eye(3)
        v = V_inv @ p
    else:
        # General case
        omega_hat = (theta / (2 * np.sin(theta))) * (R_mat - R_mat.T)
        omega = unskew(omega_hat)
        
        # Compute V_inv analytically
        omega_norm = np.linalg.norm(omega)
        V_inv = np.eye(3) - 0.5 * omega_hat + (1 / theta**2) * (1 - theta * np.sin(theta) / (2 * (1 - np.cos(theta)))) * omega_hat @ omega_hat
        v = V_inv @ p
    
    return np.concatenate([omega, v])


def SE3_exp(twist: np.ndarray) -> np.ndarray:
    """
    Compute the matrix exponential of an se(3) twist vector.
    
    Maps se(3) -> SE(3) (Lie algebra to Lie group).
    Uses closed-form Rodriguez formula for efficiency and numerical stability.
    
    This is the inverse of SE3_log(): SE3_log(SE3_exp(twist)) ≈ twist
    
    The exponential "integrates" the twist for unit time to produce
    a rigid body transformation.
    
    Args:
        twist: 6D vector [omega, v] in se(3)
               omega: angular velocity (rotation axis * angle)
               v: linear velocity
        
    Returns:
        4x4 homogeneous transformation matrix in SE(3)
        
    Note:
        Uses Rodriguez formula for rotation and the V matrix for
        the coupled rotation-translation relationship.
    """
    omega = twist[:3]
    v = twist[3:]
    theta = np.linalg.norm(omega)
    
    T = np.eye(4)
    
    if theta < 1e-8:
        # Near zero rotation: use first-order approximation
        T[:3, :3] = np.eye(3) + skew(omega)
        T[:3, 3] = v
    else:
        # Rodriguez formula for rotation
        omega_hat = skew(omega)
        omega_hat_normalized = omega_hat / theta
        
        R_mat = np.eye(3) + np.sin(theta) * omega_hat_normalized + (1 - np.cos(theta)) * omega_hat_normalized @ omega_hat_normalized
        
        # V matrix for translation (couples rotation and translation)
        V = np.eye(3) + ((1 - np.cos(theta)) / theta**2) * omega_hat + ((theta - np.sin(theta)) / theta**3) * omega_hat @ omega_hat
        
        T[:3, :3] = R_mat
        T[:3, 3] = V @ v
    
    return T


def pose_to_SE3(pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
    """
    Convert position and quaternion to SE(3) transformation matrix.
    
    Args:
        pos: 3D position [x, y, z]
        quat: Quaternion [x, y, z, w] (scipy convention)
        
    Returns:
        4x4 homogeneous transformation matrix
    """
    T = np.eye(4)
    T[:3, :3] = R.from_quat(quat).as_matrix()
    T[:3, 3] = pos
    return T


def SE3_to_pose(T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert SE(3) transformation matrix to position and quaternion.
    
    Args:
        T: 4x4 homogeneous transformation matrix
        
    Returns:
        Tuple of (position, quaternion) where:
        - position: [x, y, z]
        - quaternion: [x, y, z, w] (scipy convention)
    """
    pos = T[:3, 3].copy()
    quat = R.from_matrix(T[:3, :3]).as_quat()
    return pos, quat


def compute_se3_distance(twist: np.ndarray, translation_weight: float = 1.0, 
                         rotation_weight: float = 1.0) -> float:
    """
    Compute a weighted distance metric from an se(3) twist.
    
    This provides a measure of "how far" a transformation is,
    combining rotation and translation appropriately. Useful for:
    - Determining trajectory length
    - Comparing transformations
    - Convergence checking
    
    Args:
        twist: 6D twist [omega, v]
        translation_weight: Weight for translation component (default 1.0)
        rotation_weight: Weight for rotation component (default 1.0)
        
    Returns:
        Weighted distance metric (float)
        
    Note:
        The rotation distance is the angle in radians.
        The translation distance is the Euclidean norm.
    """
    omega = twist[:3]
    v = twist[3:]
    
    rot_dist = np.linalg.norm(omega)  # Rotation angle in radians
    trans_dist = np.linalg.norm(v)    # Translation distance
    
    return translation_weight * trans_dist + rotation_weight * rot_dist


def interpolate_SE3(T_start: np.ndarray, T_end: np.ndarray, s: float) -> np.ndarray:
    """
    Interpolate between two SE(3) transformations using geodesic interpolation.
    
    Computes T(s) = T_start @ exp(s * log(T_start^-1 @ T_end))
    
    This produces smooth, natural motion along the geodesic (shortest path)
    on the SE(3) manifold.
    
    Args:
        T_start: Starting 4x4 transformation matrix
        T_end: Ending 4x4 transformation matrix
        s: Interpolation parameter in [0, 1]
           s=0 gives T_start, s=1 gives T_end
           
    Returns:
        Interpolated 4x4 transformation matrix
    """
    # Compute relative transformation
    T_start_inv = np.eye(4)
    T_start_inv[:3, :3] = T_start[:3, :3].T
    T_start_inv[:3, 3] = -T_start[:3, :3].T @ T_start[:3, 3]
    
    T_rel = T_start_inv @ T_end
    
    # Map to Lie algebra, scale, and map back
    twist = SE3_log(T_rel)
    T_interp = T_start @ SE3_exp(s * twist)
    
    return T_interp
