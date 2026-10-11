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
from scipy.spatial.transform import Rotation as R, Slerp
from typing import Tuple

from bimanual_manipulation import constants


def skew(v: np.ndarray) -> np.ndarray:
    """Create a skew-symmetric matrix from a 3D vector.

    The skew-symmetric matrix [v]× satisfies: [v]× @ u = v × u (cross product)

    For vector v = [v1, v2, v3], returns:
        [[ 0, -v3,  v2],
         [ v3,  0, -v1],
         [-v2,  v1,  0]]

    Args:
        v: Vector of shape ``(3,)``.

    Returns:
        Skew-symmetric matrix of shape ``(3, 3)``.
    """
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])


def unskew(S: np.ndarray) -> np.ndarray:
    """Extract the 3D vector from a skew-symmetric matrix.

    Inverse of skew(): unskew(skew(v)) = v. Reads only the lower/upper
    off-diagonal entries ``S[2, 1], S[0, 2], S[1, 0]``; symmetry is not checked.

    Args:
        S: Skew-symmetric matrix of shape ``(3, 3)``.

    Returns:
        Vector of shape ``(3,)``.
    """
    return np.array([S[2, 1], S[0, 2], S[1, 0]])


def se3_to_matrix(twist: np.ndarray) -> np.ndarray:
    """Convert a 6D twist vector to its 4x4 se(3) matrix representation.

    Twist format: [omega (3), v (3)] where omega is angular velocity
    and v is linear velocity.

    The se(3) matrix has the form:
        [[skew(omega), v],
         [0, 0, 0,     0]]

    Args:
        twist: Twist ``[omega, v]`` of shape ``(6,)``.

    Returns:
        se(3) matrix of shape ``(4, 4)``.
    """
    omega = twist[:3]
    v = twist[3:]
    mat = np.zeros((4, 4))
    mat[:3, :3] = skew(omega)
    mat[:3, 3] = v
    return mat


def matrix_to_se3(mat: np.ndarray) -> np.ndarray:
    """Convert a 4x4 se(3) matrix to a 6D twist vector.

    Inverse of se3_to_matrix(): matrix_to_se3(se3_to_matrix(twist)) = twist

    Args:
        mat: se(3) matrix of shape ``(4, 4)``.

    Returns:
        Twist ``[omega, v]`` of shape ``(6,)``.
    """
    omega = unskew(mat[:3, :3])
    v = mat[:3, 3]
    return np.concatenate([omega, v])


def SE3_log(T: np.ndarray) -> np.ndarray:
    """Compute the matrix logarithm of an SE(3) transformation matrix.

    Maps SE(3) -> se(3) (Lie group to Lie algebra) in closed form.

    This is the inverse of SE3_exp(): SE3_exp(SE3_log(T)) ≈ T

    The logarithm gives the "twist" that, when integrated for unit time,
    produces the transformation T.

    Args:
        T: Homogeneous transformation matrix of shape ``(4, 4)``.

    Returns:
        Twist ``[omega, v]`` of shape ``(6,)``, with ``|omega|`` the rotation
        angle in radians.

    Note:
        Handles three cases:
        - Near identity (theta < 1e-8): ``omega = 0``, ``v = p``
        - Near 180° rotation (|theta - π| < 1e-8): axis from the diagonal
        - General case: Closed-form Rodrigues logarithm
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
    """Compute the matrix exponential of an se(3) twist vector.

    Maps se(3) -> SE(3) (Lie algebra to Lie group) using the closed-form
    Rodrigues formula.

    This is the inverse of SE3_log(): SE3_log(SE3_exp(twist)) ≈ twist

    The exponential "integrates" the twist for unit time to produce
    a rigid body transformation.

    Args:
        twist: Twist ``[omega, v]`` of shape ``(6,)``, where ``omega`` is the
            rotation axis times angle (radians) and ``v`` the linear part.

    Returns:
        Homogeneous transformation matrix of shape ``(4, 4)``.

    Note:
        Uses the Rodrigues formula for rotation and the V matrix for
        the coupled rotation-translation relationship. For
        ``|omega| < 1e-8`` the rotation is the first-order
        ``I + skew(omega)`` and the translation is ``v``.
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
    """Convert a position and quaternion to an SE(3) transformation matrix.

    Args:
        pos: Position ``[x, y, z]``.
        quat: Quaternion ``[x, y, z, w]`` (scipy convention).

    Returns:
        Homogeneous transformation matrix of shape ``(4, 4)``.
    """
    T = np.eye(4)
    T[:3, :3] = R.from_quat(quat).as_matrix()
    T[:3, 3] = pos
    return T


def SE3_to_pose(T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert an SE(3) transformation matrix to position and quaternion.

    Args:
        T: Homogeneous transformation matrix of shape ``(4, 4)``.

    Returns:
        ``(position, quaternion)`` with position ``(3,)`` and quaternion
        ``[x, y, z, w]`` (scipy convention) ``(4,)``.
    """
    pos = T[:3, 3].copy()
    quat = R.from_matrix(T[:3, :3]).as_quat()
    return pos, quat


def compute_se3_distance(twist: np.ndarray, translation_weight: float = 1.0, 
                         rotation_weight: float = 1.0) -> float:
    """Compute a weighted distance metric from an se(3) twist.

    Returns ``translation_weight * |v| + rotation_weight * |omega|``.
    Useful for trajectory length, comparing transformations, and
    convergence checking.

    Args:
        twist: Twist ``[omega, v]`` of shape ``(6,)``.
        translation_weight: Weight for the linear component.
        rotation_weight: Weight for the angular component.

    Returns:
        Weighted distance (float).

    Note:
        ``|omega|`` is the rotation angle in radians. ``|v|`` is the norm of
        the twist's linear part, which equals the translation distance only
        when the rotation is zero.
    """
    omega = twist[:3]
    v = twist[3:]
    
    rot_dist = np.linalg.norm(omega)  # Rotation angle in radians
    trans_dist = np.linalg.norm(v)    # Translation distance
    
    return translation_weight * trans_dist + rotation_weight * rot_dist


def interpolate_SE3(T_start: np.ndarray, T_end: np.ndarray, s: float) -> np.ndarray:
    """Interpolate between two SE(3) transformations along the geodesic.

    Computes T(s) = T_start @ exp(s * log(T_start^-1 @ T_end))

    Args:
        T_start: Starting transformation matrix ``(4, 4)``.
        T_end: Ending transformation matrix ``(4, 4)``.
        s: Interpolation parameter in ``[0, 1]``; ``s=0`` gives ``T_start``,
            ``s=1`` gives ``T_end``.

    Returns:
        Interpolated transformation matrix of shape ``(4, 4)``.
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


def interpolate_pose(start_pos: np.ndarray, start_quat: np.ndarray, 
                     end_pos: np.ndarray, end_quat: np.ndarray, 
                     t: float,
                     interpolation_type: str = None) -> Tuple[np.ndarray, np.ndarray]:
    """Interpolate between two poses.

    Supports two modes controlled by ``interpolation_type``
    (defaults to ``constants.INTERPOLATION_TYPE``):

    - ``'linear'``: Decoupled interpolation — linear (lerp) for translation,
      spherical linear (slerp) for rotation. Faster, but rotation and
      translation are not coupled.
    - ``'SE3'`` (any value other than ``'linear'``): Geodesic interpolation
      on the SE(3) manifold. Produces coupled rotation-translation motion.

    Args:
        start_pos: Start position ``[x, y, z]``.
        start_quat: Start quaternion ``[x, y, z, w]``.
        end_pos: End position ``[x, y, z]``.
        end_quat: End quaternion ``[x, y, z, w]``.
        t: Interpolation parameter in ``[0, 1]``.
        interpolation_type: ``'linear'`` or ``'SE3'`` (default:
            ``constants.INTERPOLATION_TYPE``).

    Returns:
        ``(position, quaternion)`` of shapes ``(3,)`` and ``(4,)``,
        quaternion ``[x, y, z, w]``.
    """
    if interpolation_type is None:
        interpolation_type = constants.INTERPOLATION_TYPE
    
    if interpolation_type == 'linear':
        pos = (1 - t) * np.asarray(start_pos) + t * np.asarray(end_pos)
        
        r_start = R.from_quat(start_quat)
        r_end = R.from_quat(end_quat)
        slerp = Slerp([0.0, 1.0], R.concatenate([r_start, r_end]))
        quat = slerp([t]).as_quat()[0]
        
        return pos, quat
    else:
        # SE(3) geodesic interpolation
        T_start = pose_to_SE3(start_pos, start_quat)
        T_end = pose_to_SE3(end_pos, end_quat)
        T_interp = interpolate_SE3(T_start, T_end, t)
        return SE3_to_pose(T_interp)
