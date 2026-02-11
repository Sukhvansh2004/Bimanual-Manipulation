"""
Unified grasp pose generator for objects in the DualArmEnvironment.

Uses the Grasp Service (via ServiceClient) to generate grasps.
"""

import numpy as np
from typing import List, Optional, Union
import logging
from contextlib import nullcontext

from robosuite_franka_manipulation import constants
from robosuite_franka_manipulation.services.services_core import ServiceClient
from robosuite_franka_manipulation.transforms import StampedPose
from robosuite_franka_manipulation.schema import GraspSchema, GraspRequest, GraspResponse
from robosuite_franka_manipulation.environments.dual_arm import DualArmEnvironment

# Configure logging
logger = logging.getLogger(__name__)

class GraspGenerator:
    """
    Generates grasp poses for objects in the environment using the Grasp Service.
    """
    
    def __init__(self, env: DualArmEnvironment, service_client: Optional[ServiceClient] = None):
        """
        Initialize with the environment.
        
        Args:
            env: The DualArmEnvironment instance.
            service_client: Optional existing ServiceClient instance.
        """
        self.env = env
        self.service_client = service_client if service_client is not None else ServiceClient()

    def _get_object_pose_stamped(self, object_name: str, env_lock=None) -> Optional[StampedPose]:
        """Get object pose as StampedPose."""
        ctx = env_lock if env_lock is not None else nullcontext()
        with ctx:
            # env.get_object_pose returns a 4x4 matrix
            pose_mat = self.env.get_object_pose(object_name)
            
        if pose_mat is None:
            return None
            
        return StampedPose.from_matrix(pose_mat, frame_id=constants.WORLD_FRAME)

    def generate_grasps(
        self,
        object_name: str,
        gripper_types: Union[str, List[str]],
        num_grasps: int = constants.DEFAULT_NUM_GRASPS,
        env_lock=None
    ) -> List[GraspSchema]:
        """
        Generate grasps for a specific object and gripper type(s) using the Grasp Service.
        
        Args:
            object_name: Name of the object in the environment.
            gripper_types: Type of gripper(s) ('parallel_jaw', 'suction', 'franka_panda', etc.).
            num_grasps: Number of grasps to generate.
            env_lock: Optional lock for environment access.
            
        Returns:
            List of GraspSchema objects.
        """
        # 1. Resolve Object Data
        pose = self._get_object_pose_stamped(object_name, env_lock=env_lock)
        if pose is None:
            logger.error(f"Could not get pose for object: {object_name}")
            return []
            
        if isinstance(gripper_types, str):
            g_types = [gripper_types]
        else:
            g_types = gripper_types
        
        req = GraspRequest(
            command="GET_GRASPS",
            object_name=object_name,
            object_pose=pose,
            gripper_type=g_types,
            num_grasps=num_grasps,
        )
        
        logger.info(f"Requesting {num_grasps} grasps for {object_name} ({g_types}) from Grasp Service...")
        
        response_dict = self.service_client.call("grasp_service", req.model_dump())
        
        if response_dict.get('status') != 'success':
            logger.error(f"Grasp service call failed: {response_dict.get('message')}")
            return []
            
        try:
            response = GraspResponse(**response_dict)
            if response.grasps:
                logger.info(f"Received {len(response.grasps)} grasps.")
                return response.grasps
            else:
                logger.warning("Grasp service returned 0 grasps.")
                return []
        except Exception as e:
            logger.error(f"Failed to parse grasp response: {e}")
            return []

    def __del__(self):
        self.service_client.close()


def maximize_entropy_grasps(grasps: List[GraspSchema], num_grasps: int) -> List[GraspSchema]:
    """
    Select a subset of grasps that maximizes entropy (diversity and quality).
    Uses a greedy Determinantal Point Process (DPP) approach.
    
    The entropy of a subset S in a DPP defined by L-ensemble matrix L is log det(L_S).
    We greedily select grasps to maximize this determinant.
    
    Args:
        grasps: List of GraspSchema objects.
        num_grasps: Number of grasps to select.
        
    Returns:
        A list of selected GraspSchema objects.
    """
    if not grasps or num_grasps <= 0:
        return []
    
    if len(grasps) <= num_grasps:
        # Sort by score anyway if we return all of them
        return sorted(grasps, key=lambda g: g.score, reverse=True)

    # 1. Prepare features for each grasp
    # We use position and orientation as features
    num_total = len(grasps)
    positions = np.array([g.pos for g in grasps])
    quaternions = np.array([g.quat for g in grasps])
    # Ensure scores are positive for DPP L-ensemble
    scores = np.array([max(g.score, 1e-4) for g in grasps])
    
    # 2. Compute similarity kernel matrix K
    # K_ij = exp(-0.5 * d(g_i, g_j)^2)
    
    # Pairwise squared Euclidean distance for positions
    # (dist_pos^2)_ij = ||p_i - p_j||^2
    diff_pos = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist_pos_sq = np.sum(diff_pos**2, axis=-1)
    
    # Pairwise distance for orientations
    # d_rot = 1 - |<q_i, q_j>|
    # dot_products = |<q_i, q_j>|
    dot_products = np.abs(np.einsum('id,jd->ij', quaternions, quaternions))
    # Clip for numerical stability
    dot_products = np.clip(dot_products, 0.0, 1.0)
    dist_rot = 1.0 - dot_products
    
    # Combined distance squared
    # sigma_p and sigma_r control the diversity scale
    sigma_p = 0.05  # 5cm
    sigma_r = constants.TRAJECTORY_ORIENTATION_SCALING if hasattr(constants, 'TRAJECTORY_ORIENTATION_SCALING') else 0.05
    
    # d^2 = (dist_pos / sigma_p)^2 + (dist_rot / sigma_r)^2
    d_sq = dist_pos_sq / (sigma_p**2) + (dist_rot**2) / (sigma_r**2)
    K = np.exp(-0.5 * d_sq)
    
    # 3. Construct L-ensemble matrix L
    # L_ij = score_i * score_j * K_ij
    L = scores[:, np.newaxis] * scores[np.newaxis, :] * K
    
    # 4. Greedy selection to maximize det(L_S)
    selected_indices = []
    remaining_indices = list(range(num_total))
    
    # Start with the grasp with the highest score
    first_idx = int(np.argmax(scores))
    selected_indices.append(first_idx)
    remaining_indices.remove(first_idx)
    
    # Iteratively add elements that maximize the conditional log-determinant (Schur complement)
    for _ in range(num_grasps - 1):
        best_gain = -np.inf
        best_idx = -1
        
        S = selected_indices
        L_SS = L[np.ix_(S, S)]
        
        try:
            # Add a small epsilon for numerical stability
            L_SS_inv = np.linalg.inv(L_SS + np.eye(len(S)) * 1e-6)
        except np.linalg.LinAlgError:
            L_SS_inv = np.linalg.pinv(L_SS)
            
        # This loop can be vectorized if num_total is very large
        L_ii_diag = np.diag(L)
        # We only need L_iS for i in remaining_indices
        L_rem_S = L[remaining_indices, :][:, S]
        
        # gains = L_ii - L_iS @ L_SS_inv @ L_Si
        # L_iS is (len(rem), len(S))
        # L_SS_inv is (len(S), len(S))
        # Result of L_rem_S @ L_SS_inv @ L_rem_S.T is (len(rem), len(rem))
        # We only want the diagonal
        intermediate = L_rem_S @ L_SS_inv
        gains = L_ii_diag[remaining_indices] - np.einsum('ij,ij->i', intermediate, L_rem_S)
        
        best_idx_in_rem = np.argmax(gains)
        best_idx = remaining_indices[best_idx_in_rem]
        
        if gains[best_idx_in_rem] < 0:
            # No more positive gain possible (numerically)
            break
            
        selected_indices.append(best_idx)
        remaining_indices.remove(best_idx)
        
    return [grasps[i] for i in selected_indices]
