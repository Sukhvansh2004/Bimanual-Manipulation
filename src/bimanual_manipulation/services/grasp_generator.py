"""
Unified grasp pose generator for objects in the DualArmEnvironment.

Uses the Grasp Service (via ServiceClient) to generate grasps.
"""

import numpy as np
from typing import List, Optional, Union
import logging
from contextlib import nullcontext

from bimanual_manipulation import constants
from bimanual_manipulation.services.services_core import ServiceClient
from bimanual_manipulation.transforms import StampedPose
from bimanual_manipulation.schema import GraspSchema, GraspRequest, GraspResponse
from bimanual_manipulation.environments.dual_arm import DualArmEnvironment

# Configure logging
logger = logging.getLogger(__name__)

class GraspGenerator:
    """Generates grasp poses for objects in the environment using the Grasp Service."""
    
    def __init__(self, env: DualArmEnvironment, service_client: Optional[ServiceClient] = None):
        """Initialize with the environment.

        Args:
            env: The DualArmEnvironment instance.
            service_client: Optional existing ServiceClient instance; a new
                one is created if ``None``.
        """
        self.env = env
        self.service_client = service_client if service_client is not None else ServiceClient()

    def _get_object_pose_stamped(self, object_name: str, env_lock=None) -> Optional[StampedPose]:
        """Get an object's world pose as a StampedPose.

        Args:
            object_name: Name of the object in the environment.
            env_lock: Optional lock held while reading the pose.

        Returns:
            Pose in ``constants.WORLD_FRAME``, or ``None`` if unavailable.
        """
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
        collision_check: Optional[str],
        num_grasps: int = constants.DEFAULT_NUM_GRASPS,
        env_lock=None
    ) -> List[GraspSchema]:
        """Generate grasps for an object and gripper type(s) using the Grasp Service.

        Args:
            object_name: Name of the object in the environment.
            gripper_types: Gripper type key(s) from ``constants.GRIPPER_TYPES``
                (``'finger'``, ``'suction'``).
            collision_check: Type of collision checking (``'Inter'``,
                ``'Intra'``, ``None``).
            num_grasps: Number of grasps to generate.
            env_lock: Optional lock for environment access.

        Returns:
            List of GraspSchema objects; empty if the pose is unavailable, the
            call fails, or the response cannot be parsed.

        Raises:
            pydantic.ValidationError: If ``gripper_types`` or
                ``collision_check`` are invalid for ``GraspRequest``.
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
            collision_check=collision_check,
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
        """Close the service client on garbage collection."""
        self.service_client.close()
