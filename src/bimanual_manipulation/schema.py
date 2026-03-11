"""
Pydantic models for data validation and JSON serialization across services.
"""

from typing import List, Optional, Dict, Any, Union, Literal
from pydantic import BaseModel, Field, field_validator, ConfigDict
import numpy as np
import time

from bimanual_manipulation import constants
from bimanual_manipulation.transforms import StampedPose

class BaseSchema(BaseModel):
    """Base schema with common configuration."""
    model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)

class DetectionSchema(BaseSchema):
    """Schema for a single object detection."""
    name: str
    id: Union[int, str]
    pose: StampedPose
    pcd: List[List[float]] = Field(default_factory=list, description="Point cloud [N, 3]")
    mask: Optional[List[List[int]]] = Field(None, description="Segmentation mask")

    @field_validator('pcd', 'mask', mode='before')
    @classmethod
    def validate_array_list(cls, v: Any) -> Any:
        if isinstance(v, np.ndarray):
            return v.tolist()
        return v

class GraspSchema(StampedPose):
    """Schema for a predicted grasp."""
    score: float
    id: Union[int, str]
    pcd_size: int = 0
    gripper_type: str = Field(..., description="Type of gripper used for this grasp")

    @field_validator('gripper_type')
    @classmethod
    def validate_gripper_type(cls, v: str) -> str:
        if v not in constants.GRIPPER_TYPE_MAPPING.values():
            raise ValueError(f"Invalid gripper_type: '{v}'. Must be one of {constants.GRIPPER_TYPES}")
        return v
    
# --- Service Request/Response Models ---

class ResponseStatus(BaseSchema):
    """Standard response status."""
    status: str = "success"
    message: Optional[str] = None
    timestamp: float = Field(default_factory=time.time)

# FastSAM Service
class FastSAMRequest(BaseSchema):
    command: str
    prompt: Optional[str] = None

class FastSAMResponse(ResponseStatus):
    prompt: Optional[str] = None
    detections: Optional[List[DetectionSchema]] = None

# Grasp Service
class GraspRequest(BaseSchema):
    command: str
    pcd: Optional[List[List[float]]] = None
    object_pose: Optional[StampedPose] = None
    object_name: Optional[str] = None
    num_grasps: int = constants.DEFAULT_NUM_GRASPS
    collision_check: Literal[None, "Inter", "Intra"]
    visualize_gripper_mesh: bool = constants.DEFAULT_VISUALIZE_GRIPPER_MESH
    gripper_type: List[str] = Field(..., min_length=1)

    @field_validator('gripper_type')
    @classmethod
    def validate_gripper_type(cls, v: List[str]) -> List[str]:
        for g_type in v:
            if g_type not in constants.GRIPPER_TYPES:
                raise ValueError(f"Invalid gripper_type: '{g_type}'. Must be one of {constants.GRIPPER_TYPES}")
        return v

    @field_validator('pcd', mode='before')
    @classmethod
    def validate_pcd(cls, v: Any) -> Any:
        if isinstance(v, np.ndarray):
            return v.tolist()
        return v

class GraspResponse(ResponseStatus):
    grasps: Optional[List[GraspSchema]]
    num_grasps_predicted: int = 0
    pcd_size: int = 0
    gripper_type: List[str] = Field(..., min_length=1)

    @field_validator('gripper_type')
    @classmethod
    def validate_gripper_type(cls, v: List[str]) -> List[str]:
        for g_type in v:
            if g_type not in constants.GRIPPER_TYPES:
                raise ValueError(f"Invalid gripper_type: '{g_type}'. Must be one of {constants.GRIPPER_TYPES}")
        return v

# Move To Pose Service
class MoveToPoseRequest(BaseSchema):
    command: str
    robot_id: int = 0
    arm: str = "right"
    pose: StampedPose
    gripper_action: float = 0.0
    num_steps: Optional[int] = None

class FollowTrajectoryRequest(BaseSchema):
    command: str
    robot_id: int = 0
    arm: str = "right"
    trajectory: List[StampedPose]
    gripper_action: float = 0.0
    num_steps: Optional[int] = None
    smoothen: bool = False

class GetArmStatusRequest(BaseSchema):
    command: str
    robot_id: int = 0
    arm: str = "right"

class CancelMoveRequest(BaseSchema):
    command: str
    robot_id: int = 0
    arm: str = "right"

class MoveToJointConfigRequest(BaseSchema):
    """Request to move to a target joint configuration (JOINT_POSITION controller only)."""
    command: str
    robot_id: int = 0
    arm: str = "right"
    joint_config: List[float] = Field(..., min_length=7, max_length=7,
                                       description="Target joint angles (7 DOF)")
    gripper_action: float = 0.0
    num_steps: int = constants.TRAJECTORY_NUM_POINTS

class FollowJointTrajectoryRequest(BaseSchema):
    """Request to follow a joint-space trajectory (JOINT_POSITION controller only)."""
    command: str
    robot_id: int = 0
    arm: str = "right"
    joint_trajectory: List[List[float]] = Field(..., min_length=1,
                                                 description="List of joint configs, each 7 DOF")
    gripper_action: float = 0.0
    num_steps: Optional[int] = None
    smoothen: bool = False

class MoveToPoseResponse(ResponseStatus):
    pass

class ArmStatusResponse(ResponseStatus):
    status_code: int = 1 # -1: moving, 1: success, 0: failure

# --- Service Manager Schemas ---

class ServiceInfo(BaseSchema):
    """Information about a registered service."""
    name: str
    host: str
    port: int
    type: str
    description: Optional[str] = ""
    registered_at: float = Field(default_factory=time.time)

class RegisterServiceRequest(BaseSchema):
    command: str = "REGISTER"
    name: str
    host: str
    port: int
    type: str
    description: Optional[str] = ""

class LookupServiceRequest(BaseSchema):
    command: str = "LOOKUP"
    name: str

class CallServiceRequest(BaseSchema):
    command: str = "CALL_SERVICE"
    service_name: str
    request: Dict[str, Any]

class ListServicesRequest(BaseSchema):
    command: str = "LIST_SERVICES"

class ServiceListResponse(ResponseStatus):
    services: Dict[str, ServiceInfo] = {}

class ServiceLookupResponse(ResponseStatus):
    service: Optional[ServiceInfo] = None
