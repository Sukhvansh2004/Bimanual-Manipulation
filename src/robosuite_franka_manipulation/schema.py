"""
Pydantic models for data validation and JSON serialization across services.
"""

from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, field_validator, ConfigDict
import numpy as np
import time

from robosuite_franka_manipulation import constants

class BaseSchema(BaseModel):
    """Base schema with common configuration."""
    model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)

def _numpy_to_list(v: Any) -> Any:
    """Helper to convert numpy arrays to lists."""
    if isinstance(v, np.ndarray):
        return v.flatten().tolist()
    return v

class PoseSchema(BaseSchema):
    """Schema for StampedPose data."""
    pos: List[float] = Field(..., description="Position [x, y, z]")
    quat: List[float] = Field(..., description="Quaternion [x, y, z, w]")
    frame_id: str = Field(constants.WORLD_FRAME, description="Coordinate frame ID")
    timestamp: float = Field(default_factory=time.time, description="Timestamp")

    @field_validator('pos', 'quat', mode='before')
    @classmethod
    def validate_numpy(cls, v: Any) -> Any:
        return _numpy_to_list(v)

class DetectionSchema(BaseSchema):
    """Schema for a single object detection."""
    name: str
    id: Union[int, str]
    pose: PoseSchema
    pcd: List[List[float]] = Field(default_factory=list, description="Point cloud [N, 3]")
    mask: Optional[List[List[int]]] = Field(None, description="Segmentation mask")

    @field_validator('pcd', 'mask', mode='before')
    @classmethod
    def validate_array_list(cls, v: Any) -> Any:
        if isinstance(v, np.ndarray):
            return v.tolist()
        return v

class GraspSchema(BaseSchema):
    """Schema for a predicted grasp."""
    pos: List[float]
    quat: List[float]
    score: float
    id: int
    frame_id: str = constants.WORLD_FRAME
    pcd_size: int = 0
    prediction_timestamp: float = Field(default_factory=time.time)

    @field_validator('pos', 'quat', mode='before')
    @classmethod
    def validate_numpy(cls, v: Any) -> Any:
        return _numpy_to_list(v)

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
    object_pose: Optional[Union[PoseSchema, Dict[str, Any]]] = None
    num_grasps: int = constants.DEFAULT_NUM_GRASPS
    gripper_type: str = "finger"

    @field_validator('pcd', mode='before')
    @classmethod
    def validate_pcd(cls, v: Any) -> Any:
        if isinstance(v, np.ndarray):
            return v.tolist()
        return v

class GraspResponse(ResponseStatus):
    grasps: Optional[List[GraspSchema]] = None
    num_grasps_predicted: int = 0
    pcd_size: int = 0
    gripper_type: Optional[str] = None

# EEF Service
class EEFRequest(BaseSchema):
    command: str

class EEFResponse(BaseSchema):
    success: bool
    error: Optional[str] = None
    eef_coordinates: Optional[Dict[str, PoseSchema]] = None
    timestamp: float = Field(default_factory=time.time)
    frame: str = constants.WORLD_FRAME
