"""
StampedPose and Transform classes for representing poses with frame information.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Union, List, Sequence, Any
from scipy.spatial.transform import Rotation as R
import time
from pydantic import BaseModel, Field, ConfigDict, field_validator, field_serializer, AliasChoices
from bimanual_manipulation import constants

# Type for array-like inputs (lists, tuples, np.ndarray)
ArrayInput = Union[np.ndarray, Sequence[float], List[float]]


class StampedPose(BaseModel):
    """A pose (position + orientation) with an associated coordinate frame.

    ``pos`` and ``quat`` are stored as read-only ``float64`` arrays; the
    quaternion is normalized on validation (unless its norm is ~0). The
    constructor also accepts ``position`` / ``quaternion`` as aliases.

    Attributes:
        pos: Position ``[x, y, z]``, shape ``(3,)``.
        quat: Orientation quaternion ``[x, y, z, w]`` (scipy convention),
            shape ``(4,)``.
        frame_id: Frame the pose is expressed in (default
            ``constants.WORLD_FRAME``).
        timestamp: Creation time from ``time.time()`` unless given.
    """
    model_config = ConfigDict(
        arbitrary_types_allowed=True, 
        populate_by_name=True,
        validate_assignment=True 
    )

    pos: np.ndarray = Field(validation_alias=AliasChoices('pos', 'position'))
    quat: np.ndarray = Field(validation_alias=AliasChoices('quat', 'quaternion'))
    frame_id: str = Field(default=constants.WORLD_FRAME)
    timestamp: float = Field(default_factory=time.time)

    @property
    def position(self) -> np.ndarray:
        """Alias for :attr:`pos`.

        Returns:
            Position array of shape ``(3,)``.
        """
        return self.pos
    
    @position.setter
    def position(self, value: Any):
        """Set :attr:`pos` (validated on assignment).

        Args:
            value: Array-like of shape ``(3,)``.
        """
        self.pos = value

    @property
    def quaternion(self) -> np.ndarray:
        """Alias for :attr:`quat`.

        Returns:
            Quaternion ``[x, y, z, w]`` of shape ``(4,)``.
        """
        return self.quat
    
    @quaternion.setter
    def quaternion(self, value: Any):
        """Set :attr:`quat` (validated and normalized on assignment).

        Args:
            value: Array-like quaternion ``[x, y, z, w]``.
        """
        self.quat = value

    # --- Validators ---
    @field_validator("pos", "quat", mode="before")
    @classmethod
    def convert_to_numpy(cls, v: Any) -> np.ndarray:
        """Convert ``pos`` / ``quat`` input to a read-only ``float64`` array.

        Args:
            v: Array-like input.

        Returns:
            Non-writeable ``float64`` array.
        """
        arr = np.asarray(v, dtype=np.float64)
        arr.flags.writeable = False
        return arr

    @field_validator("pos")
    @classmethod
    def validate_position_shape(cls, v: np.ndarray) -> np.ndarray:
        """Check that the position has shape ``(3,)``.

        Args:
            v: Position array.

        Returns:
            The unchanged array.

        Raises:
            ValueError: If ``v.shape != (3,)``.
        """
        if v.shape != (3,):
            raise ValueError(f"Position must be 3D, got shape {v.shape}")
        return v

    @field_validator("quat")
    @classmethod
    def validate_quaternion(cls, v: np.ndarray) -> np.ndarray:
        """Check quaternion shape and normalize it.

        Quaternions with norm ``<= 1e-10`` are returned unnormalized.

        Args:
            v: Quaternion array ``[x, y, z, w]``.

        Returns:
            Read-only unit quaternion of shape ``(4,)``.

        Raises:
            ValueError: If ``v.shape != (4,)``.
        """
        if v.shape != (4,):
            raise ValueError(f"Quaternion must be 4D, got shape {v.shape}")
        
        # Normalize to ensure valid rotation
        norm = np.linalg.norm(v)
        if norm > 1e-10:
            v = v / norm
            v.flags.writeable = False # Re-lock after modification
        return v

    # --- Serializers ---
    @field_serializer("pos", "quat")
    def serialize_numpy(self, v: np.ndarray, _info) -> list:
        """Serialize ``pos`` / ``quat`` arrays to flat Python lists.

        Args:
            v: Array to serialize.
            _info: Pydantic serialization info (unused).

        Returns:
            Flattened list of floats.
        """
        return v.flatten().tolist()
    
    # --- Helpers ---
    def to_dict(self) -> dict:
        """Dump the pose to a dict (arrays serialized as lists).

        Returns:
            Dict with keys ``pos``, ``quat``, ``frame_id``, ``timestamp``.
        """
        return self.model_dump()
    
    @classmethod
    def from_matrix(cls, matrix: np.ndarray, frame_id: str) -> "StampedPose":
        """Create a pose from a 4x4 homogeneous matrix.

        Args:
            matrix: Homogeneous transform of shape ``(4, 4)``.
            frame_id: Frame the pose is expressed in.

        Returns:
            New :class:`StampedPose` with an ``[x, y, z, w]`` quaternion.

        Raises:
            ValueError: If ``matrix.shape != (4, 4)``.
        """
        if matrix.shape != (4, 4):
            raise ValueError(f"Matrix must be 4x4, got shape {matrix.shape}")
        
        return cls(
            pos=matrix[:3, 3], 
            quat=R.from_matrix(matrix[:3, :3]).as_quat(), 
            frame_id=frame_id
        )
    
    def to_matrix(self) -> np.ndarray:
        """Convert the pose to a 4x4 homogeneous matrix.

        Returns:
            Array of shape ``(4, 4)``.
        """
        matrix = np.eye(4)
        matrix[:3, :3] = R.from_quat(self.quat).as_matrix()
        matrix[:3, 3] = self.pos
        return matrix
    
    def copy(self) -> "StampedPose":
        """Return a deep copy of the pose.

        Returns:
            New :class:`StampedPose` with the same fields.
        """
        return self.model_copy(deep=True)

    def __repr__(self) -> str:
        """Return a compact string with position, quaternion, and frame.

        Returns:
            String representation (timestamp omitted).
        """
        # np.array2string makes the output cleaner than tolist() for debugging
        pos_str = np.array2string(self.pos, precision=4, separator=',', suppress_small=True)
        quat_str = np.array2string(self.quat, precision=4, separator=',', suppress_small=True)
        return f"StampedPose(pos={pos_str}, quat={quat_str}, frame='{self.frame_id}')"

@dataclass
class Transform:
    """A transformation between two coordinate frames.

    Similar to ``geometry_msgs/TransformStamped`` in ROS. Maps points
    expressed in ``child_frame`` into ``parent_frame``.

    Attributes:
        translation: Translation ``[x, y, z]`` of the child origin in the
            parent frame, shape ``(3,)``.
        rotation: Rotation quaternion ``[x, y, z, w]`` (scipy convention),
            shape ``(4,)``.
        parent_frame: The parent (reference) frame.
        child_frame: The child frame being described relative to parent.
        timestamp: Creation time; set to ``time.time()`` if ``None``.
        is_static: Whether this transform is static (doesn't change over time).

    Note:
        ``translation`` and ``rotation`` accept lists or arrays but are
        converted to ``float64`` arrays (and the quaternion normalized) in
        ``__post_init__``.
    """
    translation: ArrayInput
    rotation: ArrayInput
    parent_frame: str
    child_frame: str
    timestamp: Optional[float] = None
    is_static: bool = False
    
    def __post_init__(self):
        """Convert fields to arrays, validate shapes, and normalize the quaternion.

        Raises:
            ValueError: If ``translation`` is not shape ``(3,)`` or
                ``rotation`` is not shape ``(4,)``.
        """
        self.translation = np.asarray(self.translation, dtype=np.float64)
        self.rotation = np.asarray(self.rotation, dtype=np.float64)
        
        if self.translation.shape != (3,):
            raise ValueError(f"Translation must be 3D, got shape {self.translation.shape}")
        if self.rotation.shape != (4,):
            raise ValueError(f"Rotation must be 4D quaternion, got shape {self.rotation.shape}")
        
        # Normalize quaternion
        norm = np.linalg.norm(self.rotation)
        if norm > 1e-10:
            self.rotation = self.rotation / norm
        
        if self.timestamp is None:
            self.timestamp = time.time()
    
    @classmethod
    def identity(cls, parent_frame: str, child_frame: str, is_static: bool = True) -> "Transform":
        """Create an identity transform between two frames.

        Args:
            parent_frame: Parent frame name.
            child_frame: Child frame name.
            is_static: Whether the transform is static.

        Returns:
            Transform with zero translation and quaternion ``[0, 0, 0, 1]``.
        """
        return cls(
            translation=np.zeros(3),
            rotation=np.array([0, 0, 0, 1]),
            parent_frame=parent_frame,
            child_frame=child_frame,
            is_static=is_static
        )
    
    @classmethod
    def from_matrix(cls, matrix: np.ndarray, parent_frame: str, child_frame: str, 
                    is_static: bool = False) -> "Transform":
        """Create a Transform from a 4x4 transformation matrix.

        Args:
            matrix: Homogeneous transform ``T_parent_child`` of shape ``(4, 4)``.
            parent_frame: Parent frame name.
            child_frame: Child frame name.
            is_static: Whether the transform is static.

        Returns:
            New :class:`Transform`.

        Raises:
            ValueError: If ``matrix.shape != (4, 4)``.
        """
        if matrix.shape != (4, 4):
            raise ValueError(f"Matrix must be 4x4, got shape {matrix.shape}")
        
        translation = matrix[:3, 3]
        rotation = R.from_matrix(matrix[:3, :3]).as_quat()
        
        return cls(
            translation=translation,
            rotation=rotation,
            parent_frame=parent_frame,
            child_frame=child_frame,
            is_static=is_static
        )
    
    def to_matrix(self) -> np.ndarray:
        """Convert to a 4x4 homogeneous transformation matrix.

        Returns:
            Array ``T_parent_child`` of shape ``(4, 4)``.
        """
        rot = np.asarray(self.rotation)
        trans = np.asarray(self.translation)
        matrix = np.eye(4)
        matrix[:3, :3] = R.from_quat(rot).as_matrix()
        matrix[:3, 3] = trans
        return matrix
    
    def inverse(self) -> "Transform":
        """Get the inverse transform (frames swapped).

        Returns:
            New :class:`Transform` with ``parent_frame``/``child_frame``
            swapped and the inverse rotation/translation. ``is_static`` is
            preserved; ``timestamp`` is reset to the current time.
        """
        rot = R.from_quat(np.asarray(self.rotation))
        inv_rot = rot.inv()
        inv_translation = -inv_rot.apply(np.asarray(self.translation))
        
        return Transform(
            translation=inv_translation,
            rotation=inv_rot.as_quat(),
            parent_frame=self.child_frame,
            child_frame=self.parent_frame,
            is_static=self.is_static
        )
    
    def compose(self, other: "Transform") -> "Transform":
        """Compose this transform with another: ``self * other``.

        Args:
            other: Transform whose ``parent_frame`` equals this transform's
                ``child_frame``.

        Returns:
            New :class:`Transform` from ``self.parent_frame`` to
            ``other.child_frame``; static only if both inputs are static.

        Raises:
            ValueError: If ``self.child_frame != other.parent_frame``.
        """
        if self.child_frame != other.parent_frame:
            raise ValueError(
                f"Cannot compose: self.child_frame ({self.child_frame}) != "
                f"other.parent_frame ({other.parent_frame})"
            )
        
        # Compose rotations
        self_rot = R.from_quat(np.asarray(self.rotation))
        other_rot = R.from_quat(np.asarray(other.rotation))
        composed_rot = self_rot * other_rot
        
        # Compose translations: t_composed = t_self + R_self * t_other
        composed_translation = np.asarray(self.translation) + self_rot.apply(np.asarray(other.translation))
        
        return Transform(
            translation=composed_translation,
            rotation=composed_rot.as_quat(),
            parent_frame=self.parent_frame,
            child_frame=other.child_frame,
            is_static=self.is_static and other.is_static
        )
    
    def apply_to_pose(self, pose: StampedPose) -> StampedPose:
        """Apply this transform to a pose.

        Args:
            pose: Pose expressed in ``self.child_frame``.

        Returns:
            Transformed pose in ``self.parent_frame`` (new timestamp).

        Raises:
            ValueError: If ``pose.frame_id != self.child_frame``.
        """
        if pose.frame_id != self.child_frame:
            raise ValueError(
                f"Pose frame ({pose.frame_id}) doesn't match transform's "
                f"child_frame ({self.child_frame})"
            )
        
        # Transform position: p_parent = t + R * p_child
        self_rot = R.from_quat(np.asarray(self.rotation))
        new_position = np.asarray(self.translation) + self_rot.apply(np.asarray(pose.position))
        
        # Transform orientation: q_parent = q_self * q_pose
        pose_rot = R.from_quat(np.asarray(pose.quaternion))
        new_rotation = self_rot * pose_rot
        
        return StampedPose(
            position=new_position,
            quaternion=new_rotation.as_quat(),
            frame_id=self.parent_frame
        )
    
    def __repr__(self) -> str:
        """Return a compact string with frames, translation, and rotation.

        Returns:
            String representation.
        """
        trans = np.asarray(self.translation)
        rot = np.asarray(self.rotation)
        static_str = " (static)" if self.is_static else ""
        return (f"Transform({self.parent_frame} -> {self.child_frame}{static_str}, "
                f"t={trans.round(4).tolist()}, "
                f"q={rot.round(4).tolist()})")
