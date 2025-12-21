"""
StampedPose and Transform classes for representing poses with frame information.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Union, List
from scipy.spatial.transform import Rotation as R
import time


@dataclass
class StampedPose:
    """
    A pose (position + orientation) with an associated coordinate frame.
    
    Similar to geometry_msgs/PoseStamped in ROS.
    
    Attributes:
        position: 3D position as numpy array [x, y, z]
        quaternion: Orientation as quaternion [x, y, z, w] (scipy convention)
        frame_id: The coordinate frame this pose is expressed in
        timestamp: Optional timestamp for the pose
    """
    position: np.ndarray
    quaternion: np.ndarray
    frame_id: str
    timestamp: Optional[float] = None
    
    def __post_init__(self):
        """Ensure numpy arrays and validate quaternion."""
        self.position = np.asarray(self.position, dtype=np.float64)
        self.quaternion = np.asarray(self.quaternion, dtype=np.float64)
        
        if self.position.shape != (3,):
            raise ValueError(f"Position must be 3D, got shape {self.position.shape}")
        if self.quaternion.shape != (4,):
            raise ValueError(f"Quaternion must be 4D, got shape {self.quaternion.shape}")
        
        # Normalize quaternion
        norm = np.linalg.norm(self.quaternion)
        if norm > 1e-10:
            self.quaternion = self.quaternion / norm
        
        if self.timestamp is None:
            self.timestamp = time.time()
    
    @classmethod
    def from_dict(cls, data: dict, frame_id: str = None) -> "StampedPose":
        """
        Create StampedPose from a dictionary with 'pos'/'position' and 'quat'/'quaternion' keys.
        
        Args:
            data: Dictionary with position and quaternion
            frame_id: Frame ID (overrides dict value if provided)
            
        Returns:
            StampedPose instance
        """
        pos = data.get('pos', data.get('position'))
        quat = data.get('quat', data.get('quaternion'))
        fid = frame_id or data.get('frame_id', 'unknown')
        ts = data.get('timestamp')
        
        if pos is None or quat is None:
            raise ValueError("Dictionary must contain 'pos'/'position' and 'quat'/'quaternion' keys")
        
        return cls(position=pos, quaternion=quat, frame_id=fid, timestamp=ts)
    
    def to_dict(self) -> dict:
        """Convert to dictionary format."""
        return {
            'pos': self.position.tolist(),
            'quat': self.quaternion.tolist(),
            'frame_id': self.frame_id,
            'timestamp': self.timestamp
        }
    
    @classmethod
    def from_matrix(cls, matrix: np.ndarray, frame_id: str) -> "StampedPose":
        """
        Create StampedPose from a 4x4 transformation matrix.
        
        Args:
            matrix: 4x4 homogeneous transformation matrix
            frame_id: The coordinate frame
            
        Returns:
            StampedPose instance
        """
        if matrix.shape != (4, 4):
            raise ValueError(f"Matrix must be 4x4, got shape {matrix.shape}")
        
        position = matrix[:3, 3]
        rotation = R.from_matrix(matrix[:3, :3])
        quaternion = rotation.as_quat()
        
        return cls(position=position, quaternion=quaternion, frame_id=frame_id)
    
    def to_matrix(self) -> np.ndarray:
        """
        Convert to 4x4 homogeneous transformation matrix.
        
        Returns:
            4x4 transformation matrix
        """
        matrix = np.eye(4)
        matrix[:3, :3] = R.from_quat(self.quaternion).as_matrix()
        matrix[:3, 3] = self.position
        return matrix
    
    def rotation(self) -> R:
        """Get scipy Rotation object for the orientation."""
        return R.from_quat(self.quaternion)
    
    def copy(self) -> "StampedPose":
        """Create a deep copy of this pose."""
        return StampedPose(
            position=self.position.copy(),
            quaternion=self.quaternion.copy(),
            frame_id=self.frame_id,
            timestamp=self.timestamp
        )
    
    def __repr__(self) -> str:
        return (f"StampedPose(pos={self.position.round(4).tolist()}, "
                f"quat={self.quaternion.round(4).tolist()}, "
                f"frame='{self.frame_id}')")


@dataclass
class Transform:
    """
    A transformation between two coordinate frames.
    
    Similar to geometry_msgs/TransformStamped in ROS.
    
    Attributes:
        translation: 3D translation [x, y, z]
        rotation: Rotation as quaternion [x, y, z, w] (scipy convention)
        parent_frame: The parent (reference) frame
        child_frame: The child frame being described relative to parent
        timestamp: Optional timestamp
        is_static: Whether this transform is static (doesn't change over time)
    """
    translation: np.ndarray
    rotation: np.ndarray
    parent_frame: str
    child_frame: str
    timestamp: Optional[float] = None
    is_static: bool = False
    
    def __post_init__(self):
        """Ensure numpy arrays and validate."""
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
        """Create an identity transform between two frames."""
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
        """Create Transform from a 4x4 transformation matrix."""
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
        """Convert to 4x4 homogeneous transformation matrix."""
        matrix = np.eye(4)
        matrix[:3, :3] = R.from_quat(self.rotation).as_matrix()
        matrix[:3, 3] = self.translation
        return matrix
    
    def inverse(self) -> "Transform":
        """
        Get the inverse transform (child -> parent).
        
        Returns:
            New Transform with frames swapped and inverse transformation
        """
        rot = R.from_quat(self.rotation)
        inv_rot = rot.inv()
        inv_translation = -inv_rot.apply(self.translation)
        
        return Transform(
            translation=inv_translation,
            rotation=inv_rot.as_quat(),
            parent_frame=self.child_frame,
            child_frame=self.parent_frame,
            is_static=self.is_static
        )
    
    def compose(self, other: "Transform") -> "Transform":
        """
        Compose this transform with another: self * other.
        
        The other transform's parent_frame should match this transform's child_frame.
        
        Args:
            other: The transform to compose with
            
        Returns:
            New Transform from self.parent_frame to other.child_frame
        """
        if self.child_frame != other.parent_frame:
            raise ValueError(
                f"Cannot compose: self.child_frame ({self.child_frame}) != "
                f"other.parent_frame ({other.parent_frame})"
            )
        
        # Compose rotations
        self_rot = R.from_quat(self.rotation)
        other_rot = R.from_quat(other.rotation)
        composed_rot = self_rot * other_rot
        
        # Compose translations: t_composed = t_self + R_self * t_other
        composed_translation = self.translation + self_rot.apply(other.translation)
        
        return Transform(
            translation=composed_translation,
            rotation=composed_rot.as_quat(),
            parent_frame=self.parent_frame,
            child_frame=other.child_frame,
            is_static=self.is_static and other.is_static
        )
    
    def apply_to_pose(self, pose: StampedPose) -> StampedPose:
        """
        Apply this transform to a pose.
        
        The pose must be in the child_frame. Result will be in parent_frame.
        
        Args:
            pose: The pose to transform (must be in self.child_frame)
            
        Returns:
            Transformed pose in self.parent_frame
        """
        if pose.frame_id != self.child_frame:
            raise ValueError(
                f"Pose frame ({pose.frame_id}) doesn't match transform's "
                f"child_frame ({self.child_frame})"
            )
        
        # Transform position: p_parent = t + R * p_child
        self_rot = R.from_quat(self.rotation)
        new_position = self.translation + self_rot.apply(pose.position)
        
        # Transform orientation: q_parent = q_self * q_pose
        pose_rot = R.from_quat(pose.quaternion)
        new_rotation = self_rot * pose_rot
        
        return StampedPose(
            position=new_position,
            quaternion=new_rotation.as_quat(),
            frame_id=self.parent_frame
        )
    
    def __repr__(self) -> str:
        static_str = " (static)" if self.is_static else ""
        return (f"Transform({self.parent_frame} -> {self.child_frame}{static_str}, "
                f"t={self.translation.round(4).tolist()}, "
                f"q={self.rotation.round(4).tolist()})")
