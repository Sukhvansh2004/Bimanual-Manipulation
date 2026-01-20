#!/usr/bin/env python3
"""
Pytest-based test suite for the transforms module.
Tests StampedPose, Transform, and TransformManager.
"""

import pytest
import numpy as np
from scipy.spatial.transform import Rotation as R
from pydantic import ValidationError
from unittest.mock import Mock, MagicMock

from robosuite_franka_manipulation.transforms import (
    StampedPose,
    Transform,
    TransformManager,
)
from robosuite_franka_manipulation import constants


class TestStampedPose:
    """Test StampedPose Pydantic model functionality."""

    def test_creation_basic(self):
        """Test basic StampedPose creation using aliases."""
        pose = StampedPose(
            pos=[1.0, 2.0, 3.0],
            quat=[0, 0, 0, 1],
            frame_id="world"
        )
        
        assert pose.frame_id == "world"
        np.testing.assert_array_almost_equal(pose.position, [1.0, 2.0, 3.0])
        np.testing.assert_array_almost_equal(pose.quaternion, [0, 0, 0, 1])

    def test_creation_properties(self):
        """Test creation using property names (allowed by alias configuration)."""
        pose = StampedPose(
            position=[1.0, 2.0, 3.0],
            quaternion=[0, 0, 0, 1],
            frame_id="world"
        )
        np.testing.assert_array_almost_equal(pose.pos, [1.0, 2.0, 3.0])

    def test_position_as_numpy_array(self):
        """Test that position is converted to numpy array."""
        pose = StampedPose(
            pos=[1, 2, 3],
            quat=[0, 0, 0, 1],
            frame_id="test"
        )
        
        assert isinstance(pose.position, np.ndarray)
        assert pose.position.dtype == np.float64

    def test_quaternion_normalization(self):
        """Test that quaternion is normalized on creation."""
        pose = StampedPose(
            pos=[0, 0, 0],
            quat=[0, 0, 0, 2],  # Not unit length
            frame_id="test"
        )
        
        norm = np.linalg.norm(pose.quaternion)
        assert abs(norm - 1.0) < 1e-10

    def test_invalid_position_shape(self):
        """Test that invalid position shape raises error."""
        with pytest.raises(ValidationError):
            StampedPose(
                pos=[1, 2],  # Only 2D
                quat=[0, 0, 0, 1],
                frame_id="test"
            )

    def test_invalid_quaternion_shape(self):
        """Test that invalid quaternion shape raises error."""
        with pytest.raises(ValidationError):
            StampedPose(
                pos=[1, 2, 3],
                quat=[0, 0, 1],  # Only 3D
                frame_id="test"
            )

    def test_from_dict(self):
        """Test creating StampedPose from dictionary."""
        data = {
            'pos': [1.0, 2.0, 3.0],
            'quat': [0, 0, 0, 1],
            'frame_id': 'world'
        }
        
        pose = StampedPose.from_dict(data)
        
        assert pose.frame_id == 'world'
        np.testing.assert_array_almost_equal(pose.position, [1.0, 2.0, 3.0])

    def test_from_dict_alternative_keys(self):
        """Test from_dict with alternative key names."""
        data = {
            'position': [1.0, 2.0, 3.0],
            'quaternion': [0, 0, 0, 1],
        }
        
        pose = StampedPose.from_dict(data, frame_id='custom')
        
        assert pose.frame_id == 'custom'
        np.testing.assert_array_almost_equal(pose.position, [1.0, 2.0, 3.0])

    def test_to_dict(self):
        """Test converting StampedPose to dictionary."""
        pose = StampedPose(
            pos=[1.0, 2.0, 3.0],
            quat=[0, 0, 0, 1],
            frame_id="world"
        )
        
        d = pose.to_dict()
        
        assert 'pos' in d
        assert 'quat' in d
        assert 'frame_id' in d
        assert d['frame_id'] == 'world'

    def test_from_matrix(self):
        """Test creating StampedPose from 4x4 transformation matrix."""
        matrix = np.eye(4)
        matrix[:3, 3] = [1.0, 2.0, 3.0]
        matrix[:3, :3] = R.from_euler('z', 90, degrees=True).as_matrix()
        
        pose = StampedPose.from_matrix(matrix, "world")
        
        assert pose.frame_id == "world"
        np.testing.assert_array_almost_equal(pose.position, [1.0, 2.0, 3.0])

    def test_copy(self):
        """Test that copy creates independent copy."""
        original = StampedPose(
            pos=[1, 2, 3],
            quat=[0, 0, 0, 1],
            frame_id="world"
        )
        
        copied = original.copy()
        
        # Modify original property (uses setter)
        original.position = [999, 999, 999]
        
        # Copy should be unchanged
        assert copied.position[0] == 1.0


class TestTransform:
    """Test Transform class functionality."""

    def test_creation_basic(self):
        """Test basic Transform creation."""
        transform = Transform(
            translation=[1, 0, 0],
            rotation=[0, 0, 0, 1],
            parent_frame="world",
            child_frame="robot_base"
        )
        
        assert transform.parent_frame == "world"
        assert transform.child_frame == "robot_base"
        np.testing.assert_array_almost_equal(transform.translation, [1, 0, 0])

    def test_apply_to_pose(self):
        """Test applying transform to a pose."""
        transform = Transform(
            translation=[1, 0, 0],
            rotation=[0, 0, 0, 1],
            parent_frame="world",
            child_frame="local"
        )
        
        pose = StampedPose(
            pos=[0, 0, 0],
            quat=[0, 0, 0, 1],
            frame_id="local"
        )
        
        result = transform.apply_to_pose(pose)
        
        assert result.frame_id == "world"
        np.testing.assert_array_almost_equal(result.position, [1, 0, 0])


class TestTransformManager:
    """Test TransformManager functionality."""

    @pytest.fixture
    def tf_manager(self):
        """Fixture to create a TransformManager without environment."""
        return TransformManager(env=None, auto_discover=False)

    def test_static_transform_graph(self, tf_manager):
        """Test building and traversing a static transform graph."""
        # A -> B: translate [1, 0, 0]
        tf_manager.add_static_transform(Transform(
            translation=[1, 0, 0], rotation=[0, 0, 0, 1],
            parent_frame="A", child_frame="B"
        ))
        
        # B -> C: translate [0, 1, 0]
        tf_manager.add_static_transform(Transform(
            translation=[0, 1, 0], rotation=[0, 0, 0, 1],
            parent_frame="B", child_frame="C"
        ))
        
        # Test A -> C (direct composition)
        # Expected: [1, 1, 0]
        t_AC = tf_manager.get_transform("A", "C")
        np.testing.assert_array_almost_equal(t_AC.translation, [1, 1, 0])
        
        # Test C -> A (inverse traversal)
        # Expected: [-1, -1, 0]
        t_CA = tf_manager.get_transform("C", "A")
        np.testing.assert_array_almost_equal(t_CA.translation, [-1, -1, 0])

    def test_transform_pose(self, tf_manager):
        """Test transforming a pose across frames."""
        tf_manager.add_static_transform(Transform(
            translation=[1, 0, 0], rotation=[0, 0, 0, 1],
            parent_frame="world", child_frame="table"
        ))
        
        # Pose on table: [0.5, 0.5, 0] relative to table
        pose_table = StampedPose(
            pos=[0.5, 0.5, 0], quat=[0, 0, 0, 1], frame_id="table"
        )
        
        # Transform to world
        pose_world = tf_manager.transform_pose(pose_table, "world")
        
        # Expected: [1.5, 0.5, 0] in world
        np.testing.assert_array_almost_equal(pose_world.position, [1.5, 0.5, 0])
        assert pose_world.frame_id == "world"

    def test_unconnected_frames(self, tf_manager):
        """Test error handling for unconnected frames."""
        tf_manager.add_static_transform(Transform(
            translation=[0,0,0], rotation=[0,0,0,1],
            parent_frame="A", child_frame="B"
        ))
        
        tf_manager.add_static_transform(Transform(
            translation=[0,0,0], rotation=[0,0,0,1],
            parent_frame="C", child_frame="D"
        ))
        
        with pytest.raises(ValueError, match="No transform path"):
            tf_manager.get_transform("A", "D")

    def test_object_frames(self, tf_manager):
        """Test adding and updating object frames."""
        # Initial object pose in world
        pose1 = StampedPose(pos=[1, 0, 0], quat=[0, 0, 0, 1], frame_id="world")
        frame_name = tf_manager.add_object_frame("cube", pose1)
        assert frame_name == "object_cube"
        
        # Check transform world -> object_cube (inverse of pose)
        t_world_obj = tf_manager.get_transform("world", "object_cube")
        np.testing.assert_array_almost_equal(t_world_obj.translation, [1, 0, 0])
        
        # Update object pose
        pose2 = StampedPose(pos=[2, 0, 0], quat=[0, 0, 0, 1], frame_id="world")
        tf_manager.update_object_frame("cube", pose2)
        
        t_world_obj_new = tf_manager.get_transform("world", "object_cube")
        np.testing.assert_array_almost_equal(t_world_obj_new.translation, [2, 0, 0])

    def test_dynamic_transform_mock(self):
        """Test dynamic transform with mocked environment."""
        # Mock environment structure
        mock_env = Mock()
        mock_env.robots = []
        mock_env.sim = Mock()
        mock_env.sim.model.ncam = 0
        
        # Create manager with mocked env, auto_discover=False to manually register
        tf_manager = TransformManager(env=mock_env, auto_discover=False)
        
        # Manually register a dynamic provider
        # e.g., robot_base -> eef
        def dynamic_provider():
            return Transform(
                translation=[0.5, 0, 0], 
                rotation=[0, 0, 0, 1],
                parent_frame="robot_base",
                child_frame="eef"
            )
        
        # Inject provider into private dict (simulating _register_eef_provider logic)
        tf_manager._dynamic_frame_providers["eef"] = dynamic_provider
        tf_manager._frame_parents["eef"] = "robot_base"
        tf_manager._known_frames.add("eef")
        tf_manager._known_frames.add("robot_base") # Ensure parent known
        
        # Query transform
        t = tf_manager.get_transform("robot_base", "eef")
        np.testing.assert_array_almost_equal(t.translation, [0.5, 0, 0])
        assert t.is_static == False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
