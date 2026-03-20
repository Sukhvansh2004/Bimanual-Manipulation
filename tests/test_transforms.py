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

from bimanual_manipulation.transforms import (
    StampedPose,
    Transform,
    TransformManager,
)
from bimanual_manipulation import constants


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
        
        pose = StampedPose(**data)
        
        assert pose.frame_id == 'world'
        np.testing.assert_array_almost_equal(pose.position, [1.0, 2.0, 3.0])

    def test_from_dict_alternative_keys(self):
        """Test from_dict with alternative key names."""
        data = {
            'position': [1.0, 2.0, 3.0],
            'quaternion': [0, 0, 0, 1],
        }
        
        pose = StampedPose(**data, frame_id='custom')
        
        assert pose.frame_id == 'custom'
        np.testing.assert_array_almost_equal(pose.position, [1.0, 2.0, 3.0])

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


# ═══════════════════════════════════════════════════════════════════════════════
# Additional StampedPose tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestStampedPoseExtended:
    """Extended tests for StampedPose."""

    def test_to_matrix_identity(self):
        """Identity pose should give identity matrix."""
        pose = StampedPose(pos=[0, 0, 0], quat=[0, 0, 0, 1], frame_id="world")
        M = pose.to_matrix()
        np.testing.assert_array_almost_equal(M, np.eye(4))

    def test_to_matrix_translation_only(self):
        """Pure translation should appear in last column."""
        pose = StampedPose(pos=[1, 2, 3], quat=[0, 0, 0, 1], frame_id="world")
        M = pose.to_matrix()
        np.testing.assert_array_almost_equal(M[:3, 3], [1, 2, 3])
        np.testing.assert_array_almost_equal(M[:3, :3], np.eye(3))

    def test_from_matrix_to_matrix_roundtrip(self):
        """from_matrix(M).to_matrix() should recover M."""
        M = np.eye(4)
        M[:3, :3] = R.from_euler('xyz', [30, 45, 60], degrees=True).as_matrix()
        M[:3, 3] = [1.5, -2.5, 0.7]
        pose = StampedPose.from_matrix(M, "test_frame")
        M_recovered = pose.to_matrix()
        np.testing.assert_array_almost_equal(M_recovered, M, decimal=10)

    def test_from_matrix_invalid_shape(self):
        """Non-4x4 matrix should raise."""
        with pytest.raises(ValueError, match="4x4"):
            StampedPose.from_matrix(np.eye(3), "world")

    def test_immutable_arrays(self):
        """Position and quaternion arrays should be immutable (writeable=False)."""
        pose = StampedPose(pos=[1, 2, 3], quat=[0, 0, 0, 1], frame_id="world")
        with pytest.raises((ValueError, TypeError)):
            pose.position[0] = 999.0
        with pytest.raises((ValueError, TypeError)):
            pose.quaternion[0] = 999.0

    def test_timestamp_auto_set(self):
        """Timestamp should be automatically populated."""
        pose = StampedPose(pos=[0, 0, 0], quat=[0, 0, 0, 1])
        assert pose.timestamp > 0

    def test_default_frame_id(self):
        """Default frame_id should be WORLD_FRAME."""
        pose = StampedPose(pos=[0, 0, 0], quat=[0, 0, 0, 1])
        assert pose.frame_id == constants.WORLD_FRAME

    def test_zero_quaternion_handling(self):
        """All-zero quaternion (degenerate) should not crash - normalization guard."""
        # Norm is 0 → normalizer check (norm > 1e-10) returns as-is
        pose = StampedPose(pos=[0, 0, 0], quat=[0, 0, 0, 0])
        assert pose is not None

    def test_serialization_roundtrip(self):
        """model_dump / re-creation should roundtrip."""
        original = StampedPose(
            pos=[1.5, -2.3, 0.7],
            quat=[0.1, 0.2, 0.3, 0.927],
            frame_id="camera0",
        )
        data = original.model_dump()
        restored = StampedPose(**data)
        np.testing.assert_array_almost_equal(restored.position, original.position, decimal=4)
        np.testing.assert_array_almost_equal(restored.quaternion, original.quaternion, decimal=4)
        assert restored.frame_id == original.frame_id

    def test_negative_quaternion_equiv(self):
        """Opposite-sign quaternions represent the same rotation; normalization should keep sign consistent."""
        q1 = np.array([0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)])
        q2 = -q1
        p1 = StampedPose(pos=[0, 0, 0], quat=q1.tolist(), frame_id="w")
        p2 = StampedPose(pos=[0, 0, 0], quat=q2.tolist(), frame_id="w")
        M1 = p1.to_matrix()
        M2 = p2.to_matrix()
        np.testing.assert_array_almost_equal(M1[:3, :3], M2[:3, :3])


# ═══════════════════════════════════════════════════════════════════════════════
# Additional Transform tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestTransformExtended:
    """Extended tests for Transform dataclass."""

    def test_identity(self):
        """Identity transform should have zero translation and identity rotation."""
        t = Transform.identity("A", "B")
        np.testing.assert_array_equal(t.translation, [0, 0, 0])
        np.testing.assert_array_almost_equal(t.rotation, [0, 0, 0, 1])
        assert t.parent_frame == "A"
        assert t.child_frame == "B"
        assert t.is_static is True

    def test_inverse_translation(self):
        """Inverse of pure translation should be negated translation."""
        t = Transform(
            translation=[1, 2, 3], rotation=[0, 0, 0, 1],
            parent_frame="A", child_frame="B"
        )
        inv = t.inverse()
        np.testing.assert_array_almost_equal(inv.translation, [-1, -2, -3])
        assert inv.parent_frame == "B"
        assert inv.child_frame == "A"

    def test_inverse_rotation(self):
        """Inverse of rotation should give inverse rotation."""
        rot = R.from_euler('z', 90, degrees=True).as_quat()
        t = Transform(
            translation=[0, 0, 0], rotation=rot,
            parent_frame="W", child_frame="R"
        )
        inv = t.inverse()
        # Compose should give identity
        composed = t.compose(inv)
        np.testing.assert_array_almost_equal(composed.translation, [0, 0, 0])

    def test_double_inverse(self):
        """T.inverse().inverse() should recover T."""
        t = Transform(
            translation=[1, -2, 3],
            rotation=R.from_euler('xyz', [10, 20, 30], degrees=True).as_quat(),
            parent_frame="A", child_frame="B"
        )
        tt = t.inverse().inverse()
        np.testing.assert_array_almost_equal(tt.translation, t.translation, decimal=10)
        # Quaternion may differ by sign but rotation should match
        M_orig = R.from_quat(t.rotation).as_matrix()
        M_recov = R.from_quat(tt.rotation).as_matrix()
        np.testing.assert_array_almost_equal(M_orig, M_recov, decimal=10)

    def test_compose_translations(self):
        """Composing two pure translations should add them."""
        t1 = Transform(
            translation=[1, 0, 0], rotation=[0, 0, 0, 1],
            parent_frame="A", child_frame="B"
        )
        t2 = Transform(
            translation=[0, 2, 0], rotation=[0, 0, 0, 1],
            parent_frame="B", child_frame="C"
        )
        composed = t1.compose(t2)
        np.testing.assert_array_almost_equal(composed.translation, [1, 2, 0])
        assert composed.parent_frame == "A"
        assert composed.child_frame == "C"

    def test_compose_frame_mismatch_raises(self):
        """Composing transforms with mismatched frames should raise."""
        t1 = Transform(translation=[0, 0, 0], rotation=[0, 0, 0, 1],
                       parent_frame="A", child_frame="B")
        t2 = Transform(translation=[0, 0, 0], rotation=[0, 0, 0, 1],
                       parent_frame="X", child_frame="Y")
        with pytest.raises(ValueError, match="Cannot compose"):
            t1.compose(t2)

    def test_from_matrix_roundtrip(self):
        """from_matrix(T.to_matrix()) should recover T."""
        t = Transform(
            translation=[1, -0.5, 2],
            rotation=R.from_euler('zyx', [45, 30, 15], degrees=True).as_quat(),
            parent_frame="W", child_frame="E"
        )
        M = t.to_matrix()
        t2 = Transform.from_matrix(M, "W", "E")
        np.testing.assert_array_almost_equal(t2.translation, t.translation, decimal=10)
        np.testing.assert_array_almost_equal(t2.to_matrix(), M, decimal=10)

    def test_from_matrix_invalid_shape(self):
        """Non-4x4 matrix should raise."""
        with pytest.raises(ValueError, match="4x4"):
            Transform.from_matrix(np.eye(3), "A", "B")

    def test_to_matrix_identity(self):
        """Identity transform should give identity matrix."""
        t = Transform.identity("A", "B")
        np.testing.assert_array_almost_equal(t.to_matrix(), np.eye(4))

    def test_quaternion_normalization(self):
        """Non-unit quaternion should be normalized."""
        t = Transform(
            translation=[0, 0, 0],
            rotation=[0, 0, 0, 2],
            parent_frame="A", child_frame="B"
        )
        np.testing.assert_allclose(np.linalg.norm(t.rotation), 1.0, atol=1e-10)

    def test_invalid_translation_shape(self):
        """Non-3D translation should raise."""
        with pytest.raises(ValueError, match="3D"):
            Transform(translation=[1, 2], rotation=[0, 0, 0, 1],
                      parent_frame="A", child_frame="B")

    def test_invalid_rotation_shape(self):
        """Non-4D rotation should raise."""
        with pytest.raises(ValueError, match="4D"):
            Transform(translation=[0, 0, 0], rotation=[0, 0, 1],
                      parent_frame="A", child_frame="B")

    def test_compose_rotation_then_translate(self):
        """Compose a rotation with a translation."""
        # First rotate 90 degrees about z, then translate 1 unit along x
        rot_90z = R.from_euler('z', 90, degrees=True).as_quat()
        t1 = Transform(translation=[0, 0, 0], rotation=rot_90z,
                       parent_frame="A", child_frame="B")
        t2 = Transform(translation=[1, 0, 0], rotation=[0, 0, 0, 1],
                       parent_frame="B", child_frame="C")
        composed = t1.compose(t2)
        # The translation [1,0,0] in B frame, after 90z rotation, becomes [0,1,0] in A frame
        np.testing.assert_array_almost_equal(composed.translation, [0, 1, 0], decimal=10)

