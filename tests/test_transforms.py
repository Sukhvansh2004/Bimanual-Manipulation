#!/usr/bin/env python3
"""
Pytest-based test suite for the transforms module.
Tests StampedPose, Transform, TransformManager, and frame utilities.
"""

import pytest
import numpy as np
from scipy.spatial.transform import Rotation as R

from robosuite_franka_manipulation.transforms import (
    StampedPose,
    Transform,
    TransformManager,
    detection_to_stamped_pose,
    grasp_to_stamped_pose,
    stamped_pose_to_dict,
    compute_approach_pose,
)
from robosuite_franka_manipulation import constants


class TestStampedPose:
    """Test StampedPose class functionality."""

    def test_creation_basic(self):
        """Test basic StampedPose creation."""
        pose = StampedPose(
            position=[1.0, 2.0, 3.0],
            quaternion=[0, 0, 0, 1],
            frame_id="world"
        )
        
        assert pose.frame_id == "world"
        np.testing.assert_array_almost_equal(pose.position, [1.0, 2.0, 3.0])
        np.testing.assert_array_almost_equal(pose.quaternion, [0, 0, 0, 1])

    def test_position_as_numpy_array(self):
        """Test that position is converted to numpy array."""
        pose = StampedPose(
            position=[1, 2, 3],
            quaternion=[0, 0, 0, 1],
            frame_id="test"
        )
        
        assert isinstance(pose.position, np.ndarray)
        assert pose.position.dtype == np.float64

    def test_quaternion_normalization(self):
        """Test that quaternion is normalized on creation."""
        # Non-normalized quaternion
        pose = StampedPose(
            position=[0, 0, 0],
            quaternion=[0, 0, 0, 2],  # Not unit length
            frame_id="test"
        )
        
        norm = np.linalg.norm(pose.quaternion)
        assert abs(norm - 1.0) < 1e-10

    def test_invalid_position_shape(self):
        """Test that invalid position shape raises error."""
        with pytest.raises(ValueError, match="Position must be 3D"):
            StampedPose(
                position=[1, 2],  # Only 2D
                quaternion=[0, 0, 0, 1],
                frame_id="test"
            )

    def test_invalid_quaternion_shape(self):
        """Test that invalid quaternion shape raises error."""
        with pytest.raises(ValueError, match="Quaternion must be 4D"):
            StampedPose(
                position=[1, 2, 3],
                quaternion=[0, 0, 1],  # Only 3D
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
            position=[1.0, 2.0, 3.0],
            quaternion=[0, 0, 0, 1],
            frame_id="world"
        )
        
        d = pose.to_dict()
        
        assert 'pos' in d
        assert 'quat' in d
        assert 'frame_id' in d
        assert d['frame_id'] == 'world'

    def test_from_matrix(self):
        """Test creating StampedPose from 4x4 transformation matrix."""
        # Create a known transformation matrix
        matrix = np.eye(4)
        matrix[:3, 3] = [1.0, 2.0, 3.0]  # Translation
        # 90 degree rotation around Z
        matrix[:3, :3] = R.from_euler('z', 90, degrees=True).as_matrix()
        
        pose = StampedPose.from_matrix(matrix, "world")
        
        assert pose.frame_id == "world"
        np.testing.assert_array_almost_equal(pose.position, [1.0, 2.0, 3.0])
        
        # Verify rotation
        rot = R.from_quat(pose.quaternion)
        expected_rot = R.from_euler('z', 90, degrees=True)
        angle_diff = (rot * expected_rot.inv()).magnitude()
        assert angle_diff < 1e-6

    def test_to_matrix(self):
        """Test converting StampedPose to 4x4 matrix."""
        pose = StampedPose(
            position=[1.0, 2.0, 3.0],
            quaternion=R.from_euler('z', 90, degrees=True).as_quat(),
            frame_id="world"
        )
        
        matrix = pose.to_matrix()
        
        assert matrix.shape == (4, 4)
        np.testing.assert_array_almost_equal(matrix[:3, 3], [1.0, 2.0, 3.0])
        assert matrix[3, 3] == 1.0
        np.testing.assert_array_almost_equal(matrix[3, :3], [0, 0, 0])

    def test_matrix_roundtrip(self):
        """Test that matrix conversion is reversible."""
        original = StampedPose(
            position=[1.5, -2.3, 0.7],
            quaternion=R.from_euler('xyz', [30, 45, 60], degrees=True).as_quat(),
            frame_id="test"
        )
        
        matrix = original.to_matrix()
        recovered = StampedPose.from_matrix(matrix, "test")
        
        np.testing.assert_array_almost_equal(original.position, recovered.position)
        # Quaternions may differ by sign (q and -q represent same rotation)
        dot = abs(np.dot(original.quaternion, recovered.quaternion))
        assert abs(dot - 1.0) < 1e-6

    def test_copy(self):
        """Test that copy creates independent copy."""
        original = StampedPose(
            position=[1, 2, 3],
            quaternion=[0, 0, 0, 1],
            frame_id="world"
        )
        
        copied = original.copy()
        
        # Modify original (position is np.ndarray after __post_init__)
        original.position[0] = 999
        
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

    def test_identity(self):
        """Test identity transform creation."""
        transform = Transform.identity("world", "child")
        
        np.testing.assert_array_almost_equal(transform.translation, [0, 0, 0])
        np.testing.assert_array_almost_equal(transform.rotation, [0, 0, 0, 1])
        assert transform.is_static == True

    def test_from_matrix(self):
        """Test creating Transform from 4x4 matrix."""
        matrix = np.eye(4)
        matrix[:3, 3] = [1, 2, 3]
        matrix[:3, :3] = R.from_euler('z', 45, degrees=True).as_matrix()
        
        transform = Transform.from_matrix(matrix, "parent", "child")
        
        np.testing.assert_array_almost_equal(transform.translation, [1, 2, 3])
        assert transform.parent_frame == "parent"
        assert transform.child_frame == "child"

    def test_to_matrix(self):
        """Test converting Transform to 4x4 matrix."""
        transform = Transform(
            translation=[1, 2, 3],
            rotation=R.from_euler('x', 90, degrees=True).as_quat(),
            parent_frame="a",
            child_frame="b"
        )
        
        matrix = transform.to_matrix()
        
        assert matrix.shape == (4, 4)
        np.testing.assert_array_almost_equal(matrix[:3, 3], [1, 2, 3])
        assert abs(matrix[3, 3] - 1.0) < 1e-10

    def test_inverse(self):
        """Test transform inversion."""
        # Translation only
        transform = Transform(
            translation=[1, 0, 0],
            rotation=[0, 0, 0, 1],
            parent_frame="A",
            child_frame="B"
        )
        
        inverse = transform.inverse()
        
        assert inverse.parent_frame == "B"
        assert inverse.child_frame == "A"
        np.testing.assert_array_almost_equal(inverse.translation, [-1, 0, 0])

    def test_inverse_with_rotation(self):
        """Test transform inversion with rotation."""
        # 90 degree rotation + translation
        transform = Transform(
            translation=[1, 0, 0],
            rotation=R.from_euler('z', 90, degrees=True).as_quat(),
            parent_frame="A",
            child_frame="B"
        )
        
        inverse = transform.inverse()
        
        # Composing should give identity
        composed_matrix = transform.to_matrix() @ inverse.to_matrix()
        np.testing.assert_array_almost_equal(composed_matrix, np.eye(4), decimal=6)

    def test_compose_translation_only(self):
        """Test composing two translation-only transforms."""
        t1 = Transform(
            translation=[1, 0, 0],
            rotation=[0, 0, 0, 1],
            parent_frame="A",
            child_frame="B"
        )
        t2 = Transform(
            translation=[0, 1, 0],
            rotation=[0, 0, 0, 1],
            parent_frame="B",
            child_frame="C"
        )
        
        composed = t1.compose(t2)
        
        assert composed.parent_frame == "A"
        assert composed.child_frame == "C"
        np.testing.assert_array_almost_equal(composed.translation, [1, 1, 0])

    def test_compose_with_rotation(self):
        """Test composing transforms with rotation."""
        # T1: rotate 90 deg around Z
        t1 = Transform(
            translation=[0, 0, 0],
            rotation=R.from_euler('z', 90, degrees=True).as_quat(),
            parent_frame="A",
            child_frame="B"
        )
        # T2: translate along X (which is Y in A's frame after rotation)
        t2 = Transform(
            translation=[1, 0, 0],
            rotation=[0, 0, 0, 1],
            parent_frame="B",
            child_frame="C"
        )
        
        composed = t1.compose(t2)
        
        # Translation in A's frame should be [0, 1, 0] (rotated)
        np.testing.assert_array_almost_equal(composed.translation, [0, 1, 0], decimal=6)

    def test_compose_frame_mismatch_error(self):
        """Test that composing with mismatched frames raises error."""
        t1 = Transform(
            translation=[1, 0, 0],
            rotation=[0, 0, 0, 1],
            parent_frame="A",
            child_frame="B"
        )
        t2 = Transform(
            translation=[0, 1, 0],
            rotation=[0, 0, 0, 1],
            parent_frame="C",  # Doesn't match t1.child_frame
            child_frame="D"
        )
        
        with pytest.raises(ValueError, match="Cannot compose"):
            t1.compose(t2)

    def test_apply_to_pose(self):
        """Test applying transform to a pose."""
        # Transform: translate by [1, 0, 0]
        transform = Transform(
            translation=[1, 0, 0],
            rotation=[0, 0, 0, 1],
            parent_frame="world",
            child_frame="local"
        )
        
        # Pose at origin in local frame
        pose = StampedPose(
            position=[0, 0, 0],
            quaternion=[0, 0, 0, 1],
            frame_id="local"
        )
        
        result = transform.apply_to_pose(pose)
        
        assert result.frame_id == "world"
        np.testing.assert_array_almost_equal(result.position, [1, 0, 0])

    def test_apply_to_pose_with_rotation(self):
        """Test applying transform with rotation to a pose."""
        # Transform: rotate 90 deg around Z, then translate
        transform = Transform(
            translation=[1, 0, 0],
            rotation=R.from_euler('z', 90, degrees=True).as_quat(),
            parent_frame="world",
            child_frame="local"
        )
        
        # Pose offset in local frame
        pose = StampedPose(
            position=[1, 0, 0],  # 1 unit along local X
            quaternion=[0, 0, 0, 1],
            frame_id="local"
        )
        
        result = transform.apply_to_pose(pose)
        
        # Local X becomes world Y after 90 deg rotation
        # Plus translation [1, 0, 0]
        np.testing.assert_array_almost_equal(result.position, [1, 1, 0], decimal=6)

    def test_apply_to_pose_frame_mismatch_error(self):
        """Test that applying to pose with wrong frame raises error."""
        transform = Transform(
            translation=[1, 0, 0],
            rotation=[0, 0, 0, 1],
            parent_frame="world",
            child_frame="local"
        )
        
        pose = StampedPose(
            position=[0, 0, 0],
            quaternion=[0, 0, 0, 1],
            frame_id="wrong_frame"  # Doesn't match child_frame
        )
        
        with pytest.raises(ValueError, match="doesn't match"):
            transform.apply_to_pose(pose)


class TestTransformChains:
    """Test complex transform chains and operations."""

    def test_full_transform_chain(self):
        """Test a chain of transforms: world -> base -> eef."""
        # World to robot base
        world_to_base = Transform(
            translation=[1, 0, 0],
            rotation=[0, 0, 0, 1],
            parent_frame="world",
            child_frame="robot_base"
        )
        
        # Robot base to end-effector
        base_to_eef = Transform(
            translation=[0.5, 0, 0.3],
            rotation=R.from_euler('y', 45, degrees=True).as_quat(),
            parent_frame="robot_base",
            child_frame="eef"
        )
        
        # Compose to get world to eef
        world_to_eef = world_to_base.compose(base_to_eef)
        
        assert world_to_eef.parent_frame == "world"
        assert world_to_eef.child_frame == "eef"
        
        # Test by applying to a pose at eef origin
        pose_eef = StampedPose(
            position=[0, 0, 0],
            quaternion=[0, 0, 0, 1],
            frame_id="eef"
        )
        
        pose_world = world_to_eef.apply_to_pose(pose_eef)
        
        assert pose_world.frame_id == "world"
        # Should be [1 + 0.5, 0, 0.3] = [1.5, 0, 0.3]
        np.testing.assert_array_almost_equal(pose_world.position, [1.5, 0, 0.3], decimal=6)

    def test_inverse_chain(self):
        """Test that inverse of composed transform works correctly."""
        t1 = Transform(
            translation=[1, 2, 3],
            rotation=R.from_euler('xyz', [10, 20, 30], degrees=True).as_quat(),
            parent_frame="A",
            child_frame="B"
        )
        t2 = Transform(
            translation=[-0.5, 0.3, 0.1],
            rotation=R.from_euler('z', 45, degrees=True).as_quat(),
            parent_frame="B",
            child_frame="C"
        )
        
        composed = t1.compose(t2)
        composed_inv = composed.inverse()
        
        # Composed * Composed_inv should be identity
        identity = composed.compose(composed_inv)
        np.testing.assert_array_almost_equal(identity.translation, [0, 0, 0], decimal=6)


class TestFrameUtilities:
    """Test frame utility functions."""

    def test_detection_to_stamped_pose(self):
        """Test converting detection dict to StampedPose."""
        detection = {
            'name': 'cube',
            'pose': {
                'pos': [1, 2, 3],
                'quat': [0, 0, 0, 1]
            }
        }
        
        pose = detection_to_stamped_pose(detection, "world")
        
        assert pose.frame_id == "world"
        np.testing.assert_array_almost_equal(pose.position, [1, 2, 3])

    def test_grasp_to_stamped_pose(self):
        """Test converting grasp dict to StampedPose."""
        grasp = {
            'pos': [0.5, 0.1, 0.3],
            'quat': R.from_euler('z', 45, degrees=True).as_quat().tolist(),
            'score': 0.95
        }
        
        pose = grasp_to_stamped_pose(grasp, "world")
        
        assert pose.frame_id == "world"
        np.testing.assert_array_almost_equal(pose.position, [0.5, 0.1, 0.3])

    def test_stamped_pose_to_dict(self):
        """Test converting StampedPose back to dict."""
        pose = StampedPose(
            position=[1, 2, 3],
            quaternion=[0, 0, 0, 1],
            frame_id="world"
        )
        
        d = stamped_pose_to_dict(pose)
        
        assert 'pos' in d
        assert 'quat' in d
        assert 'frame_id' in d
        assert d['frame_id'] == 'world'

    def test_compute_approach_pose_z_axis(self):
        """Test computing approach pose along Z axis."""
        grasp_pose = StampedPose(
            position=[1, 0, 0.5],
            quaternion=[0, 0, 0, 1],  # Identity rotation
            frame_id="world"
        )
        
        approach = compute_approach_pose(grasp_pose, approach_distance=0.1, approach_axis='z')
        
        assert approach.frame_id == "world"
        # Should be 0.1m behind grasp along Z
        np.testing.assert_array_almost_equal(approach.position, [1, 0, 0.4], decimal=6)

    def test_compute_approach_pose_rotated(self):
        """Test computing approach pose with rotated grasp."""
        # Grasp pointing along X axis (90 deg rotation around Y)
        # After this rotation, local Z points along world X
        grasp_pose = StampedPose(
            position=[1, 0, 0],
            quaternion=R.from_euler('y', 90, degrees=True).as_quat(),
            frame_id="world"
        )
        
        approach = compute_approach_pose(grasp_pose, approach_distance=0.1, approach_axis='z')
        
        # After Y rotation by 90 deg, local Z is world X
        # Approach is along NEGATIVE local Z, which is NEGATIVE world X
        # So approach position = [1, 0, 0] - 0.1 * [1, 0, 0] = [0.9, 0, 0]
        np.testing.assert_array_almost_equal(approach.position, [0.9, 0, 0], decimal=6)


class TestGraspTransformations:
    """Test grasp-specific transformation scenarios."""

    def test_grasp_object_local_to_world(self):
        """Test transforming grasp from object-local frame to world frame."""
        # Object at position [1, 0, 0.5] with no rotation
        object_pose = StampedPose(
            position=[1, 0, 0.5],
            quaternion=[0, 0, 0, 1],
            frame_id="world"
        )
        
        # Grasp in object-local frame (above object center)
        grasp_local = StampedPose(
            position=[0, 0, 0.1],  # 0.1m above object origin
            quaternion=[0, 0, 0, 1],
            frame_id="object_local"
        )
        
        # Transform from object-local to world
        object_to_world = Transform(
            translation=object_pose.position,
            rotation=object_pose.quaternion,
            parent_frame="world",
            child_frame="object_local"
        )
        
        grasp_world = object_to_world.apply_to_pose(grasp_local)
        
        assert grasp_world.frame_id == "world"
        # Expected: object_pos + local_offset = [1, 0, 0.5] + [0, 0, 0.1] = [1, 0, 0.6]
        np.testing.assert_array_almost_equal(grasp_world.position, [1, 0, 0.6], decimal=6)

    def test_grasp_with_rotated_object(self):
        """Test transforming grasp when object is rotated."""
        # Object rotated 90 degrees around Z
        object_pose = StampedPose(
            position=[1, 0, 0],
            quaternion=R.from_euler('z', 90, degrees=True).as_quat(),
            frame_id="world"
        )
        
        # Grasp offset in object's local X direction
        grasp_local = StampedPose(
            position=[0.1, 0, 0],  # 0.1m along object's X
            quaternion=[0, 0, 0, 1],
            frame_id="object_local"
        )
        
        object_to_world = Transform(
            translation=object_pose.position,
            rotation=object_pose.quaternion,
            parent_frame="world",
            child_frame="object_local"
        )
        
        grasp_world = object_to_world.apply_to_pose(grasp_local)
        
        # Object's local X is world Y after 90 deg rotation
        # Expected: [1, 0, 0] + rotation([0.1, 0, 0]) = [1, 0.1, 0]
        np.testing.assert_array_almost_equal(grasp_world.position, [1, 0.1, 0], decimal=6)

    def test_world_to_robot_base_transform(self):
        """Test transforming from world frame to robot base frame."""
        # Robot base at [0.5, 0, 0] in world, no rotation
        world_to_base = Transform(
            translation=[0.5, 0, 0],
            rotation=[0, 0, 0, 1],
            parent_frame="world",
            child_frame="robot_base"
        )
        
        # Grasp in world frame
        grasp_world = StampedPose(
            position=[1.0, 0, 0.3],
            quaternion=[0, 0, 0, 1],
            frame_id="world"
        )
        
        # To transform world -> base, we need inverse
        base_to_world_inv = world_to_base.inverse()
        
        # Create pose in world frame, apply inverse to get in base frame
        # Wait - apply_to_pose expects pose in child_frame
        # So we need: base -> world transform applied to pose in "base" frame gives world
        # We want: world -> base, so pose in world, result in base
        # This requires the inverse transform
        
        # Actually: base_to_world_inv goes from "world" to "robot_base"
        grasp_base = base_to_world_inv.apply_to_pose(grasp_world)
        
        assert grasp_base.frame_id == "robot_base"
        # Expected: [1.0, 0, 0.3] - [0.5, 0, 0] = [0.5, 0, 0.3]
        np.testing.assert_array_almost_equal(grasp_base.position, [0.5, 0, 0.3], decimal=6)


class TestQuaternionConventions:
    """Test that quaternion conventions are consistent (scipy: x, y, z, w)."""

    def test_quaternion_format_identity(self):
        """Test that identity quaternion is [0, 0, 0, 1]."""
        pose = StampedPose(
            position=[0, 0, 0],
            quaternion=[0, 0, 0, 1],
            frame_id="test"
        )
        
        rot = R.from_quat(pose.quaternion)
        # Should be identity (no rotation)
        np.testing.assert_array_almost_equal(rot.as_matrix(), np.eye(3))

    def test_quaternion_90_deg_z_rotation(self):
        """Test 90 degree rotation around Z axis."""
        quat = R.from_euler('z', 90, degrees=True).as_quat()
        
        # Verify format is [x, y, z, w]
        # For 90 deg around Z: x=0, y=0, z=sin(45)≈0.707, w=cos(45)≈0.707
        assert abs(quat[0]) < 1e-6  # x ≈ 0
        assert abs(quat[1]) < 1e-6  # y ≈ 0
        assert abs(quat[2] - np.sin(np.pi/4)) < 1e-6  # z ≈ 0.707
        assert abs(quat[3] - np.cos(np.pi/4)) < 1e-6  # w ≈ 0.707

    def test_quaternion_composition_in_transform(self):
        """Test that quaternion composition in Transform matches scipy."""
        # Two 45-degree rotations should give 90-degree rotation
        q1 = R.from_euler('z', 45, degrees=True).as_quat()
        q2 = R.from_euler('z', 45, degrees=True).as_quat()
        
        t1 = Transform(
            translation=[0, 0, 0],
            rotation=q1,
            parent_frame="A",
            child_frame="B"
        )
        t2 = Transform(
            translation=[0, 0, 0],
            rotation=q2,
            parent_frame="B",
            child_frame="C"
        )
        
        composed = t1.compose(t2)
        
        # Result should be 90-degree rotation
        result_rot = R.from_quat(composed.rotation)
        expected_rot = R.from_euler('z', 90, degrees=True)
        
        # Compare rotation matrices
        np.testing.assert_array_almost_equal(
            result_rot.as_matrix(),
            expected_rot.as_matrix(),
            decimal=6
        )


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_translation_transform(self):
        """Test transform with zero translation."""
        transform = Transform(
            translation=[0, 0, 0],
            rotation=R.from_euler('z', 45, degrees=True).as_quat(),
            parent_frame="A",
            child_frame="B"
        )
        
        pose = StampedPose(
            position=[1, 0, 0],
            quaternion=[0, 0, 0, 1],
            frame_id="B"
        )
        
        result = transform.apply_to_pose(pose)
        
        # Should just rotate the position
        expected_pos = R.from_euler('z', 45, degrees=True).apply([1, 0, 0])
        np.testing.assert_array_almost_equal(result.position, expected_pos, decimal=6)

    def test_identity_transform_chain(self):
        """Test that identity transforms don't change poses."""
        t1 = Transform.identity("A", "B")
        t2 = Transform.identity("B", "C")
        
        composed = t1.compose(t2)
        
        pose = StampedPose(
            position=[1, 2, 3],
            quaternion=R.from_euler('xyz', [10, 20, 30], degrees=True).as_quat(),
            frame_id="C"
        )
        
        result = composed.apply_to_pose(pose)
        
        np.testing.assert_array_almost_equal(result.position, pose.position)
        np.testing.assert_array_almost_equal(result.quaternion, pose.quaternion)

    def test_very_small_rotation(self):
        """Test transform with very small rotation."""
        small_angle = 1e-10  # radians
        transform = Transform(
            translation=[1, 0, 0],
            rotation=R.from_euler('z', small_angle).as_quat(),
            parent_frame="A",
            child_frame="B"
        )
        
        # Should be essentially a pure translation
        pose = StampedPose(
            position=[0, 0, 0],
            quaternion=[0, 0, 0, 1],
            frame_id="B"
        )
        
        result = transform.apply_to_pose(pose)
        
        np.testing.assert_array_almost_equal(result.position, [1, 0, 0], decimal=6)

    def test_large_translation(self):
        """Test transform with large translation values."""
        transform = Transform(
            translation=[1000, 2000, 3000],
            rotation=[0, 0, 0, 1],
            parent_frame="A",
            child_frame="B"
        )
        
        pose = StampedPose(
            position=[1, 1, 1],
            quaternion=[0, 0, 0, 1],
            frame_id="B"
        )
        
        result = transform.apply_to_pose(pose)
        
        np.testing.assert_array_almost_equal(result.position, [1001, 2001, 3001])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
