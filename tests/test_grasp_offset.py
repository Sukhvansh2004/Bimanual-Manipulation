#!/usr/bin/env python3
"""
Test suite for checking correct application of gripper offsets in GraspPredictor.
"""

import pytest
import numpy as np
from scipy.spatial.transform import Rotation as R
from unittest.mock import MagicMock, patch
from robosuite_franka_manipulation.perception.grasp_predictor import GraspPredictor
from robosuite_franka_manipulation.transforms import StampedPose
from robosuite_franka_manipulation import constants

class TestGraspOffset:
    
    @pytest.fixture
    def predictor(self):
        with patch('robosuite_franka_manipulation.perception.grasp_predictor.GraspGenSampler'), \
             patch('robosuite_franka_manipulation.perception.grasp_predictor.load_grasp_cfg'):
            # Initialize with a dummy gripper type
            predictor = GraspPredictor(gripper_type='test_gripper')
            return predictor

    def test_apply_gripper_offset_rotation_only(self, predictor):
        """Test applying only rotation offset."""
        # Setup offset for test_gripper: 90 deg around Z, no translation
        constants.GRIPPER_GRASP_OFFSETS['test_gripper'] = {
            'position': np.array([0.0, 0.0, 0.0]),
            'quaternion': R.from_euler('z', 90, degrees=True).as_quat()
        }
        
        # Original grasp: identity orientation
        grasp_pose = StampedPose(
            position=[1.0, 2.0, 3.0],
            quaternion=[0, 0, 0, 1],
            frame_id="world"
        )
        
        offset_grasp = predictor._apply_gripper_offset(grasp_pose)
        
        # Position should be unchanged
        np.testing.assert_array_almost_equal(offset_grasp.position, [1.0, 2.0, 3.0])
        
        # Orientation should be rotated 90 deg around Z
        expected_quat = R.from_euler('z', 90, degrees=True).as_quat()
        # Quaternion sign ambiguity check
        dot = np.abs(np.dot(offset_grasp.quaternion, expected_quat))
        assert abs(dot - 1.0) < 1e-6

    def test_apply_gripper_offset_translation_rotated(self, predictor):
        """
        Test that translation offset is correctly rotated by the grasp orientation.
        This verifies the fix: offset should be in grasp frame, not world frame.
        """
        # Setup offset: Move 0.1m along Z (gripper forward direction)
        offset_z = 0.1
        constants.GRIPPER_GRASP_OFFSETS['test_gripper'] = {
            'position': np.array([0.0, 0.0, offset_z]),
            'quaternion': np.array([0, 0, 0, 1]) # Identity rotation
        }
        
        # Case 1: Grasp aligned with world (Identity)
        grasp_pose_identity = StampedPose(
            position=[1.0, 1.0, 1.0],
            quaternion=[0, 0, 0, 1],
            frame_id="world"
        )
        
        offset_grasp_1 = predictor._apply_gripper_offset(grasp_pose_identity)
        # Expected: [1, 1, 1+0.1]
        np.testing.assert_array_almost_equal(offset_grasp_1.position, [1.0, 1.0, 1.1])
        
        # Case 2: Grasp rotated 90 deg around X
        # World Y becomes Grasp Z. 
        # So offset along Grasp Z should be offset along World Y.
        rot_x90 = R.from_euler('x', 90, degrees=True)
        grasp_pose_rotated = StampedPose(
            position=[1.0, 1.0, 1.0],
            quaternion=rot_x90.as_quat(),
            frame_id="world"
        )
        
        offset_grasp_2 = predictor._apply_gripper_offset(grasp_pose_rotated)
        
        # Expected: [1, 1-0.1, 1] because World Y -> Grasp Z is [0, 1, 0] -> [0, 0, 1]?
        # Let's trace:
        # R_x(90) * [0, 0, 1] (offset) 
        # Matrix: [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
        # [0, 0, 1] -> [0, -1, 0]
        # So position should change in Y by -0.1
        
        rotated_offset = rot_x90.apply([0, 0, 0.1])
        # [0, -0.1, 0]
        
        expected_pos = np.array([1.0, 1.0, 1.0]) + rotated_offset
        np.testing.assert_array_almost_equal(offset_grasp_2.position, expected_pos)
        
        # Verify it's NOT just adding [0, 0, 0.1] (which would be [1, 1, 1.1])
        assert offset_grasp_2.position[2] == 1.0 
        assert abs(offset_grasp_2.position[1] - 0.9) < 1e-6

