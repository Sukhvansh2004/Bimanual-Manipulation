#!/usr/bin/env python3
"""
Pytest-based test suite for perception functionality.
Tests object detection, grasp prediction, and perception pipeline.
"""

import pytest
import numpy as np

from robosuite_franka_manipulation.environments import make_dual_arm_env


class TestPerception:
    """Test perception components."""

    @pytest.fixture(scope="class")
    def env(self):
        """Fixture to provide environment for perception tests."""
        env = make_dual_arm_env()
        env.reset()
        yield env
        env.close()

    def test_object_detector_initialization(self, env):
        """Test that ObjectDetector can be initialized."""
        from robosuite_franka_manipulation.perception import ObjectDetector

        detector = ObjectDetector(env)
        assert detector is not None
        assert hasattr(detector, 'detect_objects')

    def test_grasp_predictor_initialization(self, env):
        """Test that GraspPredictor can be initialized."""
        from robosuite_franka_manipulation.perception import GraspPredictor

        predictor = GraspPredictor()
        assert predictor is not None
        assert hasattr(predictor, 'predict_grasps')

    def test_object_detection(self, env):
        """Test object detection functionality."""
        from robosuite_franka_manipulation.perception import ObjectDetector

        detector = ObjectDetector(env)
        objects = detector.detect_objects()

        # Should return a list (may be empty if no objects detected)
        assert isinstance(objects, list)

        if objects:
            # If objects are detected, check structure
            for obj in objects:
                assert 'name' in obj
                assert 'pose' in obj
                assert 'pcd' in obj
                assert 'mask' in obj
                assert 'grasps' in obj
                assert 'id' in obj

    def test_grasp_prediction(self, env):
        """Test grasp prediction on detected objects."""
        from robosuite_franka_manipulation.perception import ObjectDetector, GraspPredictor

        detector = ObjectDetector(env)
        predictor = GraspPredictor()

        objects = detector.detect_objects()

        if objects:
            obj = objects[0]
            grasps = predictor.predict_grasps(obj["pcd"], obj["pose"], num_grasps=3)

            # Should return a list (may be empty if prediction fails)
            assert isinstance(grasps, list)

            if grasps:
                # If grasps are predicted, check structure
                for grasp in grasps:
                    assert 'pos' in grasp
                    assert 'quat' in grasp
                    assert isinstance(grasp['pos'], np.ndarray)
                    assert isinstance(grasp['quat'], np.ndarray)

    def test_perception_pipeline_integration(self, env):
        """Test the complete perception pipeline integration."""
        from robosuite_franka_manipulation.perception import ObjectDetector

        detector = ObjectDetector(env)

        # Test that the pipeline can be called without errors
        try:
            objects = detector.detect_objects()
            # Pipeline should complete successfully (even if no objects found)
            assert isinstance(objects, list)
        except Exception as e:
            # If services are not running, it should fail gracefully
            assert "service" in str(e).lower() or "connection" in str(e).lower()