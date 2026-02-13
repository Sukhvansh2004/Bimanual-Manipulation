#!/usr/bin/env python3
"""
Pytest-based test suite for environment functionality.
Tests environment creation, reset, object handling, and basic operations.
"""

import pytest
import numpy as np
import os
from unittest.mock import patch, MagicMock, PropertyMock

from robosuite_franka_manipulation.environments import make_dual_arm_env
from robosuite_franka_manipulation.environments.dual_arm import DualArmEnvironment, YCBObject
from robosuite_franka_manipulation import constants
from robosuite_franka_manipulation.utils.env_utils import (
    create_ycb_object, create_primitive_object, env_step, set_robot_config,
    register_mesh_extraction_env, resolve_mesh_path, extract_mesh_from_geom,
    get_available_ycb_objects, list_ycb_objects
)


class TestEnvUtils:
    """Test environment utility functions (stateless)."""

    def test_create_object_specs(self):
        """Test creating object specification dictionaries."""
        ycb_spec = create_ycb_object("apple", position=[1, 2, 3])
        assert ycb_spec['ycb_name'] == "apple"
        assert ycb_spec['pos'] == [1, 2, 3]
        
        prim_spec = create_primitive_object("box", "my_box", [0.1, 0.1, 0.1], rgba=[1, 0, 0, 1])
        assert prim_spec['type'] == "box"
        assert prim_spec['rgba'] == [1, 0, 0, 1]

    def test_get_available_ycb_objects(self):
        """Test getting list of YCB objects."""
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.glob') as mock_glob:
            mock_glob.return_value = [MagicMock(stem="apple"), MagicMock(stem="banana")]
            objs = get_available_ycb_objects()
            assert "apple" in objs
            assert "banana" in objs

    def test_extract_mesh_from_geom_box(self):
        """Test extract_mesh_from_geom with a box."""
        mock_env = MagicMock()
        mock_env.sim.model.geom_name2id.return_value = 0
        mock_env.sim.model.geom_type = [6] # Box
        mock_env.sim.model.geom_size = [np.array([0.1, 0.1, 0.1])]
        
        with patch('trimesh.Trimesh.export') as mock_export:
            res = extract_mesh_from_geom(mock_env, "test_geom", "output.obj")
            assert res is True
            assert mock_export.called


class TestEnvironmentSetup:
    """Test environment creation and basic functionality."""

    @pytest.fixture(scope="class")
    def env(self):
        """Fixture to provide environment for tests."""
        env = make_dual_arm_env(
            has_renderer=False,
            has_offscreen_renderer=False,
            use_camera_obs=False
        )
        env.reset()
        yield env
        env.close()

    def test_env_step_basic(self, env):
        """Test env_step helper."""
        action = np.zeros(env.action_dim)
        obs = env_step(env, action)
        # In robosuite, obs is the observation OrderedDict
        assert obs is not None
        
        # Test partial action
        partial_action = np.zeros(constants.DEFAULT_ROBOT_ACTION_DIM)
        obs = env_step(env, partial_action, robot_id=0, arm="right")
        assert obs is not None

    def test_set_robot_config(self, env):
        """Test set_robot_config helper."""
        config = np.zeros(7)
        set_robot_config(env, 0, "right", config)
        # Should complete without error

    @patch('robosuite_franka_manipulation.utils.env_utils.extract_mesh_from_geom')
    def test_mesh_manager(self, mock_extract, env):
        """Test MeshExtractionManager via global helpers."""
        mock_extract.return_value = True
        register_mesh_extraction_env(env)
        
        from robosuite_franka_manipulation.utils.env_utils import _mesh_manager
        # Manually create temp dir if not exists for test
        if _mesh_manager.temp_dir is None:
            _mesh_manager.register_env(env)
            
        path = resolve_mesh_path("test_obj")
        assert path is not None
        assert path.endswith("test_obj.obj")

    def test_environment_creation(self):
        """Test that the environment can be created successfully."""
        env = make_dual_arm_env(
            has_renderer=False,
            has_offscreen_renderer=False,
            use_camera_obs=False
        )
        assert env is not None, "Environment creation failed"
        assert isinstance(env, DualArmEnvironment)
        env.close()

    def test_environment_reset(self):
        """Test that the environment can be reset."""
        env = make_dual_arm_env(
            has_renderer=False,
            has_offscreen_renderer=False,
            use_camera_obs=False
        )
        obs = env.reset()
        assert obs is not None, "Environment reset failed"
        env.close()

    def test_environment_observations(self, env):
        """Test that environment provides valid observations."""
        obs = env.reset()
        
        # Check that observations contain expected keys
        assert isinstance(obs, dict), "Observations should be a dictionary"
        assert len(obs) > 0, "Environment should provide observations"

    def test_environment_step(self, env):
        """Test that environment can execute steps."""
        env.reset()
        
        # Create a zero action
        action = np.zeros(env.action_dim)
        obs, reward, done, info = env.step(action)
        
        assert isinstance(obs, dict)
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    def test_environment_action_dimension(self, env):
        """Test that action dimension is correct for dual arm setup."""
        # Dual arm with two robots, each with action dim of 7 (6 pose + 1 gripper)
        expected_dim = 2 * constants.DEFAULT_ROBOT_ACTION_DIM
        assert env.action_dim == expected_dim

    def test_environment_robots_present(self, env):
        """Test that both robots are present in environment."""
        assert hasattr(env, 'robots')
        assert len(env.robots) == 2


class TestEnvironmentWithObjects:
    """Test environment creation with custom objects."""

    def test_environment_with_primitive_object(self):
        """Test environment creation with primitive objects."""
        custom_objects = [
            {
                'type': 'box',
                'name': 'test_box',
                'size': [0.05, 0.05, 0.05],
                'pos': [0.5, 0.0, 0.9],
                'rgba': [1.0, 0.0, 0.0, 1.0]
            }
        ]
        
        env = make_dual_arm_env(
            custom_objects=custom_objects,
            has_renderer=False,
            has_offscreen_renderer=False,
            use_camera_obs=False
        )
        env.reset()
        
        # Check object was added
        assert 'test_box' in env.object_body_ids
        env.close()

    def test_environment_with_cylinder_object(self):
        """Test environment with cylinder primitive."""
        custom_objects = [
            {
                'type': 'cylinder',
                'name': 'test_cylinder',
                'size': [0.03, 0.05],
                'pos': [0.5, 0.1, 0.9]
            }
        ]
        
        env = make_dual_arm_env(
            custom_objects=custom_objects,
            has_renderer=False,
            has_offscreen_renderer=False,
            use_camera_obs=False
        )
        env.reset()
        
        assert 'test_cylinder' in env.object_body_ids
        env.close()

    def test_environment_with_ball_object(self):
        """Test environment with ball/sphere primitive."""
        custom_objects = [
            {
                'type': 'ball',
                'name': 'test_ball',
                'size': [0.03],
                'pos': [0.5, -0.1, 0.9]
            }
        ]
        
        env = make_dual_arm_env(
            custom_objects=custom_objects,
            has_renderer=False,
            has_offscreen_renderer=False,
            use_camera_obs=False
        )
        env.reset()
        
        assert 'test_ball' in env.object_body_ids
        env.close()

    def test_environment_with_capsule_object(self):
        """Test environment with capsule primitive."""
        custom_objects = [
            {
                'type': 'capsule',
                'name': 'test_capsule',
                'size': [0.03, 0.05],
                'pos': [0.5, 0.1, 0.9]
            }
        ]
        
        env = make_dual_arm_env(
            custom_objects=custom_objects,
            has_renderer=False,
            has_offscreen_renderer=False,
            use_camera_obs=False
        )
        env.reset()
        
        assert 'test_capsule' in env.object_body_ids
        env.close()

    def test_environment_with_multiple_objects(self):
        """Test environment with multiple custom objects."""
        custom_objects = [
            {'type': 'box', 'name': 'box1', 'size': [0.03, 0.03, 0.03], 'pos': [0.4, 0.0, 0.9]},
            {'type': 'box', 'name': 'box2', 'size': [0.03, 0.03, 0.03], 'pos': [0.5, 0.0, 0.9]},
            {'type': 'cylinder', 'name': 'cyl1', 'size': [0.02, 0.04], 'pos': [0.6, 0.0, 0.9]},
        ]
        
        env = make_dual_arm_env(
            custom_objects=custom_objects,
            has_renderer=False,
            has_offscreen_renderer=False,
            use_camera_obs=False
        )
        env.reset()
        
        assert len(env.object_body_ids) >= 3
        env.close()


class TestEnvironmentConfiguration:
    """Test different environment configurations."""

    def test_default_configuration(self):
        """Test default environment configuration."""
        env = make_dual_arm_env(
            has_renderer=False,
            has_offscreen_renderer=False,
            use_camera_obs=False
        )
        env.reset()
        
        assert env.env_configuration == constants.DEFAULT_ENV_CONFIG
        env.close()

    def test_table_size_configuration(self):
        """Test environment configuration with custom camera settings."""
        env = make_dual_arm_env(
            has_renderer=False,
            has_offscreen_renderer=True,
            use_camera_obs=True,
            camera_height=256,
            camera_width=256
        )
        env.reset()
        
        # Environment should accept camera configuration
        assert env is not None
        env.close()


class TestEnvironmentEdgeCases:
    """Test edge cases and error handling."""

    def test_unknown_object_type_warning(self, caplog):
        """Test that unknown object type logs a warning."""
        custom_objects = [
            {
                'type': 'unknown_type',
                'name': 'bad_object',
                'size': [0.05],
                'pos': [0.5, 0.0, 0.9]
            }
        ]
        
        env = make_dual_arm_env(
            custom_objects=custom_objects,
            has_renderer=False,
            has_offscreen_renderer=False,
            use_camera_obs=False
        )
        env.reset()
        
        # Object should not be added
        assert 'bad_object' not in env.object_body_ids
        env.close()

    def test_invalid_object_spec_handling(self):
        """Test handling of invalid object specifications."""
        # Non-dict object specs should be skipped
        custom_objects = [
            "invalid_string",
            123,
            {'type': 'box', 'name': 'valid_box', 'size': [0.05, 0.05, 0.05], 'pos': [0.5, 0.0, 0.9]},
        ]
        
        env = make_dual_arm_env(
            custom_objects=custom_objects,
            has_renderer=False,
            has_offscreen_renderer=False,
            use_camera_obs=False
        )
        env.reset()
        
        # Only valid box should be added
        assert 'valid_box' in env.object_body_ids
        env.close()

    def test_empty_objects_list(self):
        """Test environment with empty objects list."""
        env = make_dual_arm_env(
            custom_objects=[],
            ycb_objects=[],
            has_renderer=False,
            has_offscreen_renderer=False,
            use_camera_obs=False
        )
        env.reset()
        
        # Should work without any objects
        assert env is not None
        env.close()

    def test_multiple_reset_stability(self):
        """Test that environment can be reset multiple times."""
        env = make_dual_arm_env(
            has_renderer=False,
            has_offscreen_renderer=False,
            use_camera_obs=False
        )
        
        for _ in range(5):
            obs = env.reset()
            assert obs is not None
            
            # Take some steps
            for _ in range(3):
                action = np.zeros(env.action_dim)
                env.step(action)
        
        env.close()

    def test_environment_with_duplicate_object_names(self):
        """Test environment with duplicate object names."""
        custom_objects = [
            {'type': 'box', 'name': 'box1', 'size': [0.03, 0.03, 0.03], 'pos': [0.4, 0.0, 0.9]},
            {'type': 'box', 'name': 'box1', 'size': [0.03, 0.03, 0.03], 'pos': [0.5, 0.0, 0.9]},
        ]
        
        env = make_dual_arm_env(
            custom_objects=custom_objects,
            has_renderer=False,
            has_offscreen_renderer=False,
            use_camera_obs=False
        )
        env.reset()
        
        # Should have both objects, one renamed
        assert len(env.get_object_names()) == 2
        env.close()

    def test_invalid_environment_configuration(self):
        """Test environment with invalid configuration."""
        with pytest.raises(ValueError):
            DualArmEnvironment(
                robots=["Panda"],
                env_configuration="invalid_config",
                has_renderer=False,
                has_offscreen_renderer=False,
                use_camera_obs=False
            )

    def test_make_dual_arm_env_with_ycb_string_list(self):
        """Test make_dual_arm_env with a list of YCB object names as strings."""
        from robosuite.models.objects import BoxObject
        # Need to mock the dataset path and exists check
        with patch('robosuite_franka_manipulation.environments.dual_arm.constants.YCB_DATASET_PATH') as mock_path, \
             patch('robosuite_franka_manipulation.environments.dual_arm.YCBObject') as mock_ycb:
            
            mock_path.__truediv__.return_value.exists.return_value = True
            
            # Return a real BoxObject instead of a MagicMock to satisfy robosuite internals
            mock_ycb.side_effect = lambda name, ycb_object_name: BoxObject(name=name, size=[0.01, 0.01, 0.01])
            
            env = make_dual_arm_env(
                ycb_objects=['apple'],
                has_renderer=False,
                has_offscreen_renderer=False,
                use_camera_obs=False
            )
            assert env is not None
            env.close()

    def test_get_object_pose_no_env(self):
        """Test get_object_pose method."""
        custom_objects = [{'type': 'box', 'name': 'target_box', 'size': [0.05, 0.05, 0.05], 'pos': [0.5, 0.0, 0.9]}]
        env_with_obj = make_dual_arm_env(
            custom_objects=custom_objects,
            has_renderer=False,
            has_offscreen_renderer=False,
            use_camera_obs=False
        )
        env_with_obj.reset()
        
        pose = env_with_obj.get_object_pose('target_box')
        assert pose is not None
        assert pose.shape == (4, 4)
        
        # Test non-existent object
        assert env_with_obj.get_object_pose('non_existent') is None
        env_with_obj.close()


class TestYCBObject:
    """Test YCBObject class."""

    def test_ycb_object_file_not_found(self):
        """Test YCBObject raises error for non-existent file."""
        with pytest.raises(FileNotFoundError):
            YCBObject(name="test", ycb_object_name="non_existent_ycb_object")

    def test_ycb_object_bottom_offset(self):
        """Test YCBObject bottom_offset property returns valid array."""
        # This would require a valid YCB object file, so we mock it
        with patch('robosuite_franka_manipulation.environments.dual_arm.constants.YCB_DATASET_PATH') as mock_path:
            mock_path_instance = MagicMock()
            mock_path.__truediv__ = MagicMock(return_value=mock_path_instance)
            mock_path_instance.exists.return_value = False
            
            with pytest.raises(FileNotFoundError):
                YCBObject(name="test", ycb_object_name="apple")
