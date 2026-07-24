#!/usr/bin/env python3
"""
Comprehensive tests for experiment configuration loader (experiment_config.py).

Covers dataclass construction, JSON loading, parsing helpers, default values,
error handling, and the collision-filter function builder.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from bimanual_manipulation.experiment_config import (
    AssignmentConfig,
    CollisionFilterConfig,
    ExecutionConfig,
    ExperimentConfig,
    OutputConfig,
    PlanningConfig,
    PoseConfig,
    RuntimeConfig,
    TaskConfig,
    VisualizationConfig,
    _parse_pose,
    get_collision_filter_fn,
    list_experiments,
    load_experiment,
)
from bimanual_manipulation import constants


# ═══════════════════════════════════════════════════════════════════════════════
# PoseConfig
# ═══════════════════════════════════════════════════════════════════════════════

class TestPoseConfig:
    def test_creation(self):
        pc = PoseConfig(
            position=np.array([1.0, 2.0, 3.0]),
            quaternion=np.array([0.0, 0.0, 0.0, 1.0]),
        )
        np.testing.assert_array_equal(pc.position, [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(pc.quaternion, [0.0, 0.0, 0.0, 1.0])

    def test_to_stamped_pose(self):
        pc = PoseConfig(
            position=np.array([1.0, 2.0, 3.0]),
            quaternion=np.array([0.0, 0.0, 0.0, 1.0]),
        )
        sp = pc.to_stamped_pose("test_frame")
        assert sp.frame_id == "test_frame"
        np.testing.assert_array_equal(sp.position, [1.0, 2.0, 3.0])

    def test_to_stamped_pose_default_frame(self):
        pc = PoseConfig(
            position=np.zeros(3), quaternion=np.array([0, 0, 0, 1.0])
        )
        sp = pc.to_stamped_pose()
        assert sp.frame_id == constants.WORLD_FRAME


# ═══════════════════════════════════════════════════════════════════════════════
# _parse_pose
# ═══════════════════════════════════════════════════════════════════════════════

class TestParsePose:
    def test_none_returns_none(self):
        assert _parse_pose(None) is None

    def test_valid_dict(self):
        data = {"position": [1, 2, 3], "quaternion": [0, 0, 0, 1]}
        pc = _parse_pose(data)
        assert isinstance(pc, PoseConfig)
        np.testing.assert_array_almost_equal(pc.position, [1, 2, 3])

    def test_float_coercion(self):
        data = {"position": [1, 2, 3], "quaternion": [0, 0, 0, 1]}
        pc = _parse_pose(data)
        assert pc.position.dtype == float
        assert pc.quaternion.dtype == float


# ═══════════════════════════════════════════════════════════════════════════════
# PlanningConfig defaults
# ═══════════════════════════════════════════════════════════════════════════════

class TestPlanningConfig:
    def test_defaults(self):
        pc = PlanningConfig()
        assert pc.num_grasps == 200
        assert pc.num_entropy_grasps == 20
        assert pc.use_dpp is True
        assert "finger" in pc.gripper_types
        assert "suction" in pc.gripper_types
        assert pc.collision_check == "Inter"
        assert pc.num_steps == constants.OBJECT_TRAJ_NUM_STEPS
        assert pc.rrt_bounds is None


# ═══════════════════════════════════════════════════════════════════════════════
# AssignmentConfig defaults
# ═══════════════════════════════════════════════════════════════════════════════

class TestAssignmentConfig:
    def test_defaults(self):
        ac = AssignmentConfig()
        assert ac.alpha == constants.ASSIGNMENT_ALPHA
        assert ac.beta == constants.ASSIGNMENT_BETA
        assert ac.epsilon == constants.ASSIGNMENT_EPSILON
        assert ac.initial_robot is None
        assert ac.beam_width is None
        assert ac.handover_stride == 1


# ═══════════════════════════════════════════════════════════════════════════════
# get_collision_filter_fn
# ═══════════════════════════════════════════════════════════════════════════════

class TestGetCollisionFilterFn:
    def test_robot_geoms_always_kept(self):
        fn = get_collision_filter_fn(CollisionFilterConfig(relevant_patterns=[]))
        assert fn("robot0_link5") is True
        assert fn("gripper0_finger") is True
        assert fn("robot1_base") is True
        assert fn("gripper1_pad") is True

    def test_task_patterns_matched(self):
        fn = get_collision_filter_fn(
            CollisionFilterConfig(relevant_patterns=["flatPlate", "boxA"])
        )
        assert fn("flatPlate_geom") is True
        assert fn("boxABottom_geom") is True
        assert fn("wallObstacle_geom") is False

    def test_no_patterns_only_robots(self):
        fn = get_collision_filter_fn(CollisionFilterConfig())
        assert fn("table_surface") is False
        assert fn("robot0_link0") is True


# ═══════════════════════════════════════════════════════════════════════════════
# list_experiments
# ═══════════════════════════════════════════════════════════════════════════════

class TestListExperiments:
    def test_returns_sorted_list(self):
        exps = list_experiments()
        assert isinstance(exps, list)
        assert len(exps) >= 4  # We know at least 4 main experiments exist
        for p in exps:
            assert p.endswith(".json")
        assert exps == sorted(exps)

    def test_nonexistent_dir_returns_empty(self):
        exps = list_experiments("/no/such/directory")
        assert exps == []

    def test_custom_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "test1.json").touch()
            Path(tmpdir, "test2.json").touch()
            Path(tmpdir, "not_json.txt").touch()
            exps = list_experiments(tmpdir)
            assert len(exps) == 2


# ═══════════════════════════════════════════════════════════════════════════════
# load_experiment (integration with real experiment files)
# ═══════════════════════════════════════════════════════════════════════════════

class TestLoadExperiment:
    """Test loading real experiment JSON files."""

    @pytest.fixture(params=list_experiments() or ["skip"])
    def experiment_path(self, request):
        """Parametrize over all available experiment files."""
        if request.param == "skip":
            pytest.skip("No experiment files found")
        return request.param

    def test_loads_successfully(self, experiment_path):
        cfg = load_experiment(experiment_path)
        assert isinstance(cfg, ExperimentConfig)

    def test_has_name_and_description(self, experiment_path):
        cfg = load_experiment(experiment_path)
        assert isinstance(cfg.name, str) and len(cfg.name) > 0
        assert isinstance(cfg.description, str)

    def test_task_has_target_pose(self, experiment_path):
        cfg = load_experiment(experiment_path)
        assert isinstance(cfg.task.target_pose, PoseConfig)
        assert cfg.task.target_pose.position.shape == (3,)
        assert cfg.task.target_pose.quaternion.shape == (4,)

    def test_task_has_manipulated_object(self, experiment_path):
        cfg = load_experiment(experiment_path)
        assert isinstance(cfg.task.manipulated_object, str)
        assert len(cfg.task.manipulated_object) > 0

    def test_raw_dict_preserved(self, experiment_path):
        cfg = load_experiment(experiment_path)
        assert isinstance(cfg.raw, dict)
        assert "task" in cfg.raw

    def test_planning_positive_steps(self, experiment_path):
        cfg = load_experiment(experiment_path)
        assert cfg.planning.num_steps > 0

    def test_all_sections_present(self, experiment_path):
        cfg = load_experiment(experiment_path)
        assert isinstance(cfg.planning, PlanningConfig)
        assert isinstance(cfg.collision_filter, CollisionFilterConfig)
        assert isinstance(cfg.visualization, VisualizationConfig)
        assert isinstance(cfg.output, OutputConfig)
        assert isinstance(cfg.execution, ExecutionConfig)
        assert isinstance(cfg.runtime, RuntimeConfig)
        assert isinstance(cfg.assignment, AssignmentConfig)


class TestLoadExperimentErrors:
    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_experiment("/no/such/file.json")

    def test_invalid_json_raises(self):
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            f.write("not valid json{{{")
            f.flush()
            try:
                with pytest.raises(json.JSONDecodeError):
                    load_experiment(f.name)
            finally:
                os.unlink(f.name)

    def test_missing_task_key_raises(self):
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            json.dump({"name": "test", "description": "test"}, f)
            f.flush()
            try:
                with pytest.raises(KeyError):
                    load_experiment(f.name)
            finally:
                os.unlink(f.name)


class TestLoadExperimentMinimalJSON:
    """Test loading a minimal but valid experiment JSON."""

    @pytest.fixture
    def minimal_json(self, tmp_path):
        data = {
            "name": "test_exp",
            "description": "A test experiment",
            "environment": {"ycb_objects": [], "custom_objects": []},
            "task": {
                "manipulated_object": "test_obj",
                "target_pose": {
                    "position": [0.1, 0.2, 0.3],
                    "quaternion": [0, 0, 0, 1],
                },
            },
        }
        p = tmp_path / "test_exp.json"
        p.write_text(json.dumps(data))
        return str(p)

    def test_minimal_loads(self, minimal_json):
        cfg = load_experiment(minimal_json)
        assert cfg.name == "test_exp"
        assert cfg.description == "A test experiment"
        np.testing.assert_array_almost_equal(
            cfg.task.target_pose.position, [0.1, 0.2, 0.3]
        )

    def test_defaults_applied(self, minimal_json):
        cfg = load_experiment(minimal_json)
        # All optional sections should get defaults
        assert cfg.planning.num_grasps == 200
        assert cfg.visualization.enable_meshcat is True
        assert cfg.output.save_trajectories is False
        assert cfg.execution.visualize is True
        assert cfg.runtime.debug is False
        assert cfg.assignment.handover_stride == 1

    def test_start_pose_none_by_default(self, minimal_json):
        cfg = load_experiment(minimal_json)
        assert cfg.task.start_pose is None

    def test_rrt_bounds_none_by_default(self, minimal_json):
        cfg = load_experiment(minimal_json)
        assert cfg.planning.rrt_bounds is None

    def test_table_offset_none_by_default(self, minimal_json):
        cfg = load_experiment(minimal_json)
        assert cfg.table_offset is None

    def test_robot_base_configs_none_by_default(self, minimal_json):
        cfg = load_experiment(minimal_json)
        assert cfg.robot_base_configs is None

    def test_table_offset_parsed(self, tmp_path):
        data = {
            "name": "table_offset_test",
            "description": "",
            "environment": {
                "table_full_size": [1.0, 0.75, 0.05],
                "table_offset": [0.0, 0.0, 0.8],
                "ycb_objects": [],
                "custom_objects": [],
            },
            "task": {
                "manipulated_object": "obj",
                "target_pose": {
                    "position": [0.1, 0.2, 0.3],
                    "quaternion": [0, 0, 0, 1],
                },
            },
        }
        p = tmp_path / "table_offset_test.json"
        p.write_text(json.dumps(data))

        cfg = load_experiment(str(p))
        assert cfg.table_offset == (0.0, 0.0, 0.8)

    def test_robot_base_configs_parsed(self, tmp_path):
        data = {
            "name": "robot_base_test",
            "description": "",
            "environment": {
                "ycb_objects": [],
                "custom_objects": [],
                "robot_base_configs": [
                    {
                        "position": [0.0, 0.1, 0.0],
                        "quaternion": [0.0, 0.0, 0.70710678, 0.70710678],
                    },
                    {
                        "position": [0.0, -0.1, 0.0],
                        "quaternion": [0.0, 0.0, -0.70710678, 0.70710678],
                    },
                ],
            },
            "task": {
                "manipulated_object": "obj",
                "target_pose": {
                    "position": [0.1, 0.2, 0.3],
                    "quaternion": [0, 0, 0, 1],
                },
            },
        }
        p = tmp_path / "robot_base_test.json"
        p.write_text(json.dumps(data))

        cfg = load_experiment(str(p))
        assert cfg.robot_base_configs is not None
        np.testing.assert_array_equal(cfg.robot_base_configs[0]["position"], [0.0, 0.1, 0.0])
        np.testing.assert_array_equal(cfg.robot_base_configs[1]["position"], [0.0, -0.1, 0.0])


class TestLoadExperimentRRTBounds:
    """Test RRT bounds parsing edge cases."""

    @pytest.fixture
    def make_json(self, tmp_path):
        def _factory(rrt_bounds):
            data = {
                "name": "rrt_test",
                "description": "",
                "environment": {"ycb_objects": [], "custom_objects": []},
                "task": {
                    "manipulated_object": "obj",
                    "target_pose": {"position": [0, 0, 0], "quaternion": [0, 0, 0, 1]},
                },
                "planning": {"rrt_bounds": rrt_bounds},
            }
            p = tmp_path / "rrt_test.json"
            p.write_text(json.dumps(data))
            return str(p)
        return _factory

    def test_valid_rrt_bounds(self, make_json):
        cfg = load_experiment(make_json([[-1, -1, -1], [1, 1, 1]]))
        assert cfg.planning.rrt_bounds is not None
        np.testing.assert_array_equal(cfg.planning.rrt_bounds, [[-1, -1, -1], [1, 1, 1]])

    def test_invalid_rrt_bounds_shape_ignored(self, make_json):
        cfg = load_experiment(make_json([[-1, -1], [1, 1]]))
        assert cfg.planning.rrt_bounds is None
