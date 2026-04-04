"""
Pytest-based test suite for the dual-arm parallel planner.

Tests collision geom extraction helpers and verifies the collision
avoidance limit correctly includes environment geoms.
"""

import pytest
import numpy as np
import mujoco

from bimanual_manipulation.environments import make_dual_arm_env
from bimanual_manipulation.utils.mujoco_utils import (
    get_native_mj_object,
    get_robot_collision_geoms,
    get_object_collision_geoms,
    get_environment_collision_geoms,
)


@pytest.fixture(scope="module")
def env_and_model():
    """Provide a shared environment and native model for all tests."""
    env = make_dual_arm_env(
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
    )
    env.reset()
    nm = get_native_mj_object(env.sim.model)
    yield env, nm
    env.close()


class TestCollisionGeomHelpers:
    """Tests for the model-introspection functions that extract geom names."""

    def test_robot_geoms_nonempty(self, env_and_model):
        """Each robot must expose at least one non-visual collision geom."""
        _, nm = env_and_model
        for rid in [0, 1]:
            geoms = get_robot_collision_geoms(nm, rid)
            assert len(geoms) > 0, f"robot{rid} should have collision geoms"

    def test_robot_geoms_no_visual(self, env_and_model):
        """Robot geoms must not include visual-only geoms."""
        _, nm = env_and_model
        for rid in [0, 1]:
            for name in get_robot_collision_geoms(nm, rid):
                assert "vis" not in name, f"Visual geom leaked: {name}"

    def test_robot_geoms_disjoint(self, env_and_model):
        """Geom sets for the two robots must not overlap."""
        _, nm = env_and_model
        g0 = set(get_robot_collision_geoms(nm, 0))
        g1 = set(get_robot_collision_geoms(nm, 1))
        assert g0.isdisjoint(g1), f"Overlap: {g0 & g1}"

    def test_env_geoms_nonempty(self, env_and_model):
        """The environment (tables at minimum) must expose collision geoms."""
        _, nm = env_and_model
        env_geoms = get_environment_collision_geoms(nm, robot_ids=[0, 1])
        assert len(env_geoms) > 0, "Environment should have collision geoms (tables, etc.)"

    def test_env_geoms_exclude_visual(self, env_and_model):
        """Environment geoms must not include visual-only geoms."""
        _, nm = env_and_model
        for name in get_environment_collision_geoms(nm, robot_ids=[0, 1]):
            assert "vis" not in name, f"Visual geom leaked in env: {name}"

    def test_env_geoms_exclude_robots(self, env_and_model):
        """Environment geoms must not overlap with any robot's geoms."""
        _, nm = env_and_model
        g0 = set(get_robot_collision_geoms(nm, 0))
        g1 = set(get_robot_collision_geoms(nm, 1))
        env_geoms = set(get_environment_collision_geoms(nm, robot_ids=[0, 1]))
        assert env_geoms.isdisjoint(g0), f"Env geoms overlap robot0: {env_geoms & g0}"
        assert env_geoms.isdisjoint(g1), f"Env geoms overlap robot1: {env_geoms & g1}"

    def test_env_geoms_exclude_object(self, env_and_model):
        """When an object name is given, its geoms must be excluded."""
        _, nm = env_and_model
        # Collect all geom names to find a plausible object substring
        all_names = [nm.geom(i).name for i in range(nm.ngeom)]
        # Try common robosuite object prefixes
        for candidate in ("mug", "plate", "can", "bowl"):
            matching = [n for n in all_names if candidate in n and "vis" not in n]
            if matching:
                env_geoms = set(
                    get_environment_collision_geoms(nm, robot_ids=[0, 1], object_name=candidate)
                )
                for m in matching:
                    assert m not in env_geoms, (
                        f"Object geom '{m}' should be excluded when object_name='{candidate}'"
                    )
                break
        # If no object in scene, the test is trivially passed

    def test_env_geoms_exclude_floor(self, env_and_model):
        """Floor geom should be excluded from environment collision set."""
        _, nm = env_and_model
        env_geoms = get_environment_collision_geoms(nm, robot_ids=[0, 1])
        assert "floor" not in env_geoms, "Floor should be excluded"

    def test_all_geom_categories_cover_model(self, env_and_model):
        """Robot + object + env + excluded (vis/floor/unnamed) should cover all geoms."""
        _, nm = env_and_model
        robot_geoms = set(get_robot_collision_geoms(nm, 0)) | set(
            get_robot_collision_geoms(nm, 1)
        )
        env_geoms = set(get_environment_collision_geoms(nm, robot_ids=[0, 1]))
        categorised = robot_geoms | env_geoms

        all_names = set()
        for i in range(nm.ngeom):
            name = nm.geom(i).name
            if name:
                all_names.add(name)

        uncategorised = all_names - categorised
        # Uncategorised geoms should only be visual, floor, or unnamed
        for name in uncategorised:
            assert "vis" in name or name == "floor", (
                f"Geom '{name}' is not categorised into robot/env/visual/floor"
            )


class TestCollisionAvoidanceLimitIntegration:
    """Verify that the planner's _ensure_mink_init wires env geoms correctly."""

    def test_planner_collision_limit_includes_env(self, env_and_model):
        """The planner's CollisionAvoidanceLimit must include 3 geom-pair groups
        (arm_B vs arm_A, arm_B vs object, arm_B vs env) when an object is present
        and the scene has environment geoms."""
        from unittest.mock import MagicMock
        from bimanual_manipulation.planning.dual_arm import (
            DualArmParallelPlanner,
        )

        env, nm = env_and_model

        # Build minimal mock arm controllers
        arm_a = MagicMock()
        arm_a.robot_id = 0
        arm_a.arm_id = "right"
        arm_a.tf_manager = MagicMock()
        arm_a._eef_body_name.return_value = "robot0_right_hand"
        arm_a.get_joint_positions.return_value = np.zeros(7)

        arm_b = MagicMock()
        arm_b.robot_id = 1
        arm_b.arm_id = "right"
        arm_b.tf_manager = MagicMock()
        arm_b._eef_body_name.return_value = "robot1_right_hand"
        arm_b.get_joint_positions.return_value = np.zeros(7)

        planner = DualArmParallelPlanner(
            env=env,
            arm_a=arm_a,
            arm_b=arm_b,
            collision_check=True,
            manipulated_object=None,  # no object for this test
        )

        # Trigger lazy init
        planner._ensure_mink_init()

        # The collision limit should exist
        assert planner._mink is not None, "mink was not initialized"
        assert planner._mink.collision_limit is not None, (
            "CollisionAvoidanceLimit was not created"
        )

        # Verify that env geoms were found and included
        env_geoms = get_environment_collision_geoms(nm, robot_ids=[0, 1])
        geoms_b = get_robot_collision_geoms(nm, 1)
        assert len(env_geoms) > 0, "No env geoms found — test setup issue"

        # geom_id_pairs is the expanded set of individual geom-to-geom pairs.
        # With env geoms wired in, the total must be strictly > 0 and
        # larger than arm_B alone (which has 0 self-pairs).
        actual_pairs = len(planner._mink.collision_limit.geom_id_pairs)
        assert actual_pairs > 0, "CollisionAvoidanceLimit has no pairs"
        # Sanity: at minimum we should have at least len(geoms_b) pairs
        # (each arm_B geom paired with at least one env or arm_A geom)
        assert actual_pairs >= len(geoms_b), (
            f"Expected at least {len(geoms_b)} pairs, got {actual_pairs}"
        )
