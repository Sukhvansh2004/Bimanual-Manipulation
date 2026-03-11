#!/usr/bin/env python3
"""
Pytest-based test suite for environment functionality.
Tests environment creation, reset, and basic operations.
"""

import pytest
from bimanual_manipulation.environments import make_dual_arm_env


class TestEnvironmentSetup:
    """Test environment creation and basic functionality."""

    def test_environment_creation(self):
        """Test that the environment can be created successfully."""
        env = make_dual_arm_env(has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False)
        assert env is not None, "Environment creation failed"
        env.close()

    def test_environment_reset(self):
        """Test that the environment can be reset."""
        env = make_dual_arm_env(has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False)
        obs = env.reset()
        assert obs is not None, "Environment reset failed"
        env.close()

    def test_environment_observations(self):
        """Test that environment provides valid observations."""
        env = make_dual_arm_env(has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False)
        obs = env.reset()

        # Check that observations contain expected keys
        assert isinstance(obs, dict), "Observations should be a dictionary"

        # Basic observation structure check
        # Note: Specific observation keys depend on environment configuration
        assert len(obs) > 0, "Environment should provide observations"

        env.close()


# ═══════════════════════════════════════════════════════════════════════════
# Tests — CollisionMonitor (unit tests with mocked in_collision)
# ═══════════════════════════════════════════════════════════════════════════

class TestCollisionMonitor:
    """Unit tests for CollisionMonitor ignore_pairs / min_depth features."""

    def test_start_sets_ignore_pairs(self):
        """ignore_pairs should be stored and forwarded to in_collision."""
        from bimanual_manipulation.utils.env_utils import CollisionMonitor
        mon = CollisionMonitor()
        pairs = [("gripper0", "gripper1")]
        mon.start("obj_0", ignore_pairs=pairs)
        assert mon._ignore_pairs == pairs

    def test_start_resets_count(self):
        """Starting monitoring should reset the collision count."""
        from bimanual_manipulation.utils.env_utils import CollisionMonitor
        mon = CollisionMonitor()
        mon._collision_count = 42
        mon.start("obj_0")
        assert mon.collision_count == 0

    def test_min_penetration_depth_default(self):
        """Default min_penetration_depth should come from constants."""
        from bimanual_manipulation.utils.env_utils import CollisionMonitor
        from bimanual_manipulation import constants
        mon = CollisionMonitor()
        mon.start()
        assert mon._min_penetration_depth == constants.COLLISION_MONITOR_MIN_DEPTH

    def test_min_penetration_depth_override(self):
        """Caller should be able to override min_penetration_depth."""
        from bimanual_manipulation.utils.env_utils import CollisionMonitor
        mon = CollisionMonitor()
        mon.start(min_penetration_depth=0.005)
        assert mon._min_penetration_depth == 0.005

    def test_check_passes_ignore_pairs_and_threshold(self):
        """check() should forward ignore_pairs and contact_dist_threshold."""
        from unittest.mock import patch, MagicMock
        from bimanual_manipulation.utils.env_utils import CollisionMonitor

        mon = CollisionMonitor()
        pairs = [("gripper0", "gripper1")]
        mon.start("plate_0", ignore_pairs=pairs, min_penetration_depth=0.002)

        mock_env = MagicMock()
        with patch(
            "bimanual_manipulation.planning.utils.collision_utils.in_collision",
        ) as mock_ic_real:
            mock_ic_real.return_value = False
            mon.check(mock_env)
            # Verify in_collision was called with the right kwargs
            call_kwargs = mock_ic_real.call_args
            assert call_kwargs is not None
            _, kw = call_kwargs
            assert kw.get("ignore_names") == ["plate_0"]
            assert kw.get("ignore_pairs") == pairs
            # contact_dist_threshold = -min_penetration_depth
            assert kw.get("contact_dist_threshold") == -0.002

    def test_stop_returns_accumulated_count(self):
        """stop() should return and reset the count."""
        from bimanual_manipulation.utils.env_utils import CollisionMonitor
        mon = CollisionMonitor()
        mon.start()
        mon._collision_count = 7
        assert mon.stop() == 7
        assert not mon.enabled