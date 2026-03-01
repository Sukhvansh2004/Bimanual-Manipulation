#!/usr/bin/env python3
"""
Pytest-based test suite for environment functionality.
Tests environment creation, reset, and basic operations.
"""

import pytest
from robosuite_franka_manipulation.environments import make_dual_arm_env


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