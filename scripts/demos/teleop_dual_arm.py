"""Teleoperate the dual-arm environment with the keyboard.

Wraps the environment in robosuite's ``VisualizationWrapper`` and runs
``bimanual_manipulation.teleop.KeyboardTeleop``.

Usage:
    python scripts/demos/teleop_dual_arm.py
    python scripts/demos/teleop_dual_arm.py --pos-sensitivity 1.5 --rot-sensitivity 0.5
"""

import argparse

from robosuite.wrappers import VisualizationWrapper

# Import the custom environment factory
from bimanual_manipulation.environments import make_dual_arm_env
# Import the teleop module
from bimanual_manipulation.teleop import KeyboardTeleop

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pos-sensitivity",
        type=float,
        default=1.0,
        help="How much to scale position user inputs",
    )
    parser.add_argument(
        "--rot-sensitivity",
        type=float,
        default=1.0,
        help="How much to scale rotation user inputs",
    )
    args = parser.parse_args()

    # Create environment using the custom factory function
    env = make_dual_arm_env()

    # Wrap this environment in a visualization wrapper
    env = VisualizationWrapper(env)

    # Create teleop controller using the modular teleop class
    teleop = KeyboardTeleop(
        env=env,
        pos_sensitivity=args.pos_sensitivity,
        rot_sensitivity=args.rot_sensitivity,
    )

    # Add keypress callback for visualization
    env.viewer.add_keypress_callback(teleop.device.on_press)

    # Run teleoperation
    teleop.run()