import robosuite as suite
import numpy as np
from robosuite.controllers import load_composite_controller_config

def make_dual_arm_handoff_env():
    """
    Creates a robosuite environment configured for a dual-arm Franka handoff task.
    Rendering is disabled due to EGL issues in the current environment.
    """
    # This ensures all registry keys and internal defaults are present.
    osc_config = load_composite_controller_config(controller="BASIC")
    
    osc_config.update({
        "impedance_mode": "fixed",
        "kp": 1000,
        "damping_ratio": 1,
        "set_gains": True,
        "clip_torques": True,
        "torque_limits": (-100, 100),
        "input_max": 1,
        "input_min": -1,
        "output_max": 0.05,
        "output_min": -0.05,
        "control_delta": True,
        "uncouple_pos_ori": True,
        # 'decoupled_op_space' is usually handled by uncouple_pos_ori in newer versions, 
        # but can be kept if your specific version requires it.
        "decoupled_op_space": True, 
        "position_limits": None,
        "orientation_limits": None,
    })

    # For dual-arm, controller_configs should be a list of configs
    controller_configs = [osc_config, osc_config]

    env = suite.make(
        "TwoArmHandover",            # Use the TwoArmHandover environment
        robots=["Panda", "Panda"],   # Specify two Panda robots
        gripper_types="default",     # Use default grippers for each robot
        controller_configs=controller_configs,
        env_configuration="opposed", # Robots face each other
        has_renderer=False,          # Disable rendering for headless
        has_offscreen_renderer=False, # Disable offscreen rendering
        use_camera_obs=False,        # Disable camera observations
        control_freq=20,             # Control frequency
        horizon=1000,                # Longer horizon for sequence
        ignore_done=True,            # Keep environment alive
        reward_shaping=False,        # Disable reward shaping
    )
    return env

if __name__ == '__main__':
    try:
        env = make_dual_arm_handoff_env()
        print("Dual-arm handoff environment created successfully (headless).")

        # Reset and step through a few times to test
        env.reset()
        for _ in range(env.horizon):
            action = env.action_space.sample()  # Sample a random action
            obs, reward, done, info = env.step(action)
            if done:
                break
        print(f"Environment stepped for {env.horizon} steps.")
        env.close()

    except Exception as e:
        print(f"Failed to create or step dual-arm handoff environment: {e}")