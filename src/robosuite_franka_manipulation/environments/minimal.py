import robosuite as suite
import mujoco

if __name__ == '__main__':
    # Try to create a minimal environment
    try:
        # Attempt to create a simple environment with a single Panda robot
        # Explicitly set all rendering-related parameters to False or None
        env = suite.make(
            "Lift",                         # Use a simple environment like Lift
            robots="Panda",                 # Single Panda robot
            has_renderer=False,
            has_offscreen_renderer=False,
            render_camera=None,
            control_freq=20,
            horizon=10,                     # Short horizon for quick test
            ignore_done=True,               # Keep environment alive for now
            reward_shaping=False,           # Disable reward shaping for simplicity
        )

        # Reset and step through a few times to test
        env.reset()
        for _ in range(env.horizon):
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            if done:
                break
        print("Minimal robosuite environment created and stepped successfully without rendering.")
        env.close()

    except Exception as e:
        print(f"Failed to create minimal robosuite environment: {e}")