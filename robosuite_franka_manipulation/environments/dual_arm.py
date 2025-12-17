import numpy as np
import robosuite as suite

if __name__ == '__main__':
    # create environment instance
    env = suite.make(
        env_name="TwoArmHandover", # try with other tasks like "Stack" and "Door"
        robots=["Panda", "Panda"],  # try with other robots like "Sawyer" and "Jaco"
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
    )

    # reset the environment
    env.reset()

    for i in range(1000):
        action = np.random.randn(*env.action_spec[0].shape) * 0.1
        obs, reward, done, info = env.step(action)  # take action in the environment
        # env.render()  # render on display - disabled for headless