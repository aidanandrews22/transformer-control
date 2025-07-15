
import gym
import time
import imageio
import os
import numpy as np
import math
from gym_continuous_cartpole import ContinuousCartPoleEnv
from gym_cartpole_swingup_lqr import swingup_lqr_controller

# env = ContinuousCartPoleEnv(render_mode="rgb_array")
env = ContinuousCartPoleEnv(masscart=2.0, masspole=0.5, length=1.0, render_mode="rgb_array")
# obs, _ = env.reset()
obs, info = env.reset(options={"init_state": [0.0, 0.0, np.pi/3, 0.0]})
frames = []
states = []
actions = []
states.append(obs)
switched = False

for _ in range(1000):
    # action = np.array([0.0])  # zero control
    # action = swingup_lqr_controller(obs, env.masscart, env.masspole, env.length)
    action, switched = swingup_lqr_controller(obs, switched, env.masscart, env.masspole, env.length)
    obs, reward, done, truncated, _ , applied_action = env.step(action)
    frame = env.render()
    frames.append(frame)
    states.append(obs)
    actions.append(applied_action)
    if done or truncated:
        obs, _ = env.reset()

env.close()

# Save video
path = os.path.join(os.getcwd(), 'videos', 'cartpole_gym')
os.makedirs(path, exist_ok=True)

file_path = os.path.join(path, 'cartpole.mp4')
# imageio.mimsave(file_path, env.render(mode='rgb_array'), fps=50)
imageio.mimsave(file_path, frames, fps=50)
states = np.array(states)
# print(f"states: {states}")
# print(f"states.shape: {states.shape}")
actions = np.array(actions).squeeze()
print(f"actions: {actions}")

