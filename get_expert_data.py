import gymnasium as gym
from stable_baselines3 import PPO # SAC, PPO, A2C, DQN
from stable_baselines3.common.env_util import make_vec_env
import numpy as np

# env = gym.make('Pendulum-v0', render_mode="rgb_array")
env = make_vec_env('Pendulum-v1', n_envs = 4)

model = PPO.load("ppo_pendulum", env=env)
# model = PPO("MlpPolicy", env, verbose=1)
# model.learn(total_timesteps=1_000_000)

vec_env = model.get_env()
n_rollouts = 1000
n_timepoints = 100
u = np.zeros((n_rollouts, n_timepoints))
observations = np.zeros((n_rollouts, n_timepoints, 3))
for r in range(n_rollouts):
	obs = vec_env.reset()
	for t in range(n_timepoints):
		# observations[r, t, :] = obs
		action, _state = model.predict(obs, deterministic=True)
		obs, reward, done, info = vec_env.step(action)
		# u[r, t] = action[0]
		vec_env.render("human")

# import pickle
# import os
# path = "/Users/james/Downloads/"
# filename = "pendulum_data.pkl"
# # os.makedirs(os.path.dirname(path))
# obj = {"observations": observations, "u": u}
# with open(path + filename, 'wb') as file: pickle.dump(obj, file)

# obs = vec_env.reset()
# for t in range(n_timepoints):
# 	action, _state = model.predict(obs, deterministic=True)
# 	obs, reward, done, info = vec_env.step(action)
# 	vec_env.render("human")

# Save the agent
# model.save("ppo_pendulum")
# del model  # delete trained model to demonstrate loading

# # Load the trained agent
# model = DQN.load("ppo_pendulum", env=env)

breakpoint()

import numpy as np
x_dim = 2
D = np.zeros((x_dim, x_dim * 2))
for i in range(x_dim):
	for j in range(x_dim * 2):
		if j == i:
			D[i, j] = 1
		elif j == i + 2:
			D[i, j] = -1
Q = 1e2 * D.T @ D
# R = 1 * np.eye(u_dim)
breakpoint()