import jax.random as jr
from svae.experiments import run_lds, run_pendulum_control
from flax.training.orbax_utils import save_args_from_target
from svae.utils import get_train_state, scale_matrix_by_norm

run_params = {
    "inference_method": "svae",
    "latent_dims": 3,
    "emission_dims": 3,
    "input_dims": 1,
    "rec_features": [512],
    "dec_features": [512],
    # "rnn_dims": 10,
    "seed": jr.PRNGKey(1),
    # "dataset_size": "medium",
    # "snr": "medium",
    "sample_kl": False,
    "use_parallel_kf": True,
    "train_batch_size": 100,
    "val_batch_size": 200,
    # "dimensionality": "medium",
    "num_timesteps": 100,
    "constrain_prior": False,
    # "constrain_dynamics": True, # dynamics now constrained by design
    "base_lr": .001,
    "lr_decay": False, # definitely won't want this on when I am learning online
    "prior_base_lr": .001,
    "prior_lr_warmup": False,
    "max_iters": 10000,
    "log_to_wandb": True,
    "min_delta": 1e-4,
    "patience": 10,
    "max_grad_norm": 10,
    "weight_decay": 0, # 0.0001
    "beta_schedule": "linear_slow", # "linear_slow", "linear_fast"
    "early_stop_start": 6000, # linear_slow beta schedule starts at 1000 and ramps up over the next 5000 (i.e. reaches 1 at 6000)
    "save_dir": "/Users/james/Dropbox (UCL)/ucgtjhe@ucl.ac.uk’s files/James MacBook/Gatsby/svae/runs/",
    "reload_dir": "/Users/james/Dropbox (UCL)/ucgtjhe@ucl.ac.uk’s files/James MacBook/Gatsby/svae/runs/",
    "reload_state": True,
    "project_name": "SVAE-control-pendulum",
    "run_type": "model_initialising" # "model_learning"
}

all_results, all_models = run_pendulum_control(run_params)

model_names = ["recognition_model", "decoder_model", "prior_model"]
optimisers = all_results[0][4]
opt_states = all_results[0][5]
states = []
ckptrs = []
for i in range(3):
    state, ckptr = get_train_state(optimisers[i], [], opt_states[i].params, run_params, model_names[i])
    states.append(state), ckptrs.append(ckptr)

params = {}
params['rec_params'] = states[0].params
params['dec_params'] = states[1].params
params['prior_params'] = states[2].params

################################################################################################

# import numpy as np
# from svae.utils import lie_params_to_constrained, construct_dynamics_matrix
# import pickle

# obj = pickle.load(open("pendulum_data.pkl", 'rb'))
# data_dict = {}
# data_dict["train_data"] = np.array(obj['observations'][:800, :, :])
# data_dict["train_u"] = np.array(obj['u'][:800, :-1, None])
# data_dict["val_data"] =  np.array(obj['observations'][800:, :, :])
# data_dict["val_u"] = np.array(obj['u'][800:, :-1, None])

# def normalise(t, t_min, t_max):

#     return (t - t_min) / (t_max - t_min) - 0.5

# def unnormalise(t, t_min, t_max):

#     return t_min + t * (t_max - t_min) - 0.5

# data_dict["train_data"] = normalise(data_dict["train_data"], np.min(data_dict["train_data"], axis = (0, 1)), np.max(data_dict["train_data"], axis = (0, 1)))
# data_dict["train_u"] = normalise(data_dict["train_u"], np.min(data_dict["train_u"], axis = (0, 1)), np.max(data_dict["train_u"], axis = (0, 1)))
# data_dict["val_data"] = normalise(data_dict["val_data"], np.min(data_dict["val_data"], axis = (0, 1)), np.max(data_dict["val_data"], axis = (0, 1)))
# data_dict["val_u"] = normalise(data_dict["val_u"], np.min(data_dict["val_u"], axis = (0, 1)), np.max(data_dict["val_u"], axis = (0, 1)))

# n_timepoints = 100
# z_dim = 3
# y_dim = 3

# m1 = params['prior_params']['m1']
# Q1 = lie_params_to_constrained(params['prior_params']['Q1'], z_dim)
# # Q1 = np.diag(np.exp(all_results[0][1]['prior_params']['Q1']))
# A = construct_dynamics_matrix(params['prior_params']["A_u"], params['prior_params']["A_v"], params['prior_params']["A_s"], z_dim)
# B = scale_matrix_by_norm(params['prior_params']['B'])
# Q = lie_params_to_constrained(params['prior_params']['Q'], z_dim)
# # Q = np.diag(np.exp(all_results[0][1]['prior_params']['Q']))

# from matplotlib import pyplot as plt

# episode = 0
# y = data_dict["train_data"][episode, :, :]
# control = data_dict["train_u"][episode, :]
# m = np.zeros((z_dim, n_timepoints))
# m[:, 0] = m1
# P = np.zeros((z_dim, z_dim, n_timepoints))
# P[:, :, 0] = Q1
# y_recon = np.zeros((n_timepoints, y_dim))
# observations_present = True
# for t in range(n_timepoints - 1):
#     if observations_present:
#         rec = all_models[0]['model'].recognition.apply(params['rec_params'], y[t, :])
#         e = rec['mu'] - m[:, t]
#         S = P[:, :, t] + rec['Sigma']
#         K = np.linalg.solve(S, P[:, :, t]).T
#         m[:, t] += K @ e
#         P[:, :, t] = (np.eye(z_dim) - K) @ P[:, :, t]
#     y_recon[t, :] = all_models[0]['model'].decoder.apply(params['dec_params'], m[:, t]).mean() # .covariance()
#     m[:, t + 1] = A @ m[:, t] + B @ control[t]
#     P[:, :, t + 1] = A @ P[:, :, t] @ A.T + Q
# plt.figure()
# plt.plot(y, 'r')
# plt.plot(y_recon, 'b--')
# plt.show(block = False)

# breakpoint()

################################################################################################

from svae.utils import lie_params_to_constrained, construct_dynamics_matrix
import numpy as np

z_dim = 3
m1 = params['prior_params']['m1']
Q1 = lie_params_to_constrained(params['prior_params']['Q1'], z_dim)
A = construct_dynamics_matrix(params['prior_params']["A_u"], params['prior_params']["A_v"], params['prior_params']["A_s"], z_dim)
B = params['prior_params']['B']
Q = lie_params_to_constrained(params['prior_params']['Q'], z_dim)

u_dim = 1
A_aug = np.block([[A, np.zeros((z_dim, z_dim))],[np.zeros((z_dim, z_dim)), np.eye(z_dim)]])
B_aug = np.concatenate((B, 1e-4 * np.ones((z_dim, u_dim))))
D = np.zeros((z_dim, z_dim * 2))
for i in range(z_dim):
    for j in range(z_dim * 2):
        if j == i:
            D[i, j] = 1
        elif j == i + z_dim:
            D[i, j] = -1
Q_aug = 1e2 * D.T @ D
R = 1 * np.eye(u_dim)

import control
K, S, E = control.dlqr(A_aug, B_aug, Q_aug, R)

################################################################################################

import numpy as np
from svae.utils import lie_params_to_constrained, construct_dynamics_matrix, scale_matrix_by_norm
import pickle

obj = pickle.load(open("pendulum_data.pkl", 'rb'))
data_dict = {}
data_dict["train_data"] = np.array(obj['observations'][:800, :, :])
data_dict["train_u"] = np.array(obj['u'][:800, :-1, None])
data_dict["val_data"] =  np.array(obj['observations'][800:, :, :])
data_dict["val_u"] = np.array(obj['u'][800:, :-1, None])

def normalise(t, t_min, t_max):

    return (t - t_min) / (t_max - t_min) - 0.5

def unnormalise(t, t_min, t_max):

    return t_min + (t + 0.5) * (t_max - t_min)

# data_dict["train_data"] = normalise(data_dict["train_data"], np.min(data_dict["train_data"], axis = (0, 1)), np.max(data_dict["train_data"], axis = (0, 1)))
# data_dict["train_u"] = normalise(data_dict["train_u"], np.min(data_dict["train_u"], axis = (0, 1)), np.max(data_dict["train_u"], axis = (0, 1)))
# data_dict["val_data"] = normalise(data_dict["val_data"], np.min(data_dict["val_data"], axis = (0, 1)), np.max(data_dict["val_data"], axis = (0, 1)))
# data_dict["val_u"] = normalise(data_dict["val_u"], np.min(data_dict["val_u"], axis = (0, 1)), np.max(data_dict["val_u"], axis = (0, 1)))

################################################################################################

# import scipy
# scipy.linalg.solve_discrete_are(A, B, Q, R)

import gym
import numpy as np

env = gym.make('Pendulum-v0')

def filter_observation(models, params, m, P, y):

    rec = models.recognition.apply(params['rec_params'], y)
    e = rec['mu'] - m
    S = P + rec['Sigma']
    K = np.linalg.solve(S, P).T
    m += K @ e
    P = (np.eye(m.size) - K) @ P

    return m, P

def get_control(m, K, z_goal):

    u = - K @ np.concatenate((m, z_goal))

    return u

def predict_next_state(m, P, A, B, u, Q):

    m = A @ m + B @ u
    P = A @ P @ A.T + Q

    return m, P

n_rollouts = 10
models = all_results[0][0]
norm_goal = normalise(np.array([1, 0, 0]), np.min(data_dict["train_data"], axis = (0, 1)), np.max(data_dict["train_data"], axis = (0, 1)))
z_goal_dist = models.recognition.apply(params['rec_params'], norm_goal) # theta and theta dot 0; x = cos(theta), y = sin(theta)
decoded_z_goal = models.decoder.apply(params['dec_params'], z_goal_dist['mu']) # theta and theta dot 0; x = cos(theta), y = sin(theta)
print(decoded_z_goal.mean())
print(unnormalise(decoded_z_goal.mean(), np.min(data_dict["train_data"], axis = (0, 1)), np.max(data_dict["train_data"], axis = (0, 1))))
breakpoint()
n_timepoints = 100
m = np.zeros((n_rollouts, z_dim, n_timepoints))
m[:, :, 0] = m1[None].repeat(n_rollouts, axis = 0)
P = np.zeros((n_rollouts, z_dim, z_dim, n_timepoints))
P[:, :, :, 0] = Q1[None].repeat(n_rollouts, axis = 0)
u = np.zeros((n_rollouts, n_timepoints, 1))
action = np.zeros((n_rollouts, n_timepoints, 1))
for r in range(n_rollouts):
    obs = env.reset()
    for t in range(n_timepoints - 1):
        y = normalise(obs, np.min(data_dict["train_data"], axis = (0, 1)), np.max(data_dict["train_data"], axis = (0, 1)))
        m[r, :, t], P[r, :, :, t] = filter_observation(models, params, m[r, :, t], P[r, :, :, t], y)
        u[r, t] = get_control(m[r, :, t], K, z_goal_dist['mu'])
        m[r, :, t + 1], P[r, :, :, t + 1] = predict_next_state(m[r, :, t], P[r, :, :, t], A, B, u[r, t], Q)
        action[r, t] = unnormalise(u[r, t], np.min(data_dict["train_u"], axis = (0, 1)), np.max(data_dict["train_u"], axis = (0, 1)))
        obs, reward, done, info = env.step(action[r, t])
        env.render("human")

breakpoint()

n_rollouts = 1000
n_timepoints = 100
u = np.zeros((n_rollouts, n_timepoints))
observations = np.zeros((n_rollouts, n_timepoints, 3))
for r in range(n_rollouts):
    obs = env.reset()
    for t in range(n_timepoints):
        observations[r, t, :] = obs
        action, _state = model.predict(obs, deterministic=True)

        obs, reward, done, info = env.step(action)
        u[r, t] = action[0]
        vec_env.render("human")

# import wandb
# model = wandb.restore("parameters.pkl", run_path="james-heald/SVAE-LDS-ICML-RE-1/noble-resonance-5")
# model = wandb.restore("parameters.pkl", run_path="/Users/james/Dropbox\ \(UCL\)/ucgtjhe@ucl.ac.uk\’s\ files/James\ MacBook/Gatsby/svae/runs/run-20231223_185238-3s3vesaa/files")
# model = wandb.restore("parameters.pkl", run_path="/Users/james/Dropbox (UCL)/ucgtjhe@ucl.ac.uk’s files/James MacBook/Gatsby/svae/runs/run-20231223_185238-3s3vesaa/files")
# def load_object_using_pickle(path, filename):
# file_name = "/Users/james/Dropbox (UCL)/ucgtjhe@ucl.ac.uk’s files/James MacBook/Gatsby/svae/runs/run-20231223_185238-3s3vesaa/files/parameters.pkl"
# with open(file_name, 'rb') as file: 
#     model = wandb.restore(file_name, run_path="/Users/james/Dropbox (UCL)/ucgtjhe@ucl.ac.uk’s files/James MacBook/Gatsby/svae/runs/run-20231223_185238-3s3vesaa/files")
#     return obj

import numpy as np
from utils import lie_params_to_constrained
import pickle

obj = pickle.load(open("pendulum_data.pkl", 'rb'))
data_dict = {}
data_dict["train_data"] = np.array(obj['observations'][:800, :, :])
data_dict["train_u"] = np.array(obj['u'][:800, :-1, None])
data_dict["val_data"] =  np.array(obj['observations'][800:, :, :])
data_dict["val_u"] = np.array(obj['u'][800:, :-1, None])

def normalise(t, t_min, t_max):

    return (t - t_min) / (t_max - t_min)

def unnormalise(t, t_min, t_max):

    return t_min + t * (t_max - t_min)

episode = 0
n_timepoints = 100
z_dim = 3
y_dim = 3
m = np.zeros((z_dim, n_timepoints))
m[:, 0] = m1
P = np.zeros((z_dim, z_dim, n_timepoints))
P[:, :, 0] = Q1
y_recon = np.zeros((y_dim, n_timepoints))

m1 = all_results[0][1]['prior_params']['m1']
Q1 = lie_params_to_constrained(all_results[0][1]['prior_params']['Q1'], z_dim)
A = all_results[0][1]['prior_params']['A']
B = all_results[0][1]['prior_params']['B']
Q = lie_params_to_constrained(all_results[0][1]['prior_params']['Q'], z_dim)

y = data_dict["train_data"][episode, :, :]
control = data_dict["train_u"][episode, :]

obs = env.reset()
for t in range(n_timepoints):
    rec = all_models[0]['model'].recognition.apply(all_results[0][1]['rec_params'], y[0])
    e = rec['mu'] - m[:, t]
    S = P[:, :, t] + rec['Sigma']
    K = np.linalg.solve(S, P[:, :, t]).T
    m[:, t] += K @ e
    P[:, :, t] = (np.eye(z_dim) - K) @ P[:, :, t]
    y_recon[:, t] = all_models[0]['model'].decoder.apply(all_results[0][1]['dec_params'], m[:, t]).mean() # .covariance()
    m[:, t + 1] = A @ m[:, t] + B @ control[t]
    P[:, :, t + 1] = A @ P[:, :, t] @ A.T + Q



import numpy as np

space_dim = 2
state_dim = space_dim * 3
u_dim = np.copy(space_dim)
dt = 0.02

A = np.eye(state_dim)
A[0,2] = dt
A[1,3] = dt

B = np.zeros((state_dim, u_dim))
B[2,0] = dt
B[3,1] = dt

z_dim = 3
u_dim = 1

A_aug = np.zeros((z_dim * 2, z_dim * 2))
A_aug[:z_dim, :z_dim] = A
A_aug[z_dim:, z_dim:] = np.eye(z_dim)

B = np.zeros((z_dim * 2, u_dim))
B[:z_dim, :] = B
B[z_dim:, :] = 0

D = np.zeros((z_dim, z_dim * 2))
for i in range(z_dim):
    for j in range(z_dim * 2):
        if j == i:
            D[i, j] = 1
        elif j == i + z_dim * 2:
            D[i, j] = -1
Q = 1e4 * D.T @ D
R = 1 * np.eye(u_dim)


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
#   action, _state = model.predict(obs, deterministic=True)
#   obs, reward, done, info = vec_env.step(action)
#   vec_env.render("human")

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