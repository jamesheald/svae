###############
# compare performance with f_twiddles removed (set to 1)
# change dynamics model to A x + B u + b for added expressivity
# in rpm, do i want to compute marginal priors using specific u in each episode, if not i would need to define prior on u and integrate u out
# change rec/dec architecture so one NN for both mean and logvar, not 2. also consider whether you want diag cov for both rec and dec?
# reparameterization_type was set to reparameterization.NOT_REPARAMETERIZED?!
# itr passed via partial to loss function - surely wrong?
# kalman smoothing is done by passing mean of emission potential as observation - weird?! line 249 distributions
# in distribution.py event_dims will not generate to nonscalar controls: input_matrix=tfp.internal.parameter_properties.ParameterProperties(event_ndims=1),
# lots of non pure functions!! e.g. _generic_sampling_element
# dynamics matrix minus eye for stability hacky

from jax import config
# config.update("jax_enable_x64", True)
# config.update("jax_debug_nans", True)
# config.update("jax_disable_jit", True)

from svae.experiments import run_pendulum_control
import jax.random as jr

run_params = {
    "inference_method": "rpm", # "svae"
    "latent_dims": 3,
    "emission_dims": 3,
    "input_dims": 1,
    "rec_trunk_features": [512],
    "rec_head_mean_features": [],
    "rec_head_var_features": [],
    "rec_diagonal_covariance" : False,
    # "dec_features": [512],
    "seed": jr.PRNGKey(1),
    "sample_kl": False,
    "use_parallel_kf": True,
    "train_size": 20, # 800
    "val_size": 5, # 200
    "train_batch_size": 20,
    "val_batch_size": 5,
    "num_timesteps": 100,
    "mask_size": 0,
    "mask_start": 0,
    "base_lr": .001,
    "lr_decay": False, # definitely won't want this on when I am learning online
    "prior_base_lr": .01,
    "prior_lr_warmup": False,
    "max_grad_norm": 10,
    "weight_decay": 0.0000, # 0.0001
    "max_iters": 40000,
    "beta_transition_begin": 1000,
    "beta_transition_steps": 1000,
    "early_stop_start": 2000, # start after beta reaches 1
    "min_delta": 1e-3,
    "patience": 10, # 10
    "checkpoint_every_n_epochs": 5, # 10
    "log_to_wandb": False,
    "log_every_n_epochs": 10,
    # "save_dir": "/nfs/nhome/live/jheald/svae/runs/first_attempt",
    # "reload_dir": "/nfs/nhome/live/jheald/svae/runs/first_attempt",
    "save_dir": "/Users/james/Dropbox (UCL)/ucgtjhe@ucl.ac.uk’s files/James MacBook/Gatsby/svae/runs/first_attempt",
    "reload_dir": "/Users/james/Dropbox (UCL)/ucgtjhe@ucl.ac.uk’s files/James MacBook/Gatsby/svae/runs/first_attempt",
    "reload_state": False,
    "project_name": "SVAE-control-pendulum",
    "run_type": "model_learning" # model_learning
}

# all_results, all_models = run_pendulum_control(run_params)
# all_results[0]: trainer.model, trainer.params, trainer.train_losses, trainer.val_losses, trainer.opts, trainer.opt_states, trainer.ckptrs

# all_results[0][0] # model deepLDS object (? == all_models[0]['model'])
# all_results[0][1] # params dict_keys(['dec_params', 'post_params', 'post_samples', 'prior_params', 'rec_params']), e.g. all_results[0][1]['prior_params'] 
# all_results[0][2] # train_losses list, 
# all_models[0]['model'] # deepLDS object
# all_models[0]['trainer'] # Trainer object
# all_models[0]['model'].prior.get_dynamics_params()

# breakpoint()

from matplotlib import pyplot as plt
# plt.plot(all_results[0][2] ,'r') # train loss
# plt.plot(all_results[0][3], 'b') # val loss
# plt.show(block = False)

run_params['run_type'] = 'none'
run_params["reload_state"] = True
all_results, all_models = run_pendulum_control(run_params)

import numpy as np
from svae.utils import lie_params_to_constrained, construct_dynamics_matrix, scale_matrix_by_norm
from svae.datasets import load_pendulum_control_data

def normalise(t, t_min, t_max):

    return (t - t_min) / (t_max - t_min) - 0.5

def unnormalise(t, t_min, t_max):

    return t_min + t * (t_max - t_min) - 0.5

import pickle
obj = pickle.load(open("pendulum_data.pkl", 'rb'))
min_ob = np.min(obj['observations'], axis = (0, 1))
max_ob = np.max(obj['observations'], axis = (0, 1))
min_u = np.min(obj['u'], axis = (0, 1))
max_u = np.max(obj['u'], axis = (0, 1))

data_dict = load_pendulum_control_data(run_params)

n_timepoints = 100
z_dim = 3
y_dim = 3

m1 = all_results[0][1]['prior_params']['m1']
Q1 = lie_params_to_constrained(all_results[0][1]['prior_params']['Q1'], z_dim)
# Q1 = np.diag(np.exp(all_results[0][1]['prior_params']['Q1']))
A = construct_dynamics_matrix(all_results[0][1]['prior_params']["A_u"], all_results[0][1]['prior_params']["A_v"], all_results[0][1]['prior_params']["A_s"], z_dim)
B = scale_matrix_by_norm(all_results[0][1]['prior_params']['B'])
Q = lie_params_to_constrained(all_results[0][1]['prior_params']['Q'], z_dim)
# Q = np.diag(np.exp(all_results[0][1]['prior_params']['Q']))

normalised_goal = normalise(np.array([1., 0., 0.]), min_ob, max_ob)
RPM_goal = all_models[0]['model'].recognition.apply(all_results[0][1]['rec_params'], normalised_goal) # obs are [cos(theta), sin(theta), theta_dot], where theta = 0 is upright (the goal)
# compute optimal feedback gain matrix K
prior_params = all_models[0]['model'].prior.get_dynamics_params(all_results[0][1]['prior_params'])
import copy
p = copy.deepcopy(prior_params)
latent_dims = 3 ######## TO CHANGE
u_dims = 1 ######## TO CHANGE
Q_lqr = np.eye(latent_dims) ######## TO CHANGE
R_lqr = np.eye(u_dims) * 1e-3 ######## TO CHANGE
x_goal = (np.linalg.solve(p["A"] - np.eye(latent_dims), p["B"])).squeeze()
x_goal /= np.linalg.norm(x_goal)
x_goal *= p["goal_norm"] ######## don't make goal unit norm away from origin
(u_eq, _, _, _) = np.linalg.lstsq(p["B"], (np.eye(latent_dims) - p["A"]) @ x_goal)
# shift the mean/precision-weighted mean of all RPM potentials so that the mean of the inferred hidden state for the goal is at x_goal
delta_mu = x_goal - RPM_goal['mu']

def filter_observation(models, params, m, P, y, delta_mu):

    rec = models.recognition.apply(params['rec_params'], y)
    e = rec['mu'] + delta_mu - m
    S = P + rec['Sigma']
    K = np.linalg.solve(S, P).T
    m += K @ e
    P = (np.eye(m.size) - K) @ P

    return m, P

def get_control(m, K):

    u = - K @ m

    return u

def predict_next_state(m, P, A, B, u, Q):

    m = A @ m + B @ u
    P = A @ P @ A.T + Q

    return m, P

n_rollouts = 1
m = np.zeros((n_rollouts, z_dim, n_timepoints))
P = np.zeros((n_rollouts, z_dim, z_dim, n_timepoints))
m[:, :, 0] = m1[None].repeat(n_rollouts, axis = 0)
P[:, :, :, 0] = Q1[None].repeat(n_rollouts, axis = 0)
if run_params["inference_method"] != "rpm":
    y_recon = np.zeros((n_rollouts, n_timepoints, y_dim))
observations_present = True
for r in range(n_rollouts):
    y = data_dict["train_data"][r, :, :]
    u = data_dict["train_u"][r, :]
    for t in range(n_timepoints - 1):
        if observations_present:
            obs = normalise(y[t, :], min_ob, max_ob)
            m[r, :, t], P[r, :, :, t] = filter_observation(all_models[0]['model'], all_results[0][1], m[r, :, t], P[r, :, :, t], obs, delta_mu)
        if run_params["inference_method"] != "rpm":
            y_recon[t, :] = all_models[0]['model'].decoder.apply(all_results[0][1]['dec_params'], m[:, t]).mean() # .covariance()
        action = normalise(u[r, t][None], min_u, max_u)
        m[r, :, t + 1], P[r, :, :, t + 1] = predict_next_state(m[r, :, t], P[r, :, :, t], A, B, action, Q)
plt.figure()
plt.plot(y, 'r')
plt.plot(m[0, :, :].T, 'g--')
if run_params["inference_method"] != "rpm":
    plt.plot(y_recon, 'b--')
plt.show(block = False)

n_rollouts = 1
m = np.zeros((n_rollouts, z_dim, n_timepoints))
P = np.zeros((n_rollouts, z_dim, z_dim, n_timepoints))
m[:, :, 0] = m1[None].repeat(n_rollouts, axis = 0)
P[:, :, :, 0] = Q1[None].repeat(n_rollouts, axis = 0)
if run_params["inference_method"] != "rpm":
    y_recon = np.zeros((n_rollouts, n_timepoints, y_dim))
observations_present = False
for r in range(n_rollouts):
    y = data_dict["train_data"][r, :, :]
    u = data_dict["train_u"][r, :]
    for t in range(n_timepoints - 1):
        if observations_present:
            obs = normalise(y[t, :], min_ob, max_ob)
            m[r, :, t], P[r, :, :, t] = filter_observation(all_models[0]['model'], all_results[0][1], m[r, :, t], P[r, :, :, t], obs, delta_mu)
        if run_params["inference_method"] != "rpm":
            y_recon[t, :] = all_models[0]['model'].decoder.apply(all_results[0][1]['dec_params'], m[:, t]).mean() # .covariance()
        action = normalise(u[r, t][None], min_u, max_u)
        m[r, :, t + 1], P[r, :, :, t + 1] = predict_next_state(m[r, :, t], P[r, :, :, t], A, B, action, Q)
plt.figure()
plt.plot(y, 'r')
plt.plot(m[0, :, :].T, 'g--')
if run_params["inference_method"] != "rpm":
    plt.plot(y_recon, 'b--')
plt.show(block = False)

breakpoint()

import gym
from jax import numpy as np
from jax.lax import scan

env = gym.make('Pendulum-v0')

x_goal

def get_previous_P(carry, inputs):

    P, A, B, Q, R = carry

    prev_P = Q + A.T @ P @ A - (A.T @ P @ B) @ np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)

    carry = prev_P, A, B, Q, R
    outputs = None

    return carry, outputs

def get_optimal_feedback_gain(A, B, Q, R):

    carry = Q, A, B, Q, R
    (P, _, _, _, _), _ = scan(get_previous_P, carry, None, length=100)
    K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)

    return K

latent_dims = 3 ######## TO CHANGE
u_dims = 1 ######## TO CHANGE
Q_lqr = np.eye(latent_dims) ######## TO CHANGE
R_lqr = np.eye(u_dims) * 1e-3 ######## TO CHANGE
K = get_optimal_feedback_gain(A, B, Q_lqr, R_lqr)

import numpy as np

n_rollouts = 1
m = np.zeros((n_rollouts, z_dim, n_timepoints))
P = np.zeros((n_rollouts, z_dim, z_dim, n_timepoints))
u = np.zeros((n_rollouts, n_timepoints, 1))
m[:, :, 0] = m1[None].repeat(n_rollouts, axis = 0)
P[:, :, :, 0] = Q1[None].repeat(n_rollouts, axis = 0)
if run_params["inference_method"] != "rpm":
    y_recon = np.zeros((n_rollouts, n_timepoints, y_dim))
observations_present = True
for r in range(n_rollouts):
    obs = env.reset()
    for t in range(n_timepoints - 1):
        if observations_present:
            obs = normalise(obs, min_ob, max_ob)
            m[r, :, t], P[r, :, :, t] = filter_observation(all_models[0]['model'], all_results[0][1], m[r, :, t], P[r, :, :, t], obs, delta_mu)
        u[r, t] = get_control(m[r, :, t], K)
        if run_params["inference_method"] != "rpm":
            y_recon[t, :] = all_models[0]['model'].decoder.apply(all_results[0][1]['dec_params'], m[:, t]).mean() # .covariance()
        m[r, :, t + 1], P[r, :, :, t + 1] = predict_next_state(m[r, :, t], P[r, :, :, t], A, B, u[r, t], Q)
        action = unnormalise(u[r, t], min_u, max_u)
        obs, reward, done, info = env.step(action)
        env.render("human")
# plt.figure()
# plt.plot(y, 'r')
# plt.plot(m[0, :, :].T, 'g--')
# if run_params["inference_method"] != "rpm":
#     plt.plot(y_recon, 'b--')
# plt.show(block = False)

breakpoint()