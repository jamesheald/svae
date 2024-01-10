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
config.update("jax_enable_x64", True)
config.update("jax_debug_nans", True)
config.update("jax_disable_jit", True)

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
    "train_batch_size": 800,
    "val_batch_size": 200,
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
    "save_dir": "/nfs/nhome/live/jheald/svae/runs/first_attempt",
    "reload_dir": "/nfs/nhome/live/jheald/svae/runs/first_attempt",
    "reload_state": False,
    "project_name": "SVAE-control-pendulum",
    "run_type": "model_learning" # model_learning
}
# "save_dir": "/Users/james/Dropbox (UCL)/ucgtjhe@ucl.ac.uk’s files/James MacBook/Gatsby/svae/runs/first_attempt",
# "reload_dir": "/Users/james/Dropbox (UCL)/ucgtjhe@ucl.ac.uk’s files/James MacBook/Gatsby/svae/runs/first_attempt",

all_results, all_models = run_pendulum_control(run_params)
# all_results[0]: trainer.model, trainer.params, trainer.train_losses, trainer.val_losses, trainer.opts, trainer.opt_states, trainer.ckptrs

# all_results[0][0] # model deepLDS object (? == all_models[0]['model'])
# all_results[0][1] # params dict_keys(['dec_params', 'post_params', 'post_samples', 'prior_params', 'rec_params']), e.g. all_results[0][1]['prior_params'] 
# all_results[0][2] # train_losses list, 
# all_models[0]['model'] # deepLDS object
# all_models[0]['trainer'] # Trainer object
# all_models[0]['model'].prior.get_dynamics_params()

from matplotlib import pyplot as plt
plt.plot(all_results[0][2] ,'r') # train loss
plt.plot(all_results[0][3], 'b') # val loss
plt.show(block = False)

run_params['run_type'] = 'none'
run_params["reload_state"] = True
all_results, all_models = run_pendulum_control(run_params)

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

    return t_min + t * (t_max - t_min) - 0.5

data_dict["train_data"] = normalise(data_dict["train_data"], np.min(data_dict["train_data"], axis = (0, 1)), np.max(data_dict["train_data"], axis = (0, 1)))
data_dict["train_u"] = normalise(data_dict["train_u"], np.min(data_dict["train_u"], axis = (0, 1)), np.max(data_dict["train_u"], axis = (0, 1)))
data_dict["val_data"] = normalise(data_dict["val_data"], np.min(data_dict["val_data"], axis = (0, 1)), np.max(data_dict["val_data"], axis = (0, 1)))
data_dict["val_u"] = normalise(data_dict["val_u"], np.min(data_dict["val_u"], axis = (0, 1)), np.max(data_dict["val_u"], axis = (0, 1)))

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

episode = 0
y = data_dict["train_data"][episode, :, :]
control = data_dict["train_u"][episode, :]
m = np.zeros((z_dim, n_timepoints))
m[:, 0] = m1
P = np.zeros((z_dim, z_dim, n_timepoints))
P[:, :, 0] = Q1
y_recon = np.zeros((n_timepoints, y_dim))
observations_present = True
for t in range(n_timepoints - 1):
    if observations_present:
        rec = all_models[0]['model'].recognition.apply(all_results[0][1]['rec_params'], y[t, :])
        e = rec['mu'] - m[:, t]
        S = P[:, :, t] + rec['Sigma']
        K = np.linalg.solve(S, P[:, :, t]).T
        m[:, t] += K @ e
        P[:, :, t] = (np.eye(z_dim) - K) @ P[:, :, t]
    y_recon[t, :] = all_models[0]['model'].decoder.apply(all_results[0][1]['dec_params'], m[:, t]).mean() # .covariance()
    m[:, t + 1] = A @ m[:, t] + B @ control[t]
    P[:, :, t + 1] = A @ P[:, :, t] @ A.T + Q
plt.figure()
plt.plot(y, 'r')
plt.plot(y_recon, 'b--')
plt.show(block = False)

episode = 0
y = data_dict["train_data"][episode, :, :]
control = data_dict["train_u"][episode, :]
m = np.zeros((z_dim, n_timepoints))
m[:, 0] = m1
P = np.zeros((z_dim, z_dim, n_timepoints))
P[:, :, 0] = Q1
y_recon = np.zeros((n_timepoints, y_dim))
observations_present = False
for t in range(n_timepoints - 1):
    if observations_present:
        rec = all_models[0]['model'].recognition.apply(all_results[0][1]['rec_params'], y[t, :])
        e = rec['mu'] - m[:, t]
        S = P[:, :, t] + rec['Sigma']
        K = np.linalg.solve(S, P[:, :, t]).T
        m[:, t] += K @ e
        P[:, :, t] = (np.eye(z_dim) - K) @ P[:, :, t]
    y_recon[t, :] = all_models[0]['model'].decoder.apply(all_results[0][1]['dec_params'], m[:, t]).mean() # .covariance()
    m[:, t + 1] = A @ m[:, t] + B @ control[t]
    P[:, :, t + 1] = A @ P[:, :, t] @ A.T + Q
plt.figure()
plt.plot(y, 'r')
plt.plot(y_recon, 'b--')
plt.show(block = False)

breakpoint()