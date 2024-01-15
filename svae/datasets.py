from jax import vmap
import jax.numpy as np
from jax.numpy.linalg import solve
import jax.random as jr

from flax.core import frozen_dict as fd

import tensorflow_probability.substrates.jax.distributions as tfd

from svae.utils import random_rotation, get_scaler
from svae.priors import LinearGaussianChainPrior
from svae.posteriors import LDSSVAEPosterior

# TODO: get rid of this and its dependencies
data_dict = {}

# Takes a linear Gaussian chain as its base
class LDS(LinearGaussianChainPrior):
    def __init__(self, latent_dims, input_dims, seq_len, base=None, posterior=None):
        super().__init__(latent_dims, seq_len)
        self.posterior = posterior or LDSSVAEPosterior(latent_dims, input_dims, seq_len)
        self.base = base or LinearGaussianChainPrior(latent_dims, seq_len) # Slightly redundant...

    # Takes unconstrained params
    def sample(self, params, u, shape, key):
        latents = self.base.sample(params, u, shape, key)
        sample_shape = latents.shape[:-1]
        key, _ = jr.split(key)
        C, d, R = params["C"], params["d"], params["R"]
        obs_noise = tfd.MultivariateNormalFullCovariance(loc=d, covariance_matrix=R)\
            .sample(sample_shape=sample_shape, seed=key)
        obs = np.einsum("ij,...tj->...ti", C, latents) + obs_noise
        return latents, obs

    # Should work with any batch dimension
    def log_prob(self, params, u, states, data):
        latent_dist = self.base.distribution(self.base.get_constrained_params(params, u))
        latent_ll = latent_dist.log_prob(states)
        C, d, R = params["C"], params["d"], params["R"]
        # Gets around batch dimensions
        noise = tfd.MultivariateNormalFullCovariance(loc=d, covariance_matrix=R)
        obs_ll = noise.log_prob(data - np.einsum("ij,...tj->...ti", C, states))
        return latent_ll + obs_ll.sum(axis=-1)

    # Assumes single data points
    def e_step(self, params, u, data):
        # Shorthand names for parameters
        C, d, R = params["C"], params["d"], params["R"]

        J = np.dot(C.T, np.linalg.solve(R, C))
        J = np.tile(J[None, :, :], (self.seq_len, 1, 1))
        # linear potential
        h = np.dot(data - d, np.linalg.solve(R, C))

        Sigma = solve(J, np.eye(self.latent_dims)[None])
        mu = vmap(solve)(J, h)

        return self.posterior.infer(self.base.get_constrained_params(params, u), {"mu": mu, "Sigma": Sigma}, u)
        
    # Also assumes single data points
    def marginal_log_likelihood(self, params, u, data):
        posterior = self.posterior.distribution(self.e_step(params, u, data))
        states = posterior.mean
        joint_ll = self.log_prob(params, u, states, data)
        posterior_ll = posterior.log_prob(states, u=u)
        # This is numerically unstable!
        lps = joint_ll - posterior_ll # log p(x, y) - log p(x|y) = log p(x) + log p(y|x) - log p(x|y) = log p(y)
        return lps

def sample_lds_dataset(run_params):    
    d = run_params["dataset_params"]
    
    global data_dict
    if data_dict is not None \
        and "dataset_params" in data_dict \
        and str(data_dict["dataset_params"]) == str(fd.freeze(d)):
        print("Using existing data.")
        print("Data MLL: ", data_dict["marginal_log_likelihood"])        
        return data_dict

    data_dict = {}

    seed = d["seed"]
    emission_dims = d["emission_dims"]
    latent_dims = d["latent_dims"]
    input_dims = d["input_dims"]
    emission_cov = d["emission_cov"]
    dynamics_cov = d["dynamics_cov"]
    num_timesteps = d["num_timesteps"]
    num_trials = d["num_trials"]
    seed_m1, seed_C, seed_d, seed_A, seed_B, seed_u, seed_sample = jr.split(seed, 7)

    R = emission_cov * np.eye(emission_dims)
    Q = dynamics_cov * np.eye(latent_dims)
    C = jr.normal(seed_C, shape=(emission_dims, latent_dims))
    d = jr.normal(seed_d, shape=(emission_dims,))
    B = jr.normal(seed_B, shape=(latent_dims, input_dims))

    # Here we let Q1 = Q
    lds = LDS(latent_dims, input_dims, num_timesteps)
    
    params = {
            "m1": jr.normal(key=seed_m1, shape=(latent_dims,)),
            "Q1": Q,
            # "m1": np.ones((latent_dims,)),
            # "Q1": 0.01 * np.eye(latent_dims),
            "Q": Q,
            "A": random_rotation(seed_A, latent_dims, theta=np.pi/20),
            "B": B,
            "R": R,
            "C": C,
            "d": d,
        }

    # sinusoidal controls with random phase
    u_keys = jr.split(seed_u, num_trials * input_dims).reshape(num_trials, input_dims, 2)
    u = vmap(vmap(lambda T, key: np.cos(np.linspace(0, 1, T) * 2 * np.pi + jr.uniform(key) * 2 * np.pi), in_axes = (None,0)), in_axes = (None,0))(num_timesteps, u_keys)
    u = u.transpose(0,2,1) # num_trials x num_timesteps x input_dims

    # constrained = lds.get_constrained_params

    # params["avg_suff_stats"] = { "Ex": constrained["Ex"], 
                                # "ExxT": constrained["ExxT"], 
                                # "ExnxT": constrained["ExnxT"] } # JHmod

    states, data = vmap(lambda u, key: lds.sample(params, u = u, shape=(), key=key))(u, jr.split(seed_sample, num_trials))
    
    mll = vmap(lds.marginal_log_likelihood, in_axes=(None, 0, 0))(params, u, data)
    mll = np.sum(mll) / data.size
    print("Data MLL: ", mll)
    
    seed_val, seed_u = jr.split(seed_sample)

    # sinusoidal controls with random phase
    u_keys = jr.split(seed_u, num_trials * input_dims).reshape(num_trials, input_dims, 2)
    val_u = vmap(vmap(lambda T, key: np.cos(np.linspace(0, 1, T) * 2 * np.pi + jr.uniform(key) * 2 * np.pi), in_axes = (None,0)), in_axes = (None,0))(num_timesteps, u_keys)
    val_u = val_u.transpose(0,2,1) # num_trials x num_timesteps x input_dims

    # val_states, val_data = lds.sample(params, val_u,
    #                           shape=(num_trials,), 
    #                           key=seed_val)
    val_states, val_data = vmap(lambda u, key: lds.sample(params, u = u, shape=(), key=key))(val_u, jr.split(seed_val, num_trials))

    scaler_obs = get_scaler('standard')
    scaler_u = get_scaler('standard')
    # data = np.concatenate((data, np.arange(num_timesteps)[None, :, None].repeat(num_trials, axis=0)),axis=2) # concatenate time to data
    # val_data = np.concatenate((val_data, np.arange(num_timesteps)[None, :, None].repeat(num_trials, axis=0)),axis=2) # concatenate time to data
    scaled_obs = scaler_obs.fit_transform(np.vstack((data.reshape(-1, data.shape[-1]), val_data.reshape(-1, val_data.shape[-1]))))
    scaled_u = scaler_u.fit_transform(np.vstack((u.reshape(-1, u.shape[-1]), val_u.reshape(-1, val_u.shape[-1]))))

    data_dict["generative_model"] = lds
    data_dict["marginal_log_likelihood"] = mll
    data_dict["train_data"] = scaled_obs[:data[:,:,0].size].reshape(num_trials, num_timesteps, -1)
    data_dict["train_u"] = scaled_u[:u[:,:,0].size].reshape(num_trials, num_timesteps, -1)
    data_dict["train_states"] = states
    data_dict["val_data"] = scaled_obs[data[:,:,0].size:].reshape(num_trials, num_timesteps, -1)
    data_dict["val_u"] = scaled_u[u[:,:,0].size:].reshape(num_trials, num_timesteps, -1)
    data_dict["val_states"] = val_states
    data_dict["dataset_params"] = fd.freeze(run_params["dataset_params"])
    data_dict["lds_params"] = params
    
    data_dict["scaled_goal"] = np.array([1., 0., 0.]) # to stop an error being thrown as code expects this
    data_dict['scaler_obs'] = scaler_obs
    data_dict['scaler_u'] = scaler_u

    return data_dict

def load_pendulum_control_data(run_params):

    import pickle
    obj = pickle.load(open("pendulum_data.pkl", 'rb'))

    scaler_obs = get_scaler('standard')
    scaler_u = get_scaler('standard')
    obj['u'] = obj['u'][:, :, None]
    assert obj['observations'].ndim == obj['u'].ndim == 3
    obs = scaler_obs.fit_transform(obj['observations'].reshape(-1, obj['observations'].shape[-1])).reshape(obj['observations'].shape)
    u = scaler_u.fit_transform(obj['u'].reshape(-1, obj['u'].shape[-1])).reshape(obj['u'].shape).squeeze()

    data_dict = {}
    data_dict["train_data"] = np.array(obs[:run_params['train_size'], :, :])
    data_dict["train_u"] = np.array(u[:run_params['train_size'], :, None])
    data_dict["val_data"] =  np.array(obs[-run_params['val_size']:, :, :])
    data_dict["val_u"] = np.array(u[-run_params['val_size']:, :, None])
    data_dict["scaled_goal"] = scaler_obs.transform(np.array([1., 0., 0.])[None]).squeeze() # obs are [cos(theta), sin(theta), theta_dot], where theta = 0 is upright (the goal)
    data_dict['scaler_obs'] = scaler_obs
    data_dict['scaler_u'] = scaler_u

    return data_dict

def load_pendulum(run_params, log=False):    
    d = run_params["dataset_params"]
    train_trials = d["train_trials"]
    val_trials = d["val_trials"]
    noise_scale = d["emission_cov"] ** 0.5
    key_train, key_val, key_pred = jr.split(d["seed"], 3)

    data = np.load("pendulum/pend_regression.npz")

    def _process_data(data, key):
        processed = data[:, ::2] / 255.0
        processed += jr.normal(key=key, shape=processed.shape) * noise_scale
        # return np.clip(processed, 0, 1)
        return processed # We are not cliping the data anymore!

    # Take subset, subsample every 2 frames, normalize to [0, 1]
    train_data = _process_data(data["train_obs"][:train_trials], key_train)
    train_states = data["train_targets"][:train_trials, ::2]
    # val_data = _process_data(data["test_obs"][:val_trials], key_val)
    data = np.load("pendulum/pend_regression_longer.npz")
    val_data = _process_data(data["test_obs"][:val_trials], key_pred)
    val_states = data["test_targets"][:val_trials, ::2]

    print("Full dataset:", data["train_obs"].shape)
    print("Subset:", train_data.shape)
    return {
        "train_data": train_data,
        "val_data": val_data,
        "train_states": train_states,
        "val_states": val_states,
    }

def load_nlb(run_params, log=False):
    d = run_params["dataset_params"]
    train_trials = d["train_trials"]
    val_trials = d["val_trials"]

    train_data = np.load("nlb-for-yz/nlb-dsmc_maze-phase_trn-split_trn.p", allow_pickle=True)
    val_data = np.load("nlb-for-yz/nlb-dsmc_maze-phase_trn-split_val.p", allow_pickle=True)

    x_train = np.asarray(train_data.tensors[0], dtype=np.float32)
    y_train = np.asarray(train_data.tensors[1], dtype=np.float32)
    x_val = np.asarray(val_data.tensors[0], dtype=np.float32)
    y_val = np.asarray(val_data.tensors[1], dtype=np.float32)

    print("Full dataset:", x_train.shape, x_val.shape)

    x_train, y_train = x_train[:train_trials], y_train[:train_trials]
    x_val, y_val = x_val[:val_trials], y_val[:val_trials]

    print("Subset:", x_train.shape, x_val.shape)

    return {
        "train_data": x_train,
        "train_targets": y_train,
        "val_data": x_val,
        "val_targets": y_val,
    }
