import copy
from copy import deepcopy

import jax.numpy as np
import jax.random as jr
key_0 = jr.PRNGKey(0)

# Tensorflow probability
import tensorflow_probability.substrates.jax.distributions as tfd
MVN = tfd.MultivariateNormalFullCovariance

from svae.utils import construct_dynamics_matrix, inv_softplus, lie_params_to_constrained, scale_matrix_by_norm
from svae.distributions import LinearGaussianChain
from svae.utils import dynamics_to_tridiag

from dynamax.utils.utils import psd_solve

class SVAEPrior:
    def init(self, key):
        """
        Returns the initial prior parameters.
        """
        pass

    def distribution(self, prior_params):
        """
        Returns a tfp distribution object
        Takes constrained params
        """
        pass

    def m_step(self, prior_params, posterior, post_params):
        """
        Returns updated prior parameters.
        """
        pass
    
    def sample(self, params, u, shape, key):
        samples = self.distribution(self.get_constrained_params(params, u)).sample(u=u, sample_shape=shape, seed=key)
        return samples

    def get_constrained_params(self, params, u):
        return deepcopy(params)

    @property
    def shape(self):
        raise NotImplementedError

class LinearGaussianChainPrior(SVAEPrior):

    def __init__(self, latent_dims, seq_len):
        self.latent_dims = latent_dims
        # The only annoying thing is that we have to specify the sequence length
        # ahead of time
        self.seq_len = seq_len

    @property
    def shape(self):
        return (self.seq_len, self.latent_dims)

    # Must be the full set of constrained parameters!
    def distribution(self, params):
        As, bs, Qs = params["As"], params["bs"], params["Qs"]
        Ex, Sigma, ExxT, ExnxT = params["Ex"], params["Sigma"], params["ExxT"], params["ExnxT"]
        return LinearGaussianChain(As, bs, Qs, Ex, Sigma, ExxT, ExnxT)

    def init(self, key):
        T, D = self.seq_len, self.latent_dims
        key_A, key = jr.split(key, 2)
        params = {
            "m1": np.zeros(D),
            "Q1": np.eye(D),
            "A": random_rotation(key_A, D, theta=np.pi/20),
            "b": np.zeros(D),
            "Q": np.eye(D)
        }
        constrained = self.get_constrained_params(params)
        return params

    def get_dynamics_params(self, params):
        return params

    def get_constrained_params(self, params, u):
        p = copy.deepcopy(params)
        tridiag = dynamics_to_tridiag(params, self.seq_len, self.latent_dims)
        p.update(tridiag)
        b = u[:-1] @ p["B"].T
        dist = LinearGaussianChain.from_stationary_dynamics(p["m1"], p["Q1"], 
                                         p["A"], b, p["Q"], self.seq_len)
        p.update({
            "As": dist._dynamics_matrix,
            "bs": dist._dynamics_bias,
            "Qs": dist._noise_covariance,
            "Ex": dist.expected_states,
            "Sigma": dist.covariance,
            "ExxT": dist.expected_states_squared,
            "ExnxT": dist.expected_states_next_states
        })

        # # natural parameters of prior marginals
        # prior_J = psd_solve(p["Sigma"], np.eye(self.latent_dims)[None])
        # prior_h = prior_J @ p["Ex"]

        # p.update({
        #     "prior_J": prior_J,
        #     "prior_h": prior_h
        # })

        return p

    def get_marginals_under_optimal_control(self, params, K):
        p = copy.deepcopy(params)

        # u_eq = np.linalg.solve(p["B"], (np.eye(self.latent_dims) - p["A"]) @ x_goal)

        # dynamics under optimal feedback control
        # x' = A @ x + B @ (u + u_eq)
        # x' = A @ x + B @ u + B @ u_eq
        # x' = A @ x - B @ K @ (x - x_goal) + B @ u_eq
        # x' = A @ x - B @ K @ x + B @ K @ x_goal + B @ u_eq
        # x' = (A - B @ K) @ x + B @ (K @ x_goal + u_eq)
        # x' = A_opt @ x + b_opt
        A_opt = p["A"] - p["B"] @ K
        # b_opt = np.tile(p["B"] @ (K @ x_goal + u_eq), (self.seq_len - 1, 1))
        b_opt = np.tile(np.zeros(self.latent_dims), (self.seq_len - 1, 1))

        dist = LinearGaussianChain.from_stationary_dynamics(p["m1"], p["Q1"], 
                                         A_opt, b_opt, p["Q"], self.seq_len)
        p.update({
            "As": dist._dynamics_matrix,
            "bs": dist._dynamics_bias,
            "Qs": dist._noise_covariance,
            "Ex": dist.expected_states,
            "Sigma": dist.covariance,
            "ExxT": dist.expected_states_squared,
            "ExnxT": dist.expected_states_next_states
        })

        # natural parameters of prior marginals
        prior_J = psd_solve(p["Sigma"], np.eye(self.latent_dims)[None])
        prior_h = prior_J @ p["Ex"]

        p.update({
            "prior_J": prior_J,
            "prior_h": prior_h
        })

        return p

class LieParameterizedLinearGaussianChainPrior(LinearGaussianChainPrior):

    def __init__(self, latent_dims, input_dims, seq_len, init_dynamics_noise_scale=1):
        super().__init__(latent_dims, seq_len)
        self.init_dynamics_noise_scale = init_dynamics_noise_scale
        self.input_dims = input_dims

    def init(self, key):
        D, U = self.latent_dims, self.input_dims
        key_A_u, key_A_v, key_A_s, key_B= jr.split(key, 4)
        # Equivalent to the unit matrix
        eps = min(self.init_dynamics_noise_scale / 100, 1e-4)
        Q_flat = np.concatenate([np.ones(D) 
            * inv_softplus(self.init_dynamics_noise_scale, eps=eps), np.zeros((D*(D-1)//2))])
        Q1_flat = np.concatenate([np.ones(D) * inv_softplus(1), np.zeros((D*(D-1)//2))])
        params = {
            "m1": np.zeros(D),
            # "Q1": np.zeros(D),
            "A_u": jr.normal(key_A_u, (D, D)),
            "A_v": jr.normal(key_A_v, (D, D)),
            "A_s": jr.normal(key_A_s, (D,)),
            "Q1": Q1_flat,
            "B": jr.normal(key_B, (D, U)),
            # "Q": np.zeros(D),
            "Q": Q_flat
        }
        return params

    def get_dynamics_params(self, params):
        return {
            "m1": params["m1"],
            "Q1": lie_params_to_constrained(params["Q1"], self.latent_dims),
            # "Q1": np.diag(np.exp(params["Q1"])),
            "A": construct_dynamics_matrix(params["A_u"], params["A_v"], params["A_s"], self.latent_dims),
            "B": scale_matrix_by_norm(params["B"]),
            # "B":params["B"],
            "Q": lie_params_to_constrained(params["Q"], self.latent_dims)   
            # "Q":  np.diag(np.exp(params["Q"]))   
        }

    def get_constrained_params(self, params, u):
        D = self.latent_dims
        p = self.get_dynamics_params(params)
        return super().get_constrained_params(p, u)