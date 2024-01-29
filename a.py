import copy
from copy import deepcopy

import jax.numpy as np
import jax.random as jr
from jax import vmap
from jax.lax import scan
key_0 = jr.PRNGKey(0)

# Tensorflow probability
import tensorflow_probability.substrates.jax.distributions as tfd
MVN = tfd.MultivariateNormalFullCovariance

from svae.utils import construct_dynamics_matrix, inv_softplus, lie_params_to_constrained, scale_matrix_by_norm, construct_covariance_matrix, random_rotation, truncate_singular_values
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
        # if u is None:
        #     # tridiag = dynamics_to_tridiag(params, self.seq_len, self.latent_dims)
        #     # p.update(tridiag)
        #     # A = p["A"] + p["B"] @ p["U"]
        #     A = p["Abar"]
        #     b = np.tile(p["B"] @ p["v"], (self.seq_len - 1, 1))
        #     Q = p["Q"] + p["B"] @ p["S"] @ p["B"].T
        #     # b = u[:-1] @ p["B"].T
        #     dist = LinearGaussianChain.from_stationary_dynamics(p["m1"], p["Q1"], 
        #                                      A, b, Q, self.seq_len)
        #     p.update({
        #         "Ex": dist.expected_states,
        #         "Sigma": dist.covariance,
        #     })
        # else:
        #     tridiag = dynamics_to_tridiag(params, self.seq_len, self.latent_dims)
        #     p.update(tridiag)
        #     dist = LinearGaussianChain.from_stationary_dynamics(p["m1"], p["Q1"], 
        #                                      p["A"], u[:-1] @ p["B"].T, p["Q"], self.seq_len)
        #     p.update({
        #         "As": dist._dynamics_matrix,
        #         "bs": dist._dynamics_bias,
        #         "Qs": dist._noise_covariance,
        #         "Ex": dist.expected_states,
        #         "Sigma": dist.covariance,
        #         "ExxT": dist.expected_states_squared,
        #         "ExnxT": dist.expected_states_next_states
        #     })
        tridiag = dynamics_to_tridiag(params, self.seq_len, self.latent_dims)
        p.update(tridiag)
        breakpoint()
        dist = LinearGaussianChain.from_stationary_dynamics(p["m1"], p["Q1"], 
                                         p["A"], u[:-1] @ p["B"].T, p["Q"], self.seq_len)
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

    # def cond_fun(self, x, eps = 1e-6):

    #     P, delta_P_norm, A, B, Q, R = x

    #     return delta_P_norm > eps

    # def get_previous_P(self, x):

    #     P, delta_P_norm, A, B, Q, R = x

    #     prev_P = Q + A.T @ P @ A - (A.T @ P @ B) @ np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)

    #     return prev_P, np.linalg.norm(P - prev_P), A, B, Q, R

    # def get_optimal_feedback_gain(self, A, B, Q, R):

    #     init_val = Q, 1e3, A, B, Q, R
    #     P, _, _, _, _, _ = while_loop(self.cond_fun, self.get_previous_P, init_val) # iterate until P converges
    #     K = np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A

    #     return K

    def get_previous_P(self, carry, inputs):

        P, A, B, Q, R = carry

        prev_P = Q + A.T @ P @ A - (A.T @ P @ B) @ np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)

        carry = prev_P, A, B, Q, R
        outputs = None

        return carry, outputs

    def get_optimal_feedback_gain(self, A, B, Q, R):

        carry = Q, A, B, Q, R
        (P, _, _, _, _), _ = scan(self.get_previous_P, carry, None, length=100)
        K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)

        return K

    def get_marginals_under_optimal_control(self, params, x_goal, u_eq, K):
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
        b_opt = np.tile(p["B"] @ (K @ x_goal + u_eq), (self.seq_len - 1, 1))
        # b_opt = np.tile(np.zeros(self.latent_dims), (self.seq_len - 1, 1))

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
        prior_J = vmap(lambda S, I: psd_solve(S, I), in_axes=(0, None))(p["Sigma"], np.eye(self.latent_dims))
        prior_h = np.einsum("ijk,ik->ij", prior_J, p["Ex"])

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
        self.seq_len = seq_len

    def init(self, key):
        D, U = self.latent_dims, self.input_dims
        key_A_u, key_A_v, key_A_s, key_U_u, key_U_v, key_U_s, key_B, key_U, key_v, key_C, key_d, key_R, key_J_aux, key_h_aux = jr.split(key, 14)
        # Equivalent to the unit matrix
        # eps = min(self.init_dynamics_noise_scale / 100, 1e-4)
        # Q_flat = np.concatenate([np.ones(D) 
        #     * inv_softplus(self.init_dynamics_noise_scale, eps=eps), np.zeros((D*(D-1)//2))])
        # Q1_flat = np.concatenate([np.ones(D) * inv_softplus(1), np.zeros((D*(D-1)//2))])
        I = np.eye(D)
        Q_flat = I[np.tril_indices(D)]
        Q1_flat = I[np.tril_indices(D)]
        S_flat = I[np.tril_indices(U)]
        R_flat = I[np.tril_indices(D)]
        # A = random_rotation(key_A_u, D, theta=np.pi/20)
        # U, S, V = np.linalg.svd(A) # A = U @ np.diag(S) @ V
        # construct_dynamics_matrix((U @ np.triu(np.ones((D, D)))).ravel(), (V.T @ np.triu(np.ones((D, D)))).ravel(), np.log(np.divide(S, 1.-S)), D)
        params = {
            "m1": np.zeros(D),
            # "Q1": np.eye(D),
            # "A": truncate_singular_values(jr.normal(key_A_u, (D, D))/np.sqrt(D)),
            "A_u": jr.normal(key_A_u, (D, D)),
            "A_v": jr.normal(key_A_v, (D, D)),
            "A_s": jr.normal(key_A_s, (D,)),
            "Abar_u": jr.normal(key_U_u, (D, D)),
            "Abar_v": jr.normal(key_U_v, (D, D)),
            "Abar_s": jr.normal(key_U_s, (D,)),
            # "U_u": jr.normal(key_U_u, (U, U)),
            # "U_v": jr.normal(key_U_v, (D, D)),
            # "U_s": jr.normal(key_U_s, (min(U,D),)),
            "Q1": Q1_flat,
            # "Q1": np.zeros(D),
            # "Q1": np.eye(D),
            "B": jr.normal(key_B, (D, U))/np.sqrt(U),
            # "U": jr.normal(key_U, (U, D))/np.sqrt(D),
            "v": jr.normal(key_v, (U,)),
            # "Q": np.eye(D),
            "Q": Q_flat,
            "S": S_flat,
            "R": R_flat,
            # "Q": np.zeros(D),
            # "Q": np.eye(D),
            "goal_norm": np.ones(1),
            "J_aux": jr.normal(key_J_aux, (100, self.seq_len,  D * (D + 1) // 2)),
            "h_aux": jr.normal(key_h_aux, (100, self.seq_len, D)),
            "C": jr.normal(key_C, shape=(D, D)),
            "d": jr.normal(key_d, shape=(D,)),
        }
        return params

    def get_dynamics_params(self, params):
        D = self.latent_dims
        return {
            "m1": params["m1"],
            # "Q1": lie_params_to_constrained(params["Q1"], self.latent_dims),
            "Q1": construct_covariance_matrix(params["Q1"], self.latent_dims),
            # "Q1": np.diag(np.exp(params["Q1"])) + 1e-4 * np.eye(D),
            "A": construct_dynamics_matrix(params["A_u"], params["A_v"], params["A_s"], self.latent_dims, self.latent_dims),
            # "A": params["A"],
            "B": scale_matrix_by_norm(params["B"]),
            # "U": scale_matrix_by_norm(params["U"]),
            "Abar": construct_dynamics_matrix(params["Abar_u"], params["Abar_v"], params["Abar_s"], self.latent_dims, self.latent_dims),
            # "U": construct_dynamics_matrix(params["U_u"], params["U_v"], params["U_s"], params["v"].size, self.latent_dims),
            "v": scale_matrix_by_norm(params["v"]),
            # "Q1": params["Q1"],
            # "Q": params["Q"],
            # "B":params["B"],
            # "Q": lie_params_to_constrained(params["Q"], self.latent_dims),
            "Q": construct_covariance_matrix(params["Q"], self.latent_dims),
            "S": construct_covariance_matrix(params["S"], params["v"].size),
            "R": construct_covariance_matrix(params["R"], self.latent_dims),   
            # "Q":  np.diag(np.exp(params["Q"])) + 1e-4 * np.eye(D),
            "goal_norm": params["goal_norm"],
            "J_aux": vmap(vmap(lambda J, d: construct_covariance_matrix(J, d), in_axes=(0,None)), in_axes=(0,None))(params["J_aux"], self.latent_dims),
            "h_aux": params["h_aux"],
            "C": params["C"],
            "d": params["d"],
        }

    def get_constrained_params(self, params, u):
        D = self.latent_dims
        p = self.get_dynamics_params(params)
        p.update({
            "U": np.linalg.pinv(p['B']) @ (p['Abar'] - p['A']),
        })
        return super().get_constrained_params(p, u)