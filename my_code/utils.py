from jax import numpy as np
from jax import random, jit, vmap, value_and_grad
from jax.lax import scan, dynamic_slice, dynamic_update_slice, stop_gradient
import flax.linen as nn
from flax.linen import sigmoid
from sklearn import linear_model
import optax

import tensorflow_probability.substrates.jax.distributions as tfd
MVN = tfd.MultivariateNormalFullCovariance

from jax.scipy.special import logsumexp

from flax.training.orbax_utils import restore_args_from_target
from flax.training import train_state
from orbax.checkpoint import AsyncCheckpointer, Checkpointer, PyTreeCheckpointHandler, CheckpointManager, CheckpointManagerOptions

from dynamax.utils.utils import psd_solve
from jax.scipy.linalg import block_diag

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

from dynamax.linear_gaussian_ssm.inference import make_lgssm_params, lgssm_smoother

from copy import deepcopy

import wandb
import matplotlib.pyplot as plt
import seaborn as sns

def get_scaler(scaler):
    
    scalers = {
        "minmax": MinMaxScaler,
        "standard": StandardScaler,
        "maxabs": MaxAbsScaler,
        "robust": RobustScaler,
    }

    return scalers.get(scaler.lower())()

def scale_y(y):

    scaler_y = get_scaler('standard')

    y_shape = y.shape
    
    return scaler_y.fit_transform(y.reshape(-1, y.shape[-1])).reshape(y_shape)

def half_log_det(Sigma, diagonal_boost = 1e-9):

    L = np.linalg.cholesky(Sigma + diagonal_boost * np.eye(Sigma.shape[-1]))
    half_log_det_Sigma = np.log(np.diagonal(L, axis1 = -2, axis2 = -1)).sum(-1)

    return half_log_det_Sigma

batch_half_log_det = vmap(half_log_det)

def construct_dynamics_matrix(u, v, s, dim, eps = 1e-3):

    U, _ = np.linalg.qr(u.reshape((dim, dim)))
    V, _ = np.linalg.qr(v.reshape((dim, dim)))
    singular_values = sigmoid(s) * (1 - eps) + eps / 2
    Sigma = np.diag(singular_values)
    A = U @ Sigma @ V.T

    return A

def truncate_singular_values(A, eps = 1e-3):

    # u, s, vt = svd(A) NotImplementedError: Singular value decomposition JVP not implemented for full matrices
    u, s, vt = np.linalg.svd(A) 

    return u @ np.diag(np.clip(s, eps, 1)) @ vt

def construct_covariance_matrix(x, dim, eps=1e-6):

    # create lower triangular matrix
    L = np.zeros((dim, dim))
    L = L.at[np.tril_indices(dim)].set(x)

    # construct covariance matrix via cholesky decomposition
    Sigma = L @ L.T

    # add scaled identity matrix for stability
    Sigma += eps * np.eye(dim)

    return Sigma

def construct_precision_matrix(x, dim, eps=1e-6):

    # create lower triangular matrix
    L = np.zeros((dim, dim))
    L = L.at[np.tril_indices(dim)].set(x)

    # construct covariance matrix via cholesky decomposition
    Lambda = L @ L.T

    # add scaled identity matrix for stability
    Lambda += eps * np.eye(dim)

    return Lambda

def dynamic_slice_add(x, start_indices, slice_sizes, y):

    x_new = dynamic_slice(x, start_indices, slice_sizes) + y
    x_updated = dynamic_update_slice(x, x_new, start_indices)

    return x_updated

def log_normaliser(J, h, diagonal_boost = 1e-9):

    # https://en.wikipedia.org/wiki/Exponential_family

    # https://math.stackexchange.com/questions/3158303/using-cholesky-decomposition-to-compute-covariance-matrix-determinant
    L = np.linalg.cholesky(J + diagonal_boost * np.eye(J.shape[-1]))
    half_log_det_precision = np.log(np.diagonal(L)).sum()

    return 0.5 * h @ psd_solve(J, h) - half_log_det_precision

def random_rotation(seed, n, theta=None):
    key1, key2 = random.split(seed)

    if theta is None:
        # Sample a random, slow rotation
        theta = 0.5 * np.pi * random.uniform(key1)

    if n == 1:
        return random.uniform(key1) * - np.eye(1)

    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    out = np.diag(random.uniform(key1, (n,)))
    out = out.at[:2, :2].set(rot)
    q = np.linalg.qr(random.uniform(key2, shape=(n, n)))[0]
    return q.dot(out).dot(q.T)

def kl_qp_natural_parameters(J_q, h_q, J_p, h_p):

    J_diff = J_q - J_p
    h_diff = h_q - h_p

    Sigma_q = psd_solve(J_q, np.eye(h_q.size))
    mu_q = Sigma_q @ h_q

    trm = np.einsum("i,i->", h_diff, mu_q) - 0.5 * (np.einsum("i,i->", mu_q, np.einsum("ij,j->i", J_diff, mu_q)) + (J_diff * Sigma_q).sum(axis = (0, 1)))

    log_normaliser_q = log_normaliser(J_q, h_q)
    log_normaliser_p = log_normaliser(J_p, h_p)

    return trm + log_normaliser_p - log_normaliser_q

def expected_log_F(mu_posterior, Sigma_posterior, rpm_mu, rpm_Sigma, key):

    MVN.reparameterization_type=tfd.FULLY_REPARAMETERIZED

    samples = MVN(loc=mu_posterior, covariance_matrix=Sigma_posterior).sample(seed=key)
    log_f_prob = vmap(vmap(lambda m, S, x: MVN(loc=m, covariance_matrix=S).log_prob(x), in_axes=(0, 0, 0)), in_axes=(0, 0, None))(rpm_mu, rpm_Sigma, samples)
    batch_size = rpm_mu.shape[0]
    # mx = log_f_prob.max(axis=0)
    # F_log_p = (mx + logsumexp(log_f_prob - mx, axis=0, b=1/batch_size)).sum()
    log_F = logsumexp(log_f_prob, axis=0, b=1/batch_size).sum()

    return log_F

batch_expected_log_F = vmap(expected_log_F, in_axes=(None,None,None,None,0))

def entropy(J):

    dim = J.shape[0]
    L = np.linalg.cholesky(J + 1e-9 * np.eye(dim))
    half_log_det_J = np.log(np.diagonal(L, axis1 = -2, axis2 = -1)).sum(-1)
    entropy = (1 + np.log(2*np.pi) - half_log_det_J * 2 / dim) * dim / 2

    return entropy

def moment_match_RPM(RPM):

    MM = {}
    MM['mu'] = RPM['mu'].mean(axis=(0,))
    mu_diff = RPM['mu'] - MM['mu']
    MM['Sigma'] = RPM["Sigma"].mean(axis=(0,)) + np.einsum("ijk,ijl->ijkl", mu_diff, mu_diff).mean(axis = 0)
    MM['J'] = vmap(lambda S: psd_solve(S, np.eye(S.shape[0])))(MM['Sigma'])
    MM['h'] = np.einsum("ijk,ik->ij", MM['J'], MM['mu'])

    return MM

def marginal_u_integrated_out(prior_params, T):

    carry = prior_params['m1'], prior_params['Q1'], prior_params
    inputs = np.arange(T)
    _, (mu, P) = scan(one_step_transition_action_integrated_out, carry, None, length = T)
    prior_marginal = {}
    prior_marginal['mu'] = mu
    prior_marginal['Sigma'] = P
    prior_marginal['J'] = vmap(lambda S: psd_solve(S, np.eye(S.shape[0])))(prior_marginal['Sigma'])
    prior_marginal['h'] = np.einsum("ijk,ik->ij", prior_marginal['J'], prior_marginal['mu'])

    return prior_marginal

def marginal(prior_params, T):

    D = prior_params['m1_F'].shape[-1]
    Q1_F = construct_covariance_matrix(prior_params['Q1_F_flat'], D)
    Q_F = construct_covariance_matrix(prior_params['Q_F_flat'], D)

    carry = prior_params['m1_F'], Q1_F, prior_params, Q_F
    inputs = np.arange(T)
    _, (mu, P) = scan(one_step_transition_no_action, carry, None, length = T)
    prior_marginal = {}
    prior_marginal['mu'] = mu
    prior_marginal['Sigma'] = P
    prior_marginal['J'] = vmap(lambda S: psd_solve(S, np.eye(S.shape[0])))(prior_marginal['Sigma'])
    prior_marginal['h'] = np.einsum("ijk,ik->ij", prior_marginal['J'], prior_marginal['mu'])

    return prior_marginal

# def load_pendulum_control_data(run_params):

#     import pickle
#     obj = pickle.load(open("pendulum_data.pkl", 'rb'))

#     scaler_obs = get_scaler('standard')
#     scaler_u = get_scaler('standard')
#     obj['u'] = obj['u'][:, :, None]
#     assert obj['observations'].ndim == obj['u'].ndim == 3
#     obs = scaler_obs.fit_transform(obj['observations'].reshape(-1, obj['observations'].shape[-1])).reshape(obj['observations'].shape)
#     u = scaler_u.fit_transform(obj['u'].reshape(-1, obj['u'].shape[-1])).reshape(obj['u'].shape).squeeze()

#     data_dict = {}
#     data_dict["train_data"] = np.array(obs[:run_params['train_size'], :, :])
#     data_dict["train_u"] = np.array(u[:run_params['train_size'], :, None])
#     data_dict["val_data"] =  np.array(obs[-run_params['val_size']:, :, :])
#     data_dict["val_u"] = np.array(u[-run_params['val_size']:, :, None])
#     data_dict["scaled_goal"] = scaler_obs.transform(np.array([1., 0., 0.])[None]).squeeze() # obs are [cos(theta), sin(theta), theta_dot], where theta = 0 is upright (the goal)
#     data_dict['scaler_obs'] = scaler_obs
#     data_dict['scaler_u'] = scaler_u

#     return data_dict

def load_pendulum_control_data():

    import pickle
    obj = pickle.load(open("/nfs/nhome/live/jheald/svae/pendulum_data.pkl", 'rb'))

    return obj['observations'], obj['u']

def generate_LDS_params(D, U, key):

    key_A_u, key_A_v, key_A_s, key_Abar_u, key_Abar_v, key_Abar_s, key_m1, key_B, key_C, key_d, key_l = random.split(key, 11)

    I = np.sqrt(0.1) * np.eye(D)
    Q_flat = I[np.tril_indices(D)]
    Q1_flat = I[np.tril_indices(D)]
    S_flat = I[np.tril_indices(U)]
    R_flat = I[np.tril_indices(D)]

    A_u = random.normal(key_A_u, (D, D))
    A_v = random.normal(key_A_v, (D, D))
    A_s = random.normal(key_A_s, (D,))

    Abar_u = random.normal(key_Abar_u, (D, D))
    Abar_v = random.normal(key_Abar_v, (D, D))
    Abar_s = random.normal(key_Abar_s, (D,))

    params = {}
    params['m1'] = random.normal(key_m1, (D, ))
    params['Q1'] = construct_covariance_matrix(Q1_flat, D)
    # params['A'] = truncate_singular_values(A_u / np.sqrt(D))
    # params['A'] = construct_dynamics_matrix(A_u, A_v, A_s, D)
    params['A'] = random_rotation(key_A_u, D, theta=np.pi/20) # theta=np.pi/20
    params['B'] = random.normal(key_m1, (D, U)) / np.sqrt(U)
    params['Q'] = construct_covariance_matrix(Q1_flat, D)
    # params['Abar'] = construct_dynamics_matrix(Abar_u, Abar_v, Abar_s, D)
    # params['Abar'] = truncate_singular_values(Abar_u / np.sqrt(D))
    params['Abar'] = random_rotation(key_Abar_u, D, theta=np.pi/20) # theta=np.pi/20
    params['K'] = np.linalg.pinv(params['B']) @ (params['Abar'] - params['A'])
    params['l'] = random.normal(key_l, (U,))
    params['S'] = construct_covariance_matrix(S_flat, U)
    params['C'] = random.normal(key_C, (D, D)) / np.sqrt(D)
    params['d'] = random.normal(key_d, (D,)) 
    params['R'] = construct_covariance_matrix(R_flat, D)

    return params

def initialise_LDS_params(D, U, key, closed_form_M_Step):

    key_A_u, key_A_v, key_A_s, key_Abar_u, key_Abar_v, key_Abar_s, key_m1, key_B, key_l, key_A_F, key_b_F, key_m1_F  = random.split(key, 12)

    I = np.sqrt(0.1) * np.eye(D)

    params = {}
    params['m1'] = np.zeros(D)
    # params['A'] = truncate_singular_values(random.normal(key_A_u, (D, D)) / np.sqrt(D))
    params['A'] = np.eye(D)
    # params['A_u'] = random.normal(key_A_u, (D, D))
    # params['A_v'] = random.normal(key_A_v, (D, D))
    # params['A_s'] = random.normal(key_A_s, (D,))
    # if options['learn b']:
    #     params['B'] = np.zeros((D, U+1))
    # else:
    #     params['B'] = np.zeros((D, U)) # random.normal(key_m1, (D, U)) / np.sqrt(U)
    params['B'] = np.zeros((D, U)) # random.normal(key_m1, (D, U)) / np.sqrt(U)

    if closed_form_M_Step:
        params['Q1'] = I
        params['Q'] = I
    else:
        params['Q1_flat'] = I[np.tril_indices(D)]
        params['Q_flat'] = I[np.tril_indices(D)]
        # params['Q1_flat'] = np.diag(I)
        # params['Q_flat'] = np.diag(I)

    # params['m1_F'] = np.zeros(D)
    # params['Q1_F_flat'] = I[np.tril_indices(D)]
    # # params['A_F'] = truncate_singular_values(random.normal(key_A_F, (D, D)) / np.sqrt(D))
    # params['A_F'] = np.eye(D)
    # params['b_F'] = np.zeros(D)
    # params['Q_F_flat'] = I[np.tril_indices(D)]
    
    # # # params['Abar_u'] = random.normal(key_Abar_u, (D, D))
    # # # params['Abar_v'] = random.normal(key_Abar_v, (D, D))
    # # # params['Abar_s'] = random.normal(key_Abar_s, (D,))
    # # params['Abar'] = truncate_singular_values(random.normal(key_Abar_u, (D, D)) / np.sqrt(D))
    # params['l'] = np.zeros(U)
    # params['S_flat'] = I[np.tril_indices(U)]

    return params

def initialise_LDS_params_via_M_step(RPM_model, rpm_params, y, u, key, options, closed_form_M_Step):

    if options['f_time_dependent']:

        D = y.shape[2] - 1

    else:

        D = y.shape[2]

        RPM = RPM_model.apply(rpm_params, y)

    prior_params = {}
    AB_1 = vmap(vmap(lambda Ex_t, Extm1, u_t: np.hstack((np.outer(Ex_t, Extm1), np.outer(Ex_t, u_t)))))(RPM['mu'][:, 1:, :], RPM['mu'][:, :-1, :], u[:, 1:, :]).sum(axis=(0,1))
    AB_2 = vmap(vmap(lambda Cov_tm1, Ex_tm1, u_t: np.block([[Cov_tm1 + np.outer(Ex_tm1, Ex_tm1), np.outer(Ex_tm1, u_t)],[np.outer(u_t, Ex_tm1), np.outer(u_t, u_t)]])))(RPM['Sigma'][:, :-1, :, :], RPM['mu'][:, :-1, :], u[:, 1:, :]).sum(axis=(0,1))
    AB = np.linalg.solve(AB_2.T, AB_1.T).T # this equals AB_1 @ np.linalg.inv(AB_2)
    prior_params["A"] = AB[:, :D]
    prior_params["B"] = AB[:, D:]
    Q = vmap(vmap(lambda Cov_t, Ex_t, Extm1, AB, u_t: Cov_t + np.outer(Ex_t, Ex_t) - AB @ np.vstack((np.outer(Extm1, Ex_t), np.outer(u_t, Ex_t))), in_axes=(0,0,0,None,0)), in_axes=(0,0,0,None,0))(RPM['Sigma'][:, 1:, :, :], RPM['mu'][:, 1:, :], RPM['mu'][:, :-1, :], AB, u[:, 1:, :]).mean(axis=(0,1))
    
    prior_params["m1"] = RPM['mu'][:, 0, :].mean(axis=(0,))
    mu_diff = RPM['mu'][:, 0, :] - prior_params["m1"]
    Q1 = RPM['Sigma'][:, 0, :, :].mean(axis=(0,)) + np.einsum("jk,jl->jkl", mu_diff, mu_diff).mean(axis = 0)

    prior_params["A"] = truncate_singular_values(prior_params["A"])

    Q = (Q + Q.T)/2
    Q += np.eye(D) * 1e-4
    Q1 = (Q1 + Q1.T)/2
    Q1 += np.eye(D) * 1e-4

    if closed_form_M_Step:

        prior_params["Q"] = Q
        prior_params["Q1"] = Q1

    else:

        prior_params["Q_flat"] = np.linalg.cholesky(Q)[np.tril_indices(D)]
        prior_params["Q1_flat"] = np.linalg.cholesky(Q1)[np.tril_indices(D)]

    key_A_u, key_A_v, key_A_s, key_Abar_u, key_Abar_v, key_Abar_s, key_m1, key_B, key_l, key_A_F, key_b_F, key_m1_F  = random.split(key, 12)
    I = np.sqrt(0.1) * np.eye(D)
    prior_params['m1_F'] = np.zeros(D)
    prior_params['Q1_F_flat'] = I[np.tril_indices(D)]
    prior_params['A_F'] = truncate_singular_values(random.normal(key_A_F, (D, D)) / np.sqrt(D))
    # prior_params['A_F'] = np.eye(D) * 0.99
    prior_params['b_F'] = np.zeros(D)
    prior_params['Q_F_flat'] = I[np.tril_indices(D)]

    return prior_params

def get_constrained_prior_params(params, U, eps=0.):

    params = deepcopy(params)

    D = params['m1'].size
    params['Q1'] = construct_covariance_matrix(params['Q1_flat'], D)
    # params['Q1'] = np.diag(np.exp(params['Q1_flat'])+eps)
    # params['Q1_F'] = construct_covariance_matrix(params['Q1_F_flat'], D)
    params['Q'] = construct_covariance_matrix(params['Q_flat'], D)
    # params['Q'] = np.diag(np.exp(params['Q_flat'])+eps)
    # params['Q_F'] = construct_covariance_matrix(params['Q_F_flat'], D)

    # params['Q'] = np.sqrt(0.1) * np.eye(D)
    # params['Q1'] = np.sqrt(0.1) * np.eye(D)

    # params['A'] = construct_dynamics_matrix(params['A_u'], params['A_v'], params['A_s'], D)
    # params['Abar'] = construct_dynamics_matrix(params['Abar_u'], params['Abar_v'], params['Abar_s'], D)
    # params['K'] = np.linalg.pinv(params['B']) @ (params['Abar'] - params['A'])
    # params['S'] = construct_covariance_matrix(params['S_flat'], U)

    return params

def log_prob_under_prior(prior_params, x, u):

    def log_prop_one_transition(A, x_prev, B, u_prev, Q, x):

        return MVN(loc=A @ x_prev + B @ u_prev, covariance_matrix=Q).log_prob(x)

    log_prop_all_transitions = vmap(log_prop_one_transition, in_axes=(None,0,None,0,None,0))

    m1 = prior_params['m1']
    Q1 = prior_params['Q1']
    A = prior_params['A']
    B = prior_params['B']
    Q = prior_params['Q']

    ll = MVN(loc=m1, covariance_matrix=Q1).log_prob(x[0])
    ll += log_prop_all_transitions(A, x[:-1], B, u[:-1], Q, x[1:]).sum()

    return ll

def log_normalizer(prior_params, smoothed, potentials, u):
    
    def predictive_distribution_one_transition(mu_filtered, Sigma_filtered, A, B, u, Q):

        mu = A @ mu_filtered + B @ u
        Sigma = A @ Sigma_filtered @ A.T + Q

        return mu, Sigma

    predictive_distribution_all_transitions = vmap(predictive_distribution_one_transition, in_axes=(0,0,None,None,0,None))

    def conditional_marginal_likelihood_one_observation(mu_pred, Sigma_pred, mu_rec, Sigma_rec):

        return MVN(loc=mu_pred, covariance_matrix=Sigma_pred + Sigma_rec).log_prob(mu_rec)

    conditional_marginal_likelihood_all_observations = vmap(conditional_marginal_likelihood_one_observation)

    m1 = prior_params['m1']
    Q1 = prior_params['Q1']
    A = prior_params['A']
    B = prior_params['B']
    Q = prior_params['Q']

    mu, Sigma = predictive_distribution_all_transitions(smoothed["filtered_means"][:-1], smoothed["filtered_covariances"][:-1], A, B, u[:-1], Q)

    mu_pred = np.concatenate([m1[None], mu])
    Sigma_pred = np.concatenate([Q1[None], Sigma])
    mu_rec, Sigma_rec = potentials["mu"], potentials["Sigma"]

    ll = conditional_marginal_likelihood_all_observations(mu_pred, Sigma_pred, mu_rec, Sigma_rec).sum()

    return ll

def log_prob_under_posterior(prior_params, emission_potentials, smoothed, u):

    def log_prop_one_emission(mu, Sigma, x):

        return MVN(loc=mu, covariance_matrix=Sigma).log_prob(x)

    log_prop_all_emissions = vmap(log_prop_one_emission)

    # ll = log_prob_under_prior(prior_params, smoothed['smoothed_means'], u)
    # ll += log_prop_all_emissions(emission_potentials["mu"], emission_potentials["Sigma"], smoothed['smoothed_means']).sum()
    ll = log_prop_all_emissions(emission_potentials["mu"], emission_potentials["Sigma"], smoothed['smoothed_means']).sum()
    ll -= log_normalizer(prior_params, smoothed, emission_potentials, u)

    return ll

def get_marginals_of_joint(mu, Sigma, T, D):

    mu_marginal = vmap(dynamic_slice, in_axes=(None,0,None))(mu, (np.arange(T) * D,), (D,))
    Sigma_marginal = vmap(dynamic_slice, in_axes=(None,0,None))(Sigma, (np.arange(T) * D, np.arange(T) * D), (D, D))

    return mu_marginal, Sigma_marginal

def policy_loss(params, u, mu, Sigma, diagonal_boost = 1e-9):

    K = params['K']
    l = params['l']
    S = params['S']

    D = params['A'].shape[0]
    U = u.size
    
    J = psd_solve(S, np.eye(U))
    L = np.linalg.cholesky(J + diagonal_boost * np.eye(U))
    half_log_det_J = np.log(np.diagonal(L)).sum()

    loss = 0.5 * (-U * np.log(2 * np.pi) + 2 * half_log_det_J - (u - l) @ J @ (u - l) + 2 * (u - l) @ J @ K @ mu - (K @ mu).T @ J @ (K @ mu) - ((K.T @ J @ K) * Sigma).sum())

    return loss

def perform_Kalman_smoothing(p, emissions_potentials, u):

    mus, Sigmas = emissions_potentials["mu"], emissions_potentials["Sigma"]

    dim = mus.shape[-1]
    C = np.eye(dim)
    d = np.zeros(dim)

    params = make_lgssm_params(p["m1"], p["Q1"], p["A"], p["Q"], C, Sigmas,
                               dynamics_input_weights=p["B"], emissions_bias=d)

    smoothed = lgssm_smoother(params, mus, u)._asdict()

    return smoothed

batch_perform_Kalman_smoothing = vmap(perform_Kalman_smoothing, in_axes=(None,0,0))

def perform_Kalman_smoothing_true_params(p, y, u):

    params = make_lgssm_params(p["m1"], p["Q1"], p["A"], p["Q"], p["C"], p["R"],
                               dynamics_input_weights=p["B"], emissions_bias=p["d"])

    smoothed = lgssm_smoother(params, y, u)._asdict()

    return smoothed

batch_perform_Kalman_smoothing_true_params = vmap(perform_Kalman_smoothing_true_params, in_axes=(None,0,0))

def R2_inferred_vs_actual_z(posterior_means, true_z):

    true_z_shape = true_z.shape

    posterior_means = posterior_means.reshape(-1, true_z_shape[-1])
    true_z = true_z.reshape(-1, true_z_shape[-1])

    reg = linear_model.LinearRegression()
    reg.fit(posterior_means, true_z) # reg.predict(posterior_means) = posterior_means @ reg.coef_.T + reg.intercept_
    R2 = reg.score(posterior_means, true_z)
    predicted_z = reg.predict(posterior_means)

    predicted_z = predicted_z.reshape(true_z_shape)

    return R2, predicted_z

def generate_data(params, T, D, U, key):

    subkey1, subkey2, subkey3, key = random.split(key, 4)

    z = np.zeros((T,D))
    y = np.zeros((T,D))
    u = np.zeros((T,U))

    z = z.at[0,:].set(MVN(loc=params['m1'], covariance_matrix=params['Q1']).sample(seed=subkey1))
    y = y.at[0,:].set(MVN(loc=params['C'] @ z[0,:] + params['d'], covariance_matrix=params['R']).sample(seed=subkey2))
    u = u.at[0,:].set(MVN(loc=params['K'] @ z[0,:] + params['l'], covariance_matrix=params['S']).sample(seed=subkey3))

    for t in range(1,T):

        subkey1, subkey2, subkey3, key = random.split(key, 4)

        z = z.at[t,:].set(MVN(loc=params['A'] @ z[t-1,:] + params['B'] @ u[t-1,:], covariance_matrix=params['Q']).sample(seed=subkey1))
        y = y.at[t,:].set(MVN(loc=params['C'] @ z[t,:] + params['d'], covariance_matrix=params['R']).sample(seed=subkey2))
        u = u.at[t,:].set(MVN(loc=params['K'] @ z[t,:] + params['l'], covariance_matrix=params['S']).sample(seed=subkey3))

    return z, y, u

batch_generate_data = vmap(generate_data, in_axes=(None,None,None,None,0))

def get_evidence_potential(params, y):

    C, d, R = params["C"], params["d"], params["R"]

    J = np.dot(C.T, np.linalg.solve(R, C))
    J = np.tile(J[None, :, :], (y.shape[0], 1, 1))
    h = np.dot(y - d, np.linalg.solve(R, C))

    return J, h

batch_get_evidence_potential = vmap(get_evidence_potential, in_axes=(None,0))

def R2_true_model(params, y, u, true_z):

    J_evidence, h_evidence = batch_get_evidence_potential(params, y)

    prior, likelihood, posterior = vmap(update_prior, in_axes=(None,0,0,0))(params, u, J_evidence, h_evidence)
    
    Sigma_posterior = vmap(lambda J: psd_solve(J, np.eye(J.shape[-1])))(posterior['J'])
    mu_posterior = np.einsum("hij,hj->hi", Sigma_posterior, posterior['h'])

    R2, predicted_z = R2_inferred_vs_actual_z(mu_posterior, true_z)

    return R2, predicted_z

def closed_form_LDS_updates(params, smoothed, u, mean_field_q):

    smoothed["expected_states_squared"] = smoothed["smoothed_covariances"] + np.einsum("...i,...j->...ij", smoothed["smoothed_means"], smoothed["smoothed_means"])

    if mean_field_q:

        smoothed["smoothed_cross_covariances"] = np.einsum("...i,...j->...ij", smoothed['smoothed_means'][:,:-1,:], smoothed['smoothed_means'][:,1:,:])

    AB_1 = vmap(vmap(lambda Extnt, Ex_t, u_t: np.hstack((Extnt.T, np.outer(Ex_t, u_t)))))(smoothed["smoothed_cross_covariances"], smoothed['smoothed_means'][:, 1:, :], u[:, 1:, :]).sum(axis=(0,1))
    AB_2 = vmap(vmap(lambda Extt, Ex_tm1, u_t: np.block([[Extt, np.outer(Ex_tm1, u_t)],[np.outer(u_t, Ex_tm1), np.outer(u_t, u_t)]])))(smoothed["expected_states_squared"][:,:-1,:,:], smoothed['smoothed_means'][:, :-1, :], u[:, 1:, :]).sum(axis=(0,1))
    AB = np.linalg.solve(AB_2.T, AB_1.T).T # this equals AB_1 @ np.linalg.inv(AB_2)
    
    D = smoothed['smoothed_means'].shape[-1]
    params["prior_params"]["A"] = AB[:, :D]
    params["prior_params"]["B"] = AB[:, D:]
    params["prior_params"]["Q"] = vmap(vmap(lambda Extt, Extnt, Ex_t, AB, u_t: Extt - AB @ np.vstack((Extnt, np.outer(u_t, Ex_t))), in_axes=(0,0,0,None,0)), in_axes=(0,0,0,None,0))(smoothed["expected_states_squared"][:, 1:, :, :], smoothed["smoothed_cross_covariances"], smoothed['smoothed_means'][:, 1:, :], AB, u[:, 1:, :]).mean(axis=(0,1))
    
    # # init_params["prior_params"]["K"] = np.einsum("...i, ...j->...ij", u, RPM['mu']).sum(axis=(0, 1)) @ psd_solve((RPM['Sigma'] + np.einsum("...i, ...j->...ij", RPM['mu'], RPM['mu'])).sum(axis=(0, 1)), np.eye())
    # K = np.einsum("...i, ...j->...ij", u, smoothed['smoothed_means']).sum(axis=(0, 1)) @ psd_solve((smoothed["expected_states_squared"]).sum(axis=(0, 1)), np.eye(D))
    # params["prior_params"]["Abar"] = params["prior_params"]["A"] - params["prior_params"]["B"] @ K
    # params["prior_params"]["S"] = vmap(vmap(lambda u, K, Ex: np.outer(u, u) - K @ np.outer(Ex, u), in_axes=(0,None,0)), in_axes=(0,None,0))(u, K, smoothed['smoothed_means']).mean()

    params["prior_params"]["m1"] = smoothed['smoothed_means'][:, 0, :].mean(axis=(0,))
    mu_diff = smoothed['smoothed_means'][:, 0, :] - params["prior_params"]["m1"]
    params["prior_params"]["Q1"] = smoothed["smoothed_covariances"][:, 0, :, :].mean(axis=(0,)) + np.einsum("jk,jl->jkl", mu_diff, mu_diff).mean(axis = 0)

    # params["prior_params"]["A"] = truncate_singular_values(params["prior_params"]["A"])
    # # params["prior_params"]["Abar"] = truncate_singular_values(params["prior_params"]["Abar"])

    # params["prior_params"]["Q"] = (params["prior_params"]["Q"] + params["prior_params"]["Q"].T)/2
    # params["prior_params"]["Q"] += np.eye(D) * 1e-6
    # params["prior_params"]["Q1"] = (params["prior_params"]["Q1"] + params["prior_params"]["Q1"].T)/2
    # params["prior_params"]["Q1"] += np.eye(D) * 1e-6

    # params["prior_params"]["Q"] = np.linalg.cholesky(params["prior_params"]["Q"])[np.tril_indices(D)]
    # params["prior_params"]["Q1"] = np.linalg.cholesky(params["prior_params"]["Q1"])[np.tril_indices(D)]

    return params

def update_prior_one_step(carry, inputs):

    current_prior_params, prior, likelihood = carry
    u, t, RPM_J, RPM_h = inputs

    # edge (transition) potential
    A = current_prior_params['A']
    b = current_prior_params['B'] @ u
    x_dim = b.size
    L = psd_solve(current_prior_params['Q'], np.eye(x_dim))
    JT = np.block([[A.T @ L @ A, - A.T @ L], [- L @ A, L]])
    hT = np.concatenate((- A.T @ L @ b, L @ b))
    # J_prior = J_prior.at[x_dim * (t - 1) : x_dim * (t + 1), x_dim * (t - 1) : x_dim * (t + 1)].add(JT)
    # h_prior = h_prior.at[x_dim * (t - 1) : x_dim * (t + 1)].add(hT)
    prior['J'] = dynamic_slice_add(prior['J'], (x_dim * (t - 1), x_dim * (t - 1)), (x_dim * 2, x_dim * 2), JT)
    prior['h'] = dynamic_slice_add(prior['h'], (x_dim * (t - 1),), (x_dim * 2,), hT)

    # evidence potential
    # J_likelihood = J_likelihood.at[x_dim * t : x_dim * (t + 1), x_dim * t : x_dim * (t + 1)].add(RPM_J)
    # h_likelihood = h_likelihood.at[x_dim * t : x_dim * (t + 1)].add(RPM_h)
    likelihood['J'] = dynamic_slice_add(likelihood['J'], (x_dim * t, x_dim * t), (x_dim, x_dim), RPM_J)
    likelihood['h'] = dynamic_slice_add(likelihood['h'], (x_dim * t,), (x_dim,), RPM_h)

    carry = current_prior_params, prior, likelihood
    outputs = None

    return carry, outputs

def update_prior(current_prior_params, u, RPM_J, RPM_h):

    n_timepoints = RPM_J.shape[0]
    x_dim = RPM_J.shape[1]

    prior = {}
    prior['J']= np.zeros((x_dim * n_timepoints, x_dim * n_timepoints))
    prior['h'] = np.zeros((x_dim * n_timepoints))

    likelihood = {}
    likelihood['J'] = np.zeros((x_dim * n_timepoints, x_dim * n_timepoints))
    likelihood['h'] = np.zeros((x_dim * n_timepoints))

    # node (prior) potential
    K0 = psd_solve(current_prior_params['Q1'], np.eye(x_dim))
    h0 = K0 @ current_prior_params['m1']
    prior['J'] = prior['J'].at[:x_dim, :x_dim].set(K0)
    prior['h'] = prior['h'].at[:x_dim].set(h0)

    likelihood['J'] = likelihood['J'].at[:x_dim,:x_dim].add(RPM_J[0])
    likelihood['h'] = likelihood['h'].at[:x_dim].add(RPM_h[0])

    carry = current_prior_params, prior, likelihood
    inputs = u[:-1], np.arange(1, n_timepoints), RPM_J[1:], RPM_h[1:]
    (_, prior, likelihood), _ = scan(update_prior_one_step, carry, inputs)

    posterior = {}
    posterior['J'] = prior['J'] + likelihood['J']
    posterior['h'] = prior['h'] + likelihood['h']

    return prior, likelihood, posterior

def one_step_transition_action_integrated_out(carry, inputs):

    mu, P, prior_params = carry

    Abar = prior_params['A_F']
    B = prior_params['B']
    Q = prior_params['Q']
    l = prior_params['l']
    S = prior_params['S']

    mu = Abar @ mu + B @ l
    P = Abar @ P @ Abar.T + Q + B @ S @ B.T

    carry = mu, P, prior_params
    outputs = mu, P

    return carry, outputs

def one_step_transition_no_action(carry, inputs):

    mu, P, prior_params, Q_F = carry

    A_F = prior_params['A_F']
    b_F = prior_params['b_F']

    mu = A_F @ mu + b_F
    P = A_F @ P @ A_F.T + Q_F

    carry = mu, P, prior_params, Q_F
    outputs = mu, P

    return carry, outputs

def get_marginal_one_step(carry, inputs):

    J_prior_marg, J_posterior_marg, h_prior_marg, h_posterior_marg, Sigma_prior, Sigma_posterior, mu_prior, mu_posterior = carry
    t = inputs

    # J_prior_marg = J_prior_marg.at[t,:,:].set(psd_solve(Sigma_prior[x_dim * t : x_dim * (t + 1), x_dim * t : x_dim * (t + 1)], np.eye(x_dim)))
    # J_posterior_marg = J_posterior_marg.at[t,:,:].set(psd_solve(Sigma_posterior[x_dim * t : x_dim * (t + 1), x_dim * t : x_dim * (t + 1)], np.eye(x_dim)))
    # h_prior_marg = h_prior_marg.at[t,:].set(J_prior_marg[t,:,:] @ mu_prior[x_dim * t : x_dim * (t + 1)])
    # h_posterior_marg = h_posterior_marg.at[t,:].set(J_posterior_marg[t,:,:] @ mu_posterior[x_dim * t : x_dim * (t + 1)])
    
    x_dim = J_prior_marg.shape[-1]
    J_prior_t = psd_solve(dynamic_slice(Sigma_prior, (x_dim * t, x_dim * t), (x_dim, x_dim)), np.eye(x_dim))
    J_prior_marg = dynamic_update_slice(J_prior_marg, J_prior_t[None], (t,0,0))
    J_posterior_t = psd_solve(dynamic_slice(Sigma_posterior, (x_dim * t, x_dim * t), (x_dim, x_dim)), np.eye(x_dim))
    J_posterior_marg = dynamic_update_slice(J_posterior_marg, J_posterior_t[None], (t,0,0))
    h_prior_t = J_prior_t @ dynamic_slice(mu_prior, (x_dim * t,), (x_dim,))
    h_prior_marg = dynamic_update_slice(h_prior_marg, h_prior_t[None], (t,0))
    h_posterior_t = J_posterior_t @ dynamic_slice(mu_posterior, (x_dim * t,), (x_dim,))
    h_posterior_marg = dynamic_update_slice(h_posterior_marg, h_posterior_t[None], (t,0))

    carry = J_prior_marg, J_posterior_marg, h_prior_marg, h_posterior_marg, Sigma_prior, Sigma_posterior, mu_prior, mu_posterior
    outputs = None

    return carry, outputs

def get_marginal(J_prior, h_prior, J_posterior, h_posterior, D, T):

    Sigma_prior = psd_solve(J_prior, np.eye(J_prior.shape[-1]))
    Sigma_posterior = psd_solve(J_posterior, np.eye(J_posterior.shape[-1]))
    mu_prior = np.einsum("jk,k->j", Sigma_prior, h_prior)
    mu_posterior = np.einsum("jk,k->j", Sigma_posterior, h_posterior)

    J_prior_marg = np.zeros((T, D, D))
    J_posterior_marg = np.zeros((T, D, D))
    h_prior_marg = np.zeros((T, D))
    h_posterior_marg = np.zeros((T, D))

    carry = J_prior_marg, J_posterior_marg, h_prior_marg, h_posterior_marg, Sigma_prior, Sigma_posterior, mu_prior, mu_posterior
    inputs = np.arange(T)
    (J_prior_marg, J_posterior_marg, h_prior_marg, h_posterior_marg, _, _, _, _), _ = scan(get_marginal_one_step, carry, inputs)

    return J_prior_marg, J_posterior_marg, h_prior_marg, h_posterior_marg

# # Computes A.T @ Q^{-1} @ A in a way that's guaranteed to be symmetric
# def inv_quad_form(Q, A):
#     sqrt_Q = np.linalg.cholesky(Q)
#     trm = solve_triangular(sqrt_Q, A, lower=True, check_finite=False)
#     return trm.T @ trm

# def inv_symmetric(Q):
#     sqrt_Q = np.linalg.cholesky(Q)
#     sqrt_Q_inv = np.linalg.inv(sqrt_Q)
#     return sqrt_Q_inv.T @ sqrt_Q_inv

# def dynamics_to_tridiag(prior_params, T, D):
    
#     Q1, m1, A, Q = prior_params["Q1"], prior_params["m1"], prior_params["A"], prior_params["Q"]
    
#     # diagonal blocks of precision matrix
#     J = np.zeros((T, D, D))
#     J = J.at[0].add(inv_symmetric(Q1))
#     J = J.at[:-1].add(inv_quad_form(Q, A))
#     J = J.at[1:].add(inv_symmetric(Q))
    
#     # lower diagonal blocks of precision matrix
#     L = -np.linalg.solve(Q, A)
#     L = np.tile(L[None, :, :], (T - 1, 1, 1))
    
#     return { "J": J, "L": L}

def dynamics_to_tridiag(prior_params, T):
    
    Q1, m1, A, Q = prior_params["Q1"], prior_params["m1"], prior_params["A"], prior_params["Q"]
    D = m1.size

    Q_inv_A = psd_solve(Q, A)

    # diagonal blocks of precision matrix
    J = np.zeros((T, D, D))
    J = J.at[0].add(psd_solve(Q1, np.eye(Q1.shape[-1])))
    J = J.at[:-1].add(A.T @ Q_inv_A)
    J = J.at[1:].add(psd_solve(Q, np.eye(Q.shape[-1])))
    
    # upper diagonal blocks of precision matrix
    L = -Q_inv_A.T
    L = np.tile(L[None, :, :], (T - 1, 1, 1))
    
    return { "J": J, "L": L}

def get_beta_schedule(options):

    return optax.linear_schedule(options["beta_init_value"], options["beta_end_value"], options["beta_transition_steps"], options["beta_transition_begin"])

def get_group_name(options):

    if options['embed_u']:
        embed_u_str = "_EmbedU"
    else:
        embed_u_str = "_NoEmbedU"

    if options['use_LDS_for_F_in_q']:
        LDS_for_F_str = "_LDSForF"
    else:
        LDS_for_F_str = "_NoLDSForF"

    if options['use_GRU_for_F_in_q']:
        GRU_for_F_str = "_GRUForF"
    else:
        GRU_for_F_str = "_NoGRUForF"

    # if options['use_policy_loss']:
    #     policy_loss_str = "_PolicyLoss"
    # else:
    #     policy_loss_str = "_NoPolicyLoss"
    policy_loss_str = "_NoPolicyLoss"

    # if options['explicitly_integrate_out_u']:
    #     explicitly_integrate_out_u_str = "_ExplicitlyIntegrateOutU"
    # else:
    #     explicitly_integrate_out_u_str = "_NoExplicitlyIntegrateOutU"
    explicitly_integrate_out_u_str = "_NoExplicitlyIntegrateOutU"

    if options['f_time_dependent']:
        f_time_dependent_str = "_FTimeDepend"
    else:
        f_time_dependent_str = "_NoFTimeDepend"

    group_name = embed_u_str + LDS_for_F_str + GRU_for_F_str + policy_loss_str + explicitly_integrate_out_u_str + f_time_dependent_str

    return group_name

# log to https://wandb.ai/james-gatsby/projects
def log_to_wandb(loss, kl_qp, ce_qf, ce_qF, mu, y, options):
  
    group_name = get_group_name(options)

    wandb.init(project=options['project_name'], group=group_name, config=options, dir=options["save_dir"])

    D = mu.shape[-1]
    palette = sns.color_palette(None, D)

    cnt = 1
    n_rows = 3
    n_cols = 2
    for row in range(n_rows):
        for col in range(n_cols):
            plt.subplot(n_rows, n_cols, cnt)
            for d in range(D):
                if col == 0:
                    plt.plot(y[row,:,d],'--', c=palette[d])
                else:
                    plt.plot(mu[row,:,d],'--', c=palette[d])
            cnt += 1
    
    to_log = { "ELBO": -loss.mean(), "KL_qp": kl_qp.mean(), "CE_qf": ce_qf.mean(), "CE_qF": ce_qF.mean(), "CE_qf - CE_qF": (ce_qf - ce_qF).mean(), "inferred (q) states": plt}

    wandb.log(to_log)