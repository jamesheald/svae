# certified pure functions

import jax.numpy as np
from jax import scipy, vmap, lax
import jax.random as jr

import tensorflow_probability.substrates.jax.distributions as tfd
from tensorflow_probability.python.internal import reparameterization
MVN = tfd.MultivariateNormalFullCovariance
# MVN.reparameterization_type = reparameterization.FULLY_REPARAMETERIZED

from dynamax.utils.utils import psd_solve
# from svae.utils import sample_from_MVN

def _make_associative_sampling_elements(params, key, filtered_means, filtered_covariances):
    """Preprocess filtering output to construct input for smoothing assocative scan."""

    def _last_sampling_element(key, m, P):

        return np.zeros_like(P), MVN(m, P).sample(seed=key)
        # return np.zeros_like(P), sample_from_MVN(m, P, key)

    def _generic_sampling_element(params, key, m, P, u):

        F = params["A"]
        B = params["B"]
        Q = params["Q"]

        eps = 1e-9
        dims = F.shape[0]
        P += np.eye(dims) * eps
        Pp = F @ P @ F.T + Q

        E  = psd_solve(Pp, F @ P).T
        g  = m - E @ (F @ m + B @ u)
        # L  = P - E @ Pp @ E.T
        L = P - E @ F @ P

        L = (L + L.T) * .5 + np.eye(dims) * eps # Add eps to the crucial covariance matrix

        h = MVN(g, L).sample(seed=key)
        # h = sample_from_MVN(g, L, key)

        return E, h

    U = params["U"]

    num_timesteps = len(filtered_means)
    dims = filtered_means.shape[-1]
    keys = jr.split(key, num_timesteps)
    last_elems = _last_sampling_element(keys[-1], filtered_means[-1], 
                                        filtered_covariances[-1])
    generic_elems = vmap(_generic_sampling_element, (None, 0, 0, 0, 0))(
        params, keys[:-1], filtered_means[:-1], filtered_covariances[:-1],
        U[:-1])
    combined_elems = tuple(np.append(gen_elm, last_elm[None,:], axis=0)
                           for gen_elm, last_elm in zip(generic_elems, last_elems))
    return combined_elems

def _make_associative_filtering_elements(params, potentials, u):
    """Preprocess observations to construct input for filtering assocative scan."""
    # https://arxiv.org/pdf/1905.13002.pdf

    def _first_filtering_element(params, mu, Sigma):

        F = params["A"]
        B = params["B"]
        Q = params["Q"]
        P1 = params["Q1"]
        m1 = params["m1"]
        dim = Q.shape[0]
        H = np.eye(dim)

        y, R = mu, Sigma

        S1 = H @ P1 @ H.T + R
        K1 = psd_solve(S1, H @ P1).T

        A = np.zeros_like(F)
        b = m1 + K1 @ (y - H @ m1)
        C = P1 - K1 @ S1 @ K1.T
        
        eta = F.T @ H.T @ psd_solve(S1, y)
        J = F.T @ H.T @ psd_solve(S1, H @ F)

        # I think the below code is wrong (it calculates S as in _generic_filtering_element, as if for trial k > 1)
        # S = H @ Q @ H.T + R
        # eta = F.T @ H.T @ psd_solve(S, y)
        # J = F.T @ H.T @ psd_solve(S, H @ F)

        return A, b, C, J, eta

    def _generic_filtering_element(params, mu, Sigma, u_prev):

        F = params["A"]
        B = params["B"]
        Q = params["Q"]
        dim = Q.shape[0]
        H = np.eye(dim)

        y, R = mu, Sigma

        Bu = (B * u_prev).squeeze()

        S = H @ Q @ H.T + R
        K = psd_solve(S, H @ Q).T

        A = F - K @ H @ F
        b = Bu + K @ (y - H @ Bu)
        C = Q - K @ H @ Q

        eta = F.T @ H.T @ psd_solve(S, y - H @ Bu)
        J = F.T @ H.T @ psd_solve(S, H @ F)

        return A, b, C, J, eta

    mus, Sigmas = potentials["mu"], potentials["Sigma"]

    first_elems = _first_filtering_element(params, mus[0], Sigmas[0])
    generic_elems = vmap(_generic_filtering_element, (None, 0, 0, 0))(params, mus[1:], Sigmas[1:], u[:-1])
    combined_elems = tuple(np.concatenate((first_elm[None,...], gen_elm))
                           for first_elm, gen_elm in zip(first_elems, generic_elems))
    return combined_elems

def lgssm_filter(params, emissions, u):
    """A parallel version of the lgssm filtering algorithm.
    See S. Särkkä and Á. F. García-Fernández (2021) - https://arxiv.org/abs/1905.13002.
    Note: This function does not yet handle `inputs` to the system.
    """

    initial_elements = _make_associative_filtering_elements(params, emissions, u)

    @vmap
    def filtering_operator(elem1, elem2):
        A1, b1, C1, J1, eta1 = elem1
        A2, b2, C2, J2, eta2 = elem2
        dim = A1.shape[0]
        I = np.eye(dim)

        I_C1J2 = I + C1 @ J2
        temp = scipy.linalg.solve(I_C1J2.T, A2.T).T
        A = temp @ A1
        b = temp @ (b1 + C1 @ eta2) + b2
        C = temp @ C1 @ A2.T + C2

        I_J2C1 = I + J2 @ C1
        temp = scipy.linalg.solve(I_J2C1.T, A1).T

        eta = temp @ (eta2 - J2 @ b1) + eta1
        J = temp @ J2 @ A1 + J1

        return A, b, C, J, eta
    
    _, filtered_means, filtered_covs, _, _ = lax.associative_scan(
                                                filtering_operator, initial_elements
                                                )

    return {
        "filtered_means": filtered_means, 
        "filtered_covariances": filtered_covs
    }

def _make_associative_smoothing_elements(params, filtered_means, filtered_covariances, U):
    """Preprocess filtering output to construct input for smoothing assocative scan."""

    def _last_smoothing_element(m, P):

        return np.zeros_like(P), m, P

    def _generic_smoothing_element(params, m, P, u):

        F = params["A"]
        B = params["B"]
        Q = params["Q"]

        Pp = F @ P @ F.T + Q

        E = psd_solve(Pp, F @ P).T
        g = m - E @ (F @ m + B @ u)
        L = P - E @ F @ P

        return E, g, L

    last_elems = _last_smoothing_element(filtered_means[-1], filtered_covariances[-1])
    generic_elems = vmap(_generic_smoothing_element, (None, 0, 0, 0))(
        params, filtered_means[:-1], filtered_covariances[:-1], U[:-1]
        )
    combined_elems = tuple(np.append(gen_elm, last_elm[None,:], axis=0)
                           for gen_elm, last_elm in zip(generic_elems, last_elems))
    return combined_elems


def parallel_lgssm_smoother(params, emissions, u):
    """A parallel version of the lgssm smoothing algorithm.
    See S. Särkkä and Á. F. García-Fernández (2021) - https://arxiv.org/abs/1905.13002.
    Note: This function does not yet handle `inputs` to the system.
    """
    filtered_posterior = lgssm_filter(params, emissions, u)
    filtered_means = filtered_posterior["filtered_means"]
    filtered_covs = filtered_posterior["filtered_covariances"]
    initial_elements = _make_associative_smoothing_elements(params, filtered_means, filtered_covs, u)

    @vmap
    def smoothing_operator(elem1, elem2):
        E1, g1, L1 = elem1
        E2, g2, L2 = elem2

        E = E2 @ E1
        g = E2 @ g1 + g2
        L = E2 @ L1 @ E2.T + L2

        return E, g, L

    _, smoothed_means, smoothed_covs, *_ = lax.associative_scan(
                                                smoothing_operator, initial_elements, reverse=True
                                                )
    return {
        "filtered_means": filtered_means,
        "filtered_covariances": filtered_covs,
        "smoothed_means": smoothed_means,
        "smoothed_covariances": smoothed_covs
    }

def lgssm_log_normalizer(dynamics_params, mu_filtered, Sigma_filtered, potentials, u):
    p = dynamics_params
    Q, A, B = p["Q"][None], p["A"][None], p["B"][None]
    AT = (p["A"].T)[None]

    I = np.eye(Q.shape[-1])

    Sigma_filtered, mu_filtered = Sigma_filtered[:-1], mu_filtered[:-1]
    Sigma = Q + A @ Sigma_filtered @ AT

    mu = (A[0] @ mu_filtered.T + B[0] @ u[:-1].T).T
    # Append the first element
    Sigma_pred = np.concatenate([p["Q1"][None], Sigma])
    mu_pred = np.concatenate([p["m1"][None], mu])
    mu_rec, Sigma_rec = potentials["mu"], potentials["Sigma"]

    def log_Z_single(mu_pred, Sigma_pred, mu_rec, Sigma_rec):
        return MVN(loc=mu_pred, covariance_matrix=Sigma_pred+Sigma_rec).log_prob(mu_rec)

    log_Z = vmap(log_Z_single)(mu_pred, Sigma_pred, mu_rec, Sigma_rec)
    return np.sum(log_Z)