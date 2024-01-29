from jax import vmap
from jax.tree_util import tree_map
import jax.numpy as np
import jax.random as jr

import numpy as onp

from dynamax.utils.utils import psd_solve
import tensorflow_probability.substrates.jax.distributions as tfd
MVN = tfd.MultivariateNormalFullCovariance

from jax.scipy.special import logsumexp

from jax.scipy.linalg import block_diag
from jax.numpy.linalg import solve

from jax.lax import stop_gradient

from svae.datasets import LDS

class RPM:
    def __init__(self,
                 recognition=None, prior=None, posterior=None,
                 delta_nat_q=None, delta_nat_f_tilde=None,
                 input_dummy=None, latent_dummy=None, u_dummy=None):
        """
        rec_net, dec_net, prior are all objects that take in parameters
        rec_net.apply(params, data) returns Gaussian potentials (parameters)
        dec_net.apply(params, latents) returns probability distributions
        prior : SVAEPrior
        """
        self.recognition = recognition
        self.prior = prior
        self.posterior = posterior
        self.delta_nat_q = delta_nat_q
        self.delta_nat_f_tilde = delta_nat_f_tilde
        self.input_dummy = input_dummy
        self.latent_dummy = latent_dummy
        self.u_dummy = u_dummy

    def init(self, key=None):
        if key is None:
            key = jr.PRNGKey(0)

        rec_key, prior_key, post_key, delta_q_key, delta_f_tilde_key = jr.split(key, 5)

        return {
            "rec_params": self.recognition.init(rec_key, self.input_dummy),
            "prior_params": self.prior.init(prior_key),
            "post_params": self.posterior.init(post_key, self.u_dummy),
            "delta_q_params": self.delta_nat_q.init(delta_q_key, self.input_dummy),
            "delta_f_tilde_params": self.delta_nat_f_tilde.init(delta_f_tilde_key, self.input_dummy),
            # "delta_f_tilde_params": self.delta_nat_f_tilde.init(delta_f_tilde_key),
        }

    def elbo(self, key, data, target, # Added the new target parameter 
             u, batch_id, prior, posterior, prior_marg, posterior_marg, RPM_batch, model_params, sample_kl=False, **params):

        J_prior, h_prior = prior
        J_posterior, h_posterior = posterior
        J_prior_marg, h_prior_marg = prior_marg
        J_posterior_marg, h_posterior_marg = posterior_marg

        # normalised_auxillary_J_full = J_posterior - prior_J[batch_id][None]
        # normalised_auxillary_h_full = h_posterior - prior_h[batch_id][None]
        # if train_params['use_ansatz']:

        #     normalised_auxillary_J_full += J_posterior[batch_id][None]
        #     normalised_auxillary_h_full += h_posterior[batch_id][None]

        # normalised_auxillary_J = np.zeros((batch_size, n_timepoints, x_dim, x_dim))
        # normalised_auxillary_h = np.zeros((batch_size, n_timepoints, x_dim))
        # for batch_id in range(batch_size):
        #     for t in range(n_timepoints):
        #         normalised_auxillary_J = prior_J + RPM_batch["J"] - prior_J[batch_id][None] + posterior_J[None] normalised_auxillary_J.at[batch_id,t,:,:].set(normalised_auxillary_J_full)
        #         normalised_auxillary_h = normalised_auxillary_h.at[batch_id,t,:].set(normalised_auxillary_h_full)

        # normalised_auxillary_log_normaliser = vmap(vmap(self.log_normaliser))(normalised_auxillary_J, normalised_auxillary_h)

        # RPM_log_normaliser = vmap(vmap(self.log_normaliser))(prior_J + RPM_batch["J"], prior_h + RPM_batch["h"])

        # log_gamma = ((normalised_auxillary_log_normaliser - RPM_log_normaliser)[batch_id] \
        #             - logsumexp(normalised_auxillary_log_normaliser - RPM_log_normaliser, axis=0)).sum()

        normalised_auxillary_J = J_prior_marg + RPM_batch["J"] - J_prior_marg[batch_id][None] # i think this is equal to RPM_batch["J"] as prior_J is independnet of u
        normalised_auxillary_h = h_prior_marg + RPM_batch["h"] - h_prior_marg[batch_id][None]
        if params['use_ansatz']:

            normalised_auxillary_J += J_posterior_marg[batch_id][None]
            normalised_auxillary_h += h_posterior_marg[batch_id][None]

        normalised_auxillary_log_normaliser = vmap(vmap(self.log_normaliser))(normalised_auxillary_J, normalised_auxillary_h)

        RPM_log_normaliser = vmap(vmap(self.log_normaliser))(J_prior_marg + RPM_batch["J"], h_prior_marg + RPM_batch["h"])

        log_gamma = ((normalised_auxillary_log_normaliser - RPM_log_normaliser)[batch_id] \
                    - logsumexp(normalised_auxillary_log_normaliser - RPM_log_normaliser, axis=0)).sum()

        # rpm_batch_J = prior_J + vmap(lambda J: block_diag(*J))(RPM_batch["J"])
        # rpm_batch_h = prior_h + RPM_batch["h"].reshape(RPM_batch["h"].shape[0], -1)

        # normalised_auxillary_J = rpm_batch - prior_J[batch_id][None] + posterior_J[batch_id][None]
        # normalised_auxillary_h = rpm_batch_h - prior_h[batch_id][None] + posterior_h[batch_id][None]
        # # if params['use_ansatz']:

        # # normalised_auxillary_J += posterior_J[batch_id][None]
        # # normalised_auxillary_h += posterior_h[batch_id][None]

        # normalised_auxillary_log_normaliser = vmap(self.log_normaliser)(normalised_auxillary_J, normalised_auxillary_h)

        # RPM_log_normaliser = vmap(vmap(self.log_normaliser))(rpm_batch, prior_h + RPM_batch["h"])

        # log_gamma = ((normalised_auxillary_log_normaliser - RPM_log_normaliser)[batch_id] \
        #             - logsumexp(normalised_auxillary_log_normaliser - RPM_log_normaliser, axis=0)).sum()

        kl_qp = self.kl_qp_natural_parameters(J_posterior[batch_id], h_posterior[batch_id], J_prior[batch_id], h_prior[batch_id])
        kl_qf = self.kl_qp_natural_parameters(J_posterior[batch_id], h_posterior[batch_id], block_diag(*normalised_auxillary_J[batch_id]), normalised_auxillary_h[batch_id].reshape(-1))

        # D = self.prior.latent_dims
        # T = RPM_batch["J"].shape[1]
        # mean_difference = (mu_posterior - mu_prior)
        # L = np.linalg.cholesky(J_posterior[batch_id] + 1e-9 * np.eye(D * T))
        # half_log_det_posterior_precision = np.log(np.diagonal(L, axis1 = -2, axis2 = -1)).sum(-1)
        # L = np.linalg.cholesky(J_prior[batch_id] + 1e-9 * np.eye(D * T))
        # half_log_det_prior_precision = np.log(np.diagonal(L, axis1 = -2, axis2 = -1)).sum(-1)
        # det_term = 2 * (half_log_det_posterior_precision - half_log_det_prior_precision)
        # mean_diff_term = mean_difference @ J_prior[batch_id] @ mean_difference
        # tr_term = np.trace(solve(J_posterior[batch_id], J_prior[batch_id]).T)
        # kl_qp2 = 0.5 * (det_term + mean_diff_term + tr_term - D * T)
        # kl_qp2 /= data.size

        kl_qp /= data.size
        kl_qf /= data.size
        log_gamma /= data.size

        free_energy = - kl_qp - kl_qf + log_gamma

        Sigma_posterior = psd_solve(J_posterior[batch_id], np.eye(J_posterior[batch_id].shape[-1]))
        mu_posterior = np.einsum("ij,j->i", Sigma_posterior, h_posterior[batch_id])

        # Sigma_prior = psd_solve(J_prior[batch_id], np.eye(J_prior[batch_id].shape[-1]))
        # mu_prior = np.einsum("ij,j->i", Sigma_prior, h_prior[batch_id])

        return {
            "free_energy": free_energy,
            # "elbo": elbo,
            "ell": 0.,
            "kl": 0.,
            "kl_qp": kl_qp,
            # "kl_correction": kl_correction,
            "kl_qf": kl_qf,
            # "E_log_aux": E_log_aux.sum(),
            # "E_log_aux1": E_log_aux1.sum(),
            # "E_log_aux2": E_log_aux2.sum(),
            # "E_log_aux3": E_log_aux3.sum(),
            "log_Gamma": log_gamma,
            # "log_Gamma2": log_Gamma2.sum(),
            # "posterior_params": posterior_params,
            "posterior_means": mu_posterior.reshape(data.shape[0], data.shape[1]),
            "rpm_means": mu_posterior.reshape(data.shape[0], data.shape[1]),
            # "rpm_means": prior_J + RPM_batch["J"], prior_h + RPM_batch["h"]
            # 'log_normaliser_implied2': log_Z,
            # 'log_normaliser_implied': log_normaliser_implied,
            # "posterior_expected_states_squared": posterior.expected_states_squared,
            # "posterior_expected_states_next_states": posterior.expected_states_next_states,
            # "posterior_samples": samples,
            # "reconstruction": mean,
            # "mask": mask,
            # "mean": mean,
            # "cov": covv,
            # "ells": ells,
            # "sample_kl": sample_kl,
        }

    def compute_objective(self, key, data, target, u, batch_id, prior, posterior, prior_marg, posterior_marg, RPM_batch, model_params, **params):
        if params["inference_method"] == "rpm":
            results = self.elbo(key, data, target, u, batch_id, prior, posterior, prior_marg, posterior_marg, RPM_batch, model_params, **params)
            results["objective"] = results["free_energy"]
        elif params["inference_method"] == "lds":
            lds = LDS(data.shape[-1], u.shape[-1], data.shape[0])
            prior_params = self.prior.get_constrained_params(model_params["prior_params"], u)
            mll, posterior_mean = lds.marginal_log_likelihood(prior_params, u, data)
            mll /= data.size
            results = {
                        "free_energy": mll,
                        "ell": 0.,
                        "kl": 0.,
                        "kl_qp": 0.,
                        "kl_qf": 0.,
                        "log_Gamma": 0.,
                        "posterior_params": 0.,
                        "posterior_means": posterior_mean,
                        "rpm_means": posterior_mean
                    }
            results["objective"] = results["free_energy"]
        return results

class RPMLDS(RPM):
    def kl_posterior_rpmfactors(self, normalised_auxillary_J, normalised_auxillary_h, posterior, posterior_entropy, u):

        # rpm_J = prior_params['J_aux'][batch_id] + RPM_batch["J"][batch_id]
        # rpm_h = prior_params['h_aux'][batch_id] + RPM_batch["h"][batch_id]

        # rpm_J = optimal_prior_params["prior_J"] + RPM_batch["J"][batch_id]
        # rpm_h = optimal_prior_params["prior_h"] + RPM_batch["h"][batch_id]

        normalised_auxillary_Sigma = vmap(lambda S, I: psd_solve(S, I), in_axes=(0, None))(normalised_auxillary_J, np.eye(self.prior.latent_dims))
        normalised_auxillary_mu = np.einsum("ijk,ik->ij", normalised_auxillary_Sigma, normalised_auxillary_h)

        cross_entropy = 0.5 * np.einsum("tij,tij->", normalised_auxillary_J, posterior.smoothed_covariances)
        cross_entropy -= MVN(loc=normalised_auxillary_mu, covariance_matrix=normalised_auxillary_Sigma).log_prob(posterior.expected_states).sum()

        # cross_entropy = 0.5 * np.einsum("tij,tij->", RPM_batch["J"][batch_id], posterior.smoothed_covariances)
        # cross_entropy -= MVN(loc=RPM_batch["mu"][batch_id], covariance_matrix=RPM_batch["Sigma"][batch_id]).log_prob(posterior.expected_states).sum()

        return cross_entropy - posterior_entropy

    def kl_posterior_prior(self, params, prior, prior_params, posterior, posterior_entropy, u):

        # https://www.lix.polytechnique.fr/~nielsen/EntropyEF-ICIP2010.pdf
        # Murphy 2, 5.1.8.1 KL is a Bregman divergence

        # D = self.prior.latent_dims
        # T = prior._expected_states.shape[0]
        # prior_precision_upper_diagonal = block_diag(np.zeros((0, D)), *prior_params["L"].transpose((0, 2, 1)), np.zeros((D, 0)))
        # prior_precision_main_diagonal = block_diag(*prior_params["J"])
        # prior_precision_lower_diagonal = block_diag(np.zeros((D, 0)), *prior_params["L"], np.zeros((0, D)))
        # prior_precision = prior_precision_upper_diagonal + prior_precision_main_diagonal + prior_precision_lower_diagonal
        # posterior_precision = prior_precision + block_diag(*solve(posterior._emissions_covariances, np.eye(D)[None]))
        # mean_difference = (posterior.smoothed_means - prior._expected_states).reshape(D * T)
        # # det_term = np.log(np.linalg.det(posterior_precision)) - np.log(np.linalg.det(prior_precision))
        # L = np.linalg.cholesky(posterior_precision + 1e-9 * np.eye(D * T))
        # half_log_det_posterior_precision = np.log(np.diagonal(L, axis1 = -2, axis2 = -1)).sum(-1)
        # L = np.linalg.cholesky(prior_precision + 1e-9 * np.eye(D * T))
        # half_log_det_prior_precision = np.log(np.diagonal(L, axis1 = -2, axis2 = -1)).sum(-1)
        # det_term = 2 * (half_log_det_posterior_precision - half_log_det_prior_precision)
        # mean_diff_term = mean_difference @ prior_precision @ mean_difference
        # tr_term = np.trace(solve(posterior_precision, prior_precision).T)
        # mykl2 = 0.5 * (det_term + mean_diff_term + tr_term - D * T)

        # prior_mean = prior._expected_states.reshape(D * T)
        # posterior_mean = posterior.smoothed_means.reshape(D * T)
        # cross_entropy1 = 0.5 * np.trace(solve(posterior_precision, prior_precision).T) - prior.log_prob(posterior.smoothed_means) 
        # cross_entropy2 = 0.5 * np.trace(solve(posterior_precision, prior_precision).T) - MVN(loc=prior_mean, covariance_matrix=solve(prior_precision, np.eye(D * T))).log_prob(posterior_mean)
        # posterior_entropy1 = 0.5 * (D * T + D * T * np.log(2 * np.pi) - 2 * half_log_det_posterior_precision)
        # posterior_entropy2 = 0.5 * D * T - MVN(loc=posterior_mean, covariance_matrix=solve(posterior_precision, np.eye(D * T))).log_prob(posterior_mean)

        # mykl3 = cross_entropy1 - posterior_entropy1
        # mykl4 = cross_entropy1 - posterior_entropy2
        # mykl5 = cross_entropy2 - posterior_entropy1
        # mykl6 = cross_entropy2 - posterior_entropy2

        # delta_log_normalizer = log_normalizer_q - log_normalizer_p
        # delta_natural1 = natural_p[0] - natural_q[0]
        # delta_natural2 = natural_p[1] - natural_q[1]
        # suff_stat1 = suff_stat_mean_p[0]
        # suff_stat2 = suff_stat_mean_p[1]
        # term1 = matmul(delta_natural1.unsqueeze(-2), suff_stat1.unsqueeze(-1)).squeeze(-1).squeeze(-1)
        # term2 = (delta_natural2 * suff_stat2).sum((-2, -1))
        # return delta_log_normalizer + term1 + term2

        # https://en.wikipedia.org/wiki/Trace_(linear_algebra)#Trace_of_a_product
        # tr(prior_precision @ posterior_covariance)
        # the trace of a matrix product is the sum of the elements of the Hadamard product 
        # subblocks with zero prior precision (i.e. those not on the tridiagonal) can therefore be ignored
        cross_entropy = 0.5 * np.einsum("tij,tij->", prior_params["J"], posterior.smoothed_covariances)
        if not params['use_delta_nat_q']:

            # assume time points are independent in q
            Sigmatnt = posterior.expected_states_next_states - np.einsum("ti,tj->tij", posterior.expected_states[:-1], posterior.expected_states[1:])
            # below, transpose Sigmatnt, as L is the LOWER diagonal of the prior precision matrix
            cross_entropy += np.einsum("tij,tij->", prior_params["L"], Sigmatnt.transpose((0, 2, 1))) # no 0.5 weighting because this term is counted twice (once for the lower diagonal and once for the upper diagonal)

        cross_entropy -= prior.log_prob(posterior.expected_states)

        return cross_entropy - posterior_entropy

    def kl_terms(self, params, posterior, prior_params, u, normalised_auxillary_J, normalised_auxillary_h, delta_q):

        prior = self.prior.distribution(prior_params)

        D = self.prior.latent_dims
        T = prior._expected_states.shape[0]
        if params['use_delta_nat_q']:

            # # don't assume time points are independent in q
            # prior_precision_upper_diagonal = block_diag(np.zeros((0, D)), *prior_params["L"].transpose((0, 2, 1)), np.zeros((D, 0)))
            # prior_precision_main_diagonal = block_diag(*prior_params["J"])
            # prior_precision_lower_diagonal = block_diag(np.zeros((D, 0)), *prior_params["L"], np.zeros((0, D)))
            # prior_precision = prior_precision_upper_diagonal + prior_precision_main_diagonal + prior_precision_lower_diagonal
            # posterior_precision = prior_precision + block_diag(*solve(posterior._emissions_covariances, np.eye(D)[None])) + block_diag(*delta_q['J'])

            # L = np.linalg.cholesky(posterior_precision + 1e-9 * np.eye(D * T))
            # half_log_det_posterior_precision = np.log(np.diagonal(L, axis1 = -2, axis2 = -1)).sum(-1)

            # k = D*T
            # posterior_entropy = k/2 + k/2*np.log(2*np.pi) - half_log_det_posterior_precision

            # assume time points are independent in q
            L = np.linalg.cholesky(posterior.smoothed_covariances + 1e-9 * np.eye(D)[None])
            half_log_det_posterior_sigma = np.log(np.diagonal(L, axis1 = -2, axis2 = -1)).sum(-1)
            posterior_entropy = (D/2 + D/2*np.log(2*np.pi) + half_log_det_posterior_sigma).sum()

        else:

            log_p_post = posterior.log_prob(posterior.expected_states, u=u)
            posterior_entropy = 0.5 * D * T - log_p_post
            # posterior_entropy = 0.5 * D * T - posterior.log_prob(posterior.expected_states, u=u)

        kl_qp = self.kl_posterior_prior(params, prior, prior_params, posterior, posterior_entropy, u)

        kl_qf = self.kl_posterior_rpmfactors(normalised_auxillary_J, normalised_auxillary_h, posterior, posterior_entropy, u)

        return kl_qp, kl_qf

    def log_normaliser(self, J, h, diagonal_boost = 1e-9):

        # https://en.wikipedia.org/wiki/Exponential_family

        # https://math.stackexchange.com/questions/3158303/using-cholesky-decomposition-to-compute-covariance-matrix-determinant
        L = np.linalg.cholesky(J + diagonal_boost * np.eye(J.shape[-1]))
        half_log_det_precision = np.log(np.diagonal(L)).sum()

        return 0.5 * h @ psd_solve(J, h) - half_log_det_precision

    def log_normaliser_sigma_mu(self, Sigma, mu, diagonal_boost = 1e-9):

        # https://en.wikipedia.org/wiki/Exponential_family

        # https://math.stackexchange.com/questions/3158303/using-cholesky-decomposition-to-compute-covariance-matrix-determinant
        L = np.linalg.cholesky(Sigma + diagonal_boost * np.eye(Sigma.shape[-1]))
        half_log_det_sigma = np.log(np.diagonal(L)).sum()

        return 0.5 * mu @ psd_solve(Sigma, mu) + half_log_det_sigma

    def posterior_log_normaliser(self, prior_params, prior_J, prior_h, RPM_batch, batch_id):

        # prior potential
        # g_0 = -self.log_normaliser_sigma_mu(prior_params["Q1"], prior_params["m1"])

        # transition potential
        # g_t = -vmap(self.log_normaliser_sigma_mu, in_axes=(None, 0))(prior_params["Q"], prior_params['bs'])

        # prior potential (the first timestep of Qs and Bs is Q1 and m1, respectively) and transition potentials
        g_0_t = -vmap(self.log_normaliser_sigma_mu)(prior_params["Qs"], prior_params['bs'])
            
        # rpm factor potential
        g_f = -vmap(self.log_normaliser)(prior_J + RPM_batch["J"][batch_id], prior_h + RPM_batch["h"][batch_id])
        
        # marginal prior potential (approximation of F)
        g_p = -vmap(self.log_normaliser)(prior_J, prior_h)

        return -(g_0_t.sum() + g_f.sum() - g_p.sum()) # log normaliser of posterior is sum of log normalizers of individual factors

    def kl_qp_natural_parameters(self, J_q, h_q, J_p, h_p):

        J_diff = J_q - J_p
        h_diff = h_q - h_p

        Sigma_q = psd_solve(J_q, np.eye(h_q.size))
        mu_q = Sigma_q @ h_q

        trm = np.einsum("i,i->", h_diff, mu_q) - 0.5 * (np.einsum("i,i->", mu_q, np.einsum("ij,j->i", J_diff, mu_q)) + (J_diff * Sigma_q).sum(axis = (0, 1)))

        log_normaliser_q = self.log_normaliser(J_q, h_q)
        log_normaliser_p = self.log_normaliser(J_p, h_p)

        return trm + log_normaliser_p - log_normaliser_q