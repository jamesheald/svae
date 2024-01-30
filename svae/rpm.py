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

    # def kl_posterior_prior(self, posterior_params, prior_params, 
    #                        samples=None):
    #     posterior = self.posterior.distribution(posterior_params)
    #     prior = self.prior.distribution(prior_params)
    #     if samples is None:
    #         return posterior.kl_divergence(prior)
    #     else:
    #         return np.mean(posterior.log_prob(samples) - prior.log_prob(samples))

    def elbo(self, key, data, target, # Added the new target parameter 
             u, batch_id, prior_marg_params, RPM_batch, model_params, sample_kl=False, **params):
        # rec_params = model_params["rec_params"]
        current_prior_params = self.prior.get_constrained_params(model_params["prior_params"], u)

        # current_prior_params = tree_map(lambda x: x[batch_id], prior_params_batch)

        potential = {}
        # potential['J'] = RPM_batch["J"][batch_id] - MM_prior["J"]
        # potential['h'] = RPM_batch["h"][batch_id] - MM_prior["h"]
        # potential['Sigma'] = vmap(lambda J, I: psd_solve(J + np.eye(3) * (abs(np.linalg.eigvals(J).min()) + 1e-4), I), in_axes=(0, None))(potential["J"], np.eye(self.prior.latent_dims))
        # potential['mu'] = np.einsum("ijk,ik->ij", potential['Sigma'], potential['h'])
        potential['Sigma'] = RPM_batch["Sigma"][batch_id]
        potential['mu'] = RPM_batch["mu"][batch_id]

        # Update: it makes more sense that inference is done in the posterior object
        posterior_params = self.posterior.infer(current_prior_params, potential, u)

        if params['stop_grad_q']:

            posterior_params = stop_gradient(posterior_params)

        # def node_potential(mu, L):

        #     # chapter 2.2.7.1 in Murphy, Probabilistic Machine Learning: Advanced Topics

        #     g = - 0.5 * (mu @ L @ mu + mu.size * np.log(2 * np.pi) + np.log(np.linalg.det(np.linalg.inv(L))))
        #     K = L
        #     h = L @ mu

        #     return g, K, h

        # def edge_potential(A, b, L):

        #     # chapter 2.2.7.5 in Murphy, Probabilistic Machine Learning: Advanced Topics
        #     # potential over (x_{t - 1}, x_t) or (x_t, y_t), in that order

        #     g = - 0.5 * (b @ L @ b + b.size * np.log(2 * np.pi) + np.log(np.linalg.det(np.linalg.inv(L)))) # is b.size correct dim?
        #     K = np.block([[A.T @ L @ A, - A.T @ L], [- L @ A, L]]) # is A transpose the right way?
        #     h = np.concatenate((- A.T @ L @ b, L @ b))

        #     return g, K, h

        # # transition potential
        # gT, KT, hT = edge_potential(prior_params['m1'], prior_params['B'] @ u[t - 1], np.linalg.inv(prior_params['Q']))

        # prior_mu = np.zeros((x_dim, T))
        # prior_cov = np.zeros((x_dim, x_dim, T))
        # posterior_mu = np.zeros((x_dim, T))
        # posterior_cov = np.zeros((x_dim, x_dim, T))
        # n = 0
        # for t in range(T):

        #     if t == 0:

        #         prior_mu[:, t] = mu0
        #         prior_cov[:, :, t] = np.linalg.inv(L0)

        #     else:

        #         prior_mu[:, t] = A @ prior_mu[:, t - 1] + b
        #         prior_cov[:, :, t] = np.linalg.inv(Lx) + A @ prior_cov[:, :, t - 1] @ A.T

        #     posterior_cov[:, :, t] = np.linalg.inv(np.linalg.inv(prior_cov[:, :, t]) + C.T @ Ly @ C)
        #     posterior_mu[:, t] = posterior_cov[:, :, t] @ (C.T @ Ly @ (y[:, t, n] - d) + np.linalg.inv(prior_cov[:, :, t]) @ prior_mu[:, t])

        #         # prior potential
        # g0, K0, h0 = node_potential(prior_params['m1'], np.linalg.inv(prior_params['Q1']))

        # g = g0
        # K = np.zeros((x_dim * T, x_dim * T))
        # K = K.at[].set(K0)
        # h = np.zeros((x_dim * T))
        # h = h.at[].set(h0)
        # for t in range(T):

        #     if t == 0:

        #         # prior
        #         g += g0
        #         K[:x_dim, :x_dim] += K0
        #         h[:x_dim] += h0

        #     else:

        #         # transition
        #         g += gT
        #         K[x_dim * (t - 1) : x_dim * (t + 1), x_dim * (t - 1) : x_dim * (t + 1)] += KT
        #         h[x_dim * (t - 1) : x_dim * (t + 1)] += hT

        #     # multiply by posterior factor (role of RPM factor)
        #     K[x_dim * t : x_dim * (t + 1), x_dim * t : x_dim * (t + 1)] += np.linalg.inv(posterior_cov[:, :, t])
        #     h[x_dim * t : x_dim * (t + 1)] += np.linalg.inv(posterior_cov[:, :, t]) @ posterior_mu[:, t]

        #     # divide by marginal prior factor
        #     K[x_dim * t : x_dim * (t + 1), x_dim * t : x_dim * (t + 1)] -= np.linalg.inv(prior_cov[:, :, t])
        #     h[x_dim * t : x_dim * (t + 1)] -= np.linalg.inv(prior_cov[:, :, t]) @ prior_mu[:, t]




        # Take samples under the posterior
        # num_samples = params.get("obj_samples") or 1
        # samples = self.posterior.sample(posterior_params, u, (num_samples,), key)

        delta_q = self.delta_nat_q.apply(model_params["delta_q_params"], data)

        # Compute kl terms
        posterior = self.posterior.distribution(posterior_params)

        # natural parameters of posterior marginals
        post_Ex = posterior.expected_states
        post_Sigma = posterior.smoothed_covariances
        posterior_J = vmap(lambda S, I: psd_solve(S, I), in_axes=(0, None))(post_Sigma, np.eye(self.prior.latent_dims))
        posterior_h = np.einsum("ijk,ik->ij", posterior_J, post_Ex)   

        if params['use_delta_nat_q']:
        
            posterior_J += delta_q['J']
            posterior_h += delta_q['h']
            posterior._smoothed_covariances = vmap(lambda J, I: psd_solve(J, I), in_axes=(0, None))(posterior_J, np.eye(self.prior.latent_dims))
            posterior._smoothed_means = np.einsum("ijk,ik->ij", posterior.smoothed_covariances, posterior_h)   

        # use all datapoints
        # normalised_auxillary_J = posterior_J[None] + RPM_batch["J"] - MM_prior["J"][None]
        # normalised_auxillary_h = posterior_h[None] + RPM_batch["h"] - MM_prior["h"][None]
        prior_J = vmap(lambda S, I: psd_solve(S, I), in_axes=(0, None))(prior_marg_params["Sigma"], np.eye(self.prior.latent_dims))
        prior_h = np.einsum("ijk,ik->ij", prior_J, prior_marg_params["Ex"])
        normalised_auxillary_J = RPM_batch["J"]
        normalised_auxillary_h = RPM_batch["h"]
        if params['use_ansatz']:

            normalised_auxillary_J += posterior_J[None]
            normalised_auxillary_h += posterior_h[None]
        # normalised_auxillary_J = prior_params['J_aux'][batch_id][None] + RPM_batch["J"]
        # normalised_auxillary_h = prior_params['h_aux'][batch_id][None] + RPM_batch["h"]

        if params['use_delta_nat_f_tilde']:

            delta_f_tilde = self.delta_nat_f_tilde.apply(model_params["delta_f_tilde_params"], data)
            normalised_auxillary_J += delta_f_tilde['J'][None]
            normalised_auxillary_h += delta_f_tilde['h'][None]

        if params['use_delta_nat_f_tilde_np']:

            normalised_auxillary_J += current_prior_params['J_aux'][batch_id][None]
            normalised_auxillary_h += current_prior_params['h_aux'][batch_id][None]

        # kl_qp, kl_qf = self.kl_terms(params, posterior, current_prior_params, u, normalised_auxillary_J[batch_id], normalised_auxillary_h[batch_id], delta_q)
        
        # expected log auxiliary factors < log \tilde{f} >
        # Murphy section 2.3.2.5
        # auxillary_J = optimal_prior_params["prior_J"] - posterior_J
        # auxillary_h = optimal_prior_params["prior_h"] - posterior_h
        # E_log_aux = np.einsum("ij,ij->i", auxillary_h, post_Ex) - 0.5 * (np.einsum("ij,ij->i", post_Ex, np.einsum("ijk,ik->ij", auxillary_J, post_Ex)) + (auxillary_J * post_Sigma).sum(axis = (1, 2)))
        # E_log_aux1 = np.einsum("ij,ij->i", auxillary_h, post_Ex)
        # E_log_aux2 = - 0.5 * np.einsum("ij,ij->i", post_Ex, np.einsum("ijk,ik->ij", auxillary_J, post_Ex))
        # E_log_aux3 = - 0.5 * (auxillary_J * post_Sigma).sum(axis = (1, 2))
        # E_log_aux = E_log_aux1 + E_log_aux2 + E_log_aux3

        # # use the current data point plus one other random data point
        # batch_size = RPM_batch["J"].shape[0]
        # timepoints = RPM_batch["J"].shape[1]
        # probs = np.ones(batch_size) / (batch_size - 1)
        # probs = probs.at[batch_id].set(0)
        # sample_datapoint = jr.choice(key, np.concatenate([np.arange(batch_size)]), p = probs, shape=(timepoints,))

        # RPM_J = np.concatenate((RPM_batch["J"][batch_id][None], RPM_batch["J"][sample_datapoint, np.arange(timepoints)][None]))
        # RPM_h = np.concatenate((RPM_batch["h"][batch_id][None], RPM_batch["h"][sample_datapoint, np.arange(timepoints)][None]))

        # # normalised_auxillary_J = posterior_J[None] + RPM_J - MM_prior["J"][None]
        # # normalised_auxillary_h = posterior_h[None] + RPM_h - MM_prior["h"][None]
        # normalised_auxillary_J = posterior_J[None] + RPM_J
        # normalised_auxillary_h = posterior_h[None] + RPM_h

        # normalised_auxillary_log_normaliser = self.log_normaliser(normalised_auxillary_J, normalised_auxillary_h)
        # # RPM_log_normaliser = self.log_normaliser(RPM_J, RPM_h)
        # RPM_log_normaliser = self.log_normaliser(optimal_prior_params["prior_J"] + RPM_J, optimal_prior_params["prior_h"] + RPM_h)

        # log_Gamma = logsumexp(normalised_auxillary_log_normaliser - RPM_log_normaliser, axis=0, b=np.array([1, batch_size - 1])[:, None].repeat(timepoints, axis = 1)/batch_size)
        # log_Gamma2 = (normalised_auxillary_log_normaliser - RPM_log_normaliser)[0]

        normalised_auxillary_log_normaliser = vmap(vmap(self.log_normaliser))(normalised_auxillary_J, normalised_auxillary_h)

        # use optimal prior 
        # RPM_log_normaliser = self.log_normaliser(optimal_prior_params["prior_J"][None] + RPM_batch["J"], optimal_prior_params["prior_h"][None] + RPM_batch["h"])

        # use (action conditioned) prior
        if params['use_prior_for_F']:

            RPM_log_normaliser = vmap(vmap(self.log_normaliser))(prior_J[None] + RPM_batch["J"], prior_h[None] + RPM_batch["h"])

        else:

            RPM_log_normaliser = vmap(vmap(self.log_normaliser))(RPM_batch["J"], RPM_batch["h"])

        log_gamma = ((normalised_auxillary_log_normaliser - RPM_log_normaliser)[batch_id] \
                    - logsumexp(normalised_auxillary_log_normaliser - RPM_log_normaliser, axis=0)).sum()

        # D = self.prior.latent_dims
        # L = np.linalg.cholesky(RPM_batch["J"][batch_id] + 1e-9 * np.eye(D)[None])
        # half_log_det = np.log(np.diagonal(L, axis1 = -2, axis2 = -1)).sum(-1)
        # g_f = - 0.5 * (b @ RPM_batch["J"][batch_id] @ b + D * np.log(2 * math.pi) - 2 * half_log_det)
        # L = np.linalg.cholesky(prior_J[batch_id] + 1e-9 * np.eye(D)[None])
        # half_log_det = np.log(np.diagonal(L, axis1 = -2, axis2 = -1)).sum(-1)
        # g_p = - 0.5 * (b @ L @ b + D * np.log(2 * math.pi) - 2 * half_log_det)
        # g_f = vmap(self.log_normaliser)(prior_J + RPM_batch["J"][batch_id], prior_h + RPM_batch["h"][batch_id])
        # g_p = vmap(self.log_normaliser)(prior_J, prior_h)
        # kl_qp_correction = (vmap(self.log_normaliser)(RPM_batch["J"][batch_id], RPM_batch["h"][batch_id]) - g_f + g_p).sum()

        # q_log_normaliser = self.posterior_log_normaliser(prior_params, prior_J, prior_h, RPM_batch, batch_id)

        D = self.prior.latent_dims
        T = RPM_log_normaliser.shape[1]
        prior_precision_upper_diagonal = block_diag(np.zeros((0, D)), *current_prior_params["L"].transpose((0, 2, 1)), np.zeros((D, 0)))
        prior_precision_main_diagonal = block_diag(*current_prior_params["J"])
        prior_precision_lower_diagonal = block_diag(np.zeros((D, 0)), *current_prior_params["L"], np.zeros((0, D)))
        prior_precision = prior_precision_upper_diagonal + prior_precision_main_diagonal + prior_precision_lower_diagonal
        posterior_precision = prior_precision + block_diag(*solve(posterior._emissions_covariances, np.eye(D)[None]))

        # log_normaliser_implied = self.log_normaliser(posterior_precision, posterior_precision @ posterior.expected_states.reshape(-1))
        # kl_correction = log_normaliser_implied - q_log_normaliser

        kl_qp = self.kl_qp_natural_parameters(posterior_precision, posterior_precision @ posterior.expected_states.reshape(-1), prior_precision, prior_precision @ current_prior_params["Ex"].reshape(-1))
        kl_qf = self.kl_qp_natural_parameters(posterior_precision, posterior_precision @ posterior.expected_states.reshape(-1), block_diag(*normalised_auxillary_J[batch_id]), normalised_auxillary_h[batch_id].reshape(-1))

        policy_loss = vmap(self.policy_loss, in_axes=(None,0,0,0))(current_prior_params, u, posterior.expected_states, posterior.smoothed_covariances).sum()

        # batch_size = RPM_log_normaliser.shape[0]
        # n_timepoints = RPM_log_normaliser.shape[1]
        # T_log_N = n_timepoints * np.log(batch_size)

        kl_qp /= data.size
        kl_qf /= data.size
        # kl_correction /= data.size
        log_gamma /= data.size
        # T_log_N /= data.size
        policy_loss /= data.size

        # free_energy = - (kl_qp + kl_correction) - (kl_qf + kl_correction) + log_gamma + T_log_N
        free_energy = policy_loss - kl_qp - kl_qf + log_gamma # + T_log_N - T_log_N

        # kl /= target.size
        # ell /= target.size
        # elbo = ell - kl

        rpm_Sigma = vmap(lambda S, I: psd_solve(S, I), in_axes=(0, None))(prior_J[batch_id] + RPM_batch["J"][batch_id], np.eye(self.prior.latent_dims))
        rpm_mu = np.einsum("ijk,ik->ij", rpm_Sigma, prior_h[batch_id] + RPM_batch["h"][batch_id])

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
            'policy_loss': policy_loss,
            "posterior_params": posterior_params,
            "posterior_means": posterior.expected_states,
            "rpm_means": rpm_mu
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

    def compute_objective(self, key, data, target, u, batch_id, prior_params_batch, RPM_batch, model_params, **params):
        if params["inference_method"] == "rpm":
            results = self.elbo(key, data, target, u, batch_id, prior_params_batch, RPM_batch, model_params, **params)
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

    def policy_loss(self, params, u, mu, Sigma, diagonal_boost = 1e-9):

        U = params['U']
        v = params['v']
        S = params['S']
        
        u_dim = u.size
        J = psd_solve(S, np.eye(u_dim))
        L = np.linalg.cholesky(J + diagonal_boost * np.eye(u_dim))
        half_log_det_J = np.log(np.diagonal(L)).sum()

        loss = 0.5 * (-u_dim * np.log(2 * np.pi) + 2 * half_log_det_J - (u - v) @ J @ (u - v) + 2 * (u - v) @ J @ U @ mu - (U @ mu).T @ J @ (U @ mu) - ((U.T @ J @ U) * Sigma).sum())

        return loss