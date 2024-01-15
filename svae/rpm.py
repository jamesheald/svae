from jax import vmap
from jax.tree_util import tree_map
import jax.numpy as np
import jax.random as jr

import numpy as onp

from dynamax.utils.utils import psd_solve
import tensorflow_probability.substrates.jax.distributions as tfd
MVN = tfd.MultivariateNormalFullCovariance

from jax.scipy.special import logsumexp

# from jax.scipy.linalg import block_diag
# from jax.numpy.linalg import solve

# from jax.lax import stop_gradient

class RPM:
    def __init__(self,
                 recognition=None, prior=None, posterior=None,
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
        self.input_dummy = input_dummy
        self.latent_dummy = latent_dummy
        self.u_dummy = u_dummy

    def init(self, key=None):
        if key is None:
            key = jr.PRNGKey(0)

        rec_key, prior_key, post_key = jr.split(key, 3)

        return {
            "rec_params": self.recognition.init(rec_key, self.input_dummy),
            "prior_params": self.prior.init(prior_key),
            "post_params": self.posterior.init(post_key, self.u_dummy)
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
             u, batch_id, optimal_prior_params, RPM_batch, model_params, sample_kl=False, **params):
        # rec_params = model_params["rec_params"]
        prior_params = self.prior.get_constrained_params(model_params["prior_params"], u)

        potential = {}
        # potential['J'] = RPM_batch["J"][batch_id] - MM_prior["J"]
        # potential['h'] = RPM_batch["h"][batch_id] - MM_prior["h"]
        # potential['Sigma'] = vmap(lambda J, I: psd_solve(J + np.eye(3) * (abs(np.linalg.eigvals(J).min()) + 1e-4), I), in_axes=(0, None))(potential["J"], np.eye(self.prior.latent_dims))
        # potential['mu'] = np.einsum("ijk,ik->ij", potential['Sigma'], potential['h'])
        potential['Sigma'] = RPM_batch["Sigma"][batch_id]
        potential['mu'] = RPM_batch["mu"][batch_id]

        # Update: it makes more sense that inference is done in the posterior object
        posterior_params = self.posterior.infer(prior_params, potential, u)




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

        # Compute kl terms
        posterior = self.posterior.distribution(posterior_params)
        if sample_kl:
            kl_qp, kl_qf = self.kl_terms(posterior, prior_params, u, RPM_batch, batch_id, optimal_prior_params, samples=samples)
        else:
            kl_qp, kl_qf = self.kl_terms(posterior, prior_params, u, RPM_batch, batch_id, optimal_prior_params)

        # natural parameters of posterior marginals
        post_Ex = posterior.expected_states
        post_Sigma = posterior.smoothed_covariances
        posterior_J = vmap(lambda S, I: psd_solve(S, I), in_axes=(0, None))(post_Sigma, np.eye(self.prior.latent_dims))
        posterior_h = np.einsum("ijk,ik->ij", posterior_J, post_Ex)
        
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

        # use all datapoints
        # normalised_auxillary_J = posterior_J[None] + RPM_batch["J"] - MM_prior["J"][None]
        # normalised_auxillary_h = posterior_h[None] + RPM_batch["h"] - MM_prior["h"][None]
        normalised_auxillary_J = posterior_J[None] + RPM_batch["J"]
        normalised_auxillary_h = posterior_h[None] + RPM_batch["h"]
        # normalised_auxillary_J = prior_params['J_aux'][batch_id][None] + RPM_batch["J"]
        # normalised_auxillary_h = prior_params['h_aux'][batch_id][None] + RPM_batch["h"]

        normalised_auxillary_log_normaliser = self.log_normaliser(normalised_auxillary_J, normalised_auxillary_h)

        # use optimal prior 
        # RPM_log_normaliser = self.log_normaliser(optimal_prior_params["prior_J"][None] + RPM_batch["J"], optimal_prior_params["prior_h"][None] + RPM_batch["h"])

        # use action conditioned prior
        prior_J = vmap(lambda S, I: psd_solve(S, I), in_axes=(0, None))(prior_params["Sigma"], np.eye(self.prior.latent_dims))
        prior_h = np.einsum("ijk,ik->ij", prior_J, prior_params["Ex"])
        RPM_log_normaliser = self.log_normaliser(prior_J[None] + RPM_batch["J"], prior_h[None] + RPM_batch["h"])

        batch_size = RPM_log_normaliser.shape[0]
        log_gamma = ((normalised_auxillary_log_normaliser - RPM_log_normaliser)[batch_id] \
                    - logsumexp(normalised_auxillary_log_normaliser - RPM_log_normaliser, axis=0)).sum()

        batch_size = RPM_log_normaliser.shape[0]
        n_timepoints = RPM_log_normaliser.shape[1]
        T_log_N = n_timepoints * np.log(batch_size)

        kl_qp /= data.size
        kl_qf /= data.size
        log_gamma /= data.size
        T_log_N /= data.size

        free_energy = - kl_qp - kl_qf + log_gamma + T_log_N

        # kl /= target.size
        # ell /= target.size
        # elbo = ell - kl

        return {
            "free_energy": free_energy,
            # "elbo": elbo,
            "ell": 0.,
            "kl": 0.,
            "kl_qp": kl_qp,
            "kl_qf": kl_qf,
            # "E_log_aux": E_log_aux.sum(),
            # "E_log_aux1": E_log_aux1.sum(),
            # "E_log_aux2": E_log_aux2.sum(),
            # "E_log_aux3": E_log_aux3.sum(),
            "log_Gamma": log_gamma,
            # "log_Gamma2": log_Gamma2.sum(),
            "posterior_params": posterior_params,
            # "posterior_samples": samples,
            # "reconstruction": mean,
            # "mask": mask,
            # "mean": mean,
            # "cov": covv,
            # "ells": ells,
            # "sample_kl": sample_kl,
        }

    def compute_objective(self, key, data, target, u, batch_id, optimal_prior_params, RPM_batch, model_params, **params):
        results = self.elbo(key, data, target, u, batch_id, optimal_prior_params, RPM_batch, model_params, **params)
        results["objective"] = results["free_energy"]
        return results

class RPMLDS(RPM):
    def kl_posterior_rpmfactors(self, optimal_prior_params, prior_params, RPM_batch, batch_id, posterior, posterior_entropy, u, samples):

        post_Ex = posterior.expected_states
        post_Sigma = posterior.smoothed_covariances
        posterior_J = vmap(lambda S, I: psd_solve(S, I), in_axes=(0, None))(post_Sigma, np.eye(self.prior.latent_dims))
        posterior_h = np.einsum("ijk,ik->ij", posterior_J, post_Ex)
        
        rpm_J = posterior_J + RPM_batch["J"][batch_id]
        rpm_h = posterior_h + RPM_batch["h"][batch_id]

        # rpm_J = prior_params['J_aux'][batch_id] + RPM_batch["J"][batch_id]
        # rpm_h = prior_params['h_aux'][batch_id] + RPM_batch["h"][batch_id]

        # rpm_J = optimal_prior_params["prior_J"] + RPM_batch["J"][batch_id]
        # rpm_h = optimal_prior_params["prior_h"] + RPM_batch["h"][batch_id]

        rpm_Sigma = vmap(lambda S, I: psd_solve(S, I), in_axes=(0, None))(rpm_J, np.eye(self.prior.latent_dims))
        rpm_mu = np.einsum("ijk,ik->ij", rpm_Sigma, rpm_h)

        if samples is None:

            cross_entropy = 0.5 * np.einsum("tij,tij->", rpm_J, posterior.smoothed_covariances)
            cross_entropy -= MVN(loc=rpm_mu, covariance_matrix=rpm_Sigma).log_prob(posterior.expected_states).sum()

            # cross_entropy = 0.5 * np.einsum("tij,tij->", RPM_batch["J"][batch_id], posterior.smoothed_covariances)
            # cross_entropy -= MVN(loc=RPM_batch["mu"][batch_id], covariance_matrix=RPM_batch["Sigma"][batch_id]).log_prob(posterior.expected_states).sum()

            return cross_entropy - posterior_entropy

        else:

            return np.mean(posterior.log_prob(samples, u=u) - MVN(loc=rpm_mu, covariance_matrix=rpm_Sigma).log_prob(samples).sum())

    def kl_posterior_prior(self, prior, prior_params, posterior, posterior_entropy, u, samples=None):

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

        if samples is None:

            # https://en.wikipedia.org/wiki/Trace_(linear_algebra)#Trace_of_a_product
            # tr(prior_precision @ posterior_covariance)
            # the trace of a matrix product is the sum of the elements of the Hadamard product 
            # subblocks with zero prior precision (i.e. those not on the tridiagonal) can therefore be ignored
            cross_entropy = 0.5 * np.einsum("tij,tij->", prior_params["J"], posterior.smoothed_covariances)
            Sigmatnt = posterior.expected_states_next_states.transpose((0, 2, 1)) - np.einsum("ti,tj->tji", posterior.expected_states[:-1], posterior.expected_states[1:])
            cross_entropy += np.einsum("tij,tij->", prior_params["L"], Sigmatnt) # no 0.5 weighting because this term is counted twice (once for the lower diagonal and once for the upper diagonal)
            cross_entropy -= prior.log_prob(posterior.expected_states)

            return cross_entropy - posterior_entropy
            
        else:

            return np.mean(posterior.log_prob(samples, u=u) - prior.log_prob(samples))

    def kl_terms(self, posterior, prior_params, u, RPM_batch, batch_id, optimal_prior_params, samples=None):

        prior = self.prior.distribution(prior_params)
        D = self.prior.latent_dims
        T = prior._expected_states.shape[0]
        posterior_entropy = 0.5 * D * T - posterior.log_prob(posterior.expected_states, u=u)

        kl_qp = self.kl_posterior_prior(prior, prior_params, posterior, posterior_entropy, u, samples)

        kl_qf = self.kl_posterior_rpmfactors(optimal_prior_params, prior_params, RPM_batch, batch_id, posterior, posterior_entropy, u, samples)

        return kl_qp, kl_qf

    def log_normaliser(self, J, h, diagonal_boost = 1e-9):

        # https://en.wikipedia.org/wiki/Exponential_family

        # https://math.stackexchange.com/questions/3158303/using-cholesky-decomposition-to-compute-covariance-matrix-determinant
        L = np.linalg.cholesky(J + diagonal_boost * np.eye(J.shape[-1])[None])
        half_log_det_precision = np.log(np.diagonal(L, axis1 = -2, axis2 = -1)).sum(-1)

        return vmap(vmap(lambda h, J, d: 0.5 * h @ psd_solve(J, h) - d))(h, J, half_log_det_precision)