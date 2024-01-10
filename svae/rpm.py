from jax import vmap
from jax.tree_util import tree_map
import jax.numpy as np
import jax.random as jr

import numpy as onp

# from jax.scipy.linalg import block_diag
# from jax.numpy.linalg import solve
from dynamax.utils.utils import psd_solve
import tensorflow_probability.substrates.jax.distributions as tfd
MVN = tfd.MultivariateNormalFullCovariance

from jax.scipy.special import logsumexp

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

        # # Mask out a large window of states
        # mask_size = params.get("mask_size")
        # T = data.shape[0]
        # D = self.prior.latent_dims
        # # mask = onp.ones((T,))
        # mask = np.ones((T,))
        # key, dropout_key = jr.split(key)
        # if mask_size:
        #     # Potential dropout...!
        #     # Use a trick to generate the mask without indexing with a tracer
        #     # mask contiguous timepoints
        #     # start_id = jr.choice(dropout_key, T - mask_size + 1)
        #     # mask = np.array(np.arange(T) >= start_id) \
        #          # * np.array(np.arange(T) < start_id + mask_size)
        #     # mask = 1 - mask
        #     # mask potentially non-contiguous timepoints
        #     mask_idx = jr.choice(dropout_key, T, (mask_size,), replace = False)
        #     mask = mask.at[mask_idx].set(0)
        #     if params.get("mask_type") == "potential":
        #         # This only works with svaes
        #         potential = self.recognition.apply(rec_params, data)
        #         # Uninformative potential
        #         infinity = 1e5
        #         uninf_potential = {"mu": np.zeros((T, D)), 
        #                            "Sigma": np.tile(np.eye(D) * infinity, (T, 1, 1))}
        #         # Replace masked parts with uninformative potentials
        #         potential = tree_map(
        #             lambda t1, t2: np.einsum("i,i...->i...", mask[:t1.shape[0]], t1) 
        #                          + np.einsum("i,i...->i...", 1-mask[:t2.shape[0]], t2), 
        #             potential, 
        #             uninf_potential)
        #     else:
        #         potential = self.recognition.apply(rec_params, 
        #                                            np.einsum("t...,t->t...", data, mask))
        # else:
        #     # Don't do any masking
        
        # potential = self.recognition.apply(rec_params, data) # these potentials represent rpm_factor / prior

        # rpm_factor = self.recognition.apply(rec_params, data)
        # rpm_factor_precision = solve(rpm_factor['Sigma'], np.eye(self.prior.latent_dims)[None])
        # prior_precision = solve(prior_params['Sigma'], np.eye(self.prior.latent_dims)[None])
        # potential = {}
        # potential['Sigma'] = solve(rpm_factor_precision - prior_precision, np.eye(self.prior.latent_dims)[None])
        # potential['mu'] = potential['Sigma'] @ (rpm_factor_precision @ rpm_factor['mu'] - prior_precision @ prior_params['Ex'])

        # def get_potentials(rpm_factor, prior_params):

        #         rpm_factor_precision = np.linalg.inv(rpm_factor['Sigma'])
        #         prior_precision = np.linalg.inv(prior_params['Sigma'])
        #         potential_Sigma = np.linalg.inv(rpm_factor_precision - prior_precision)
        #         potential_mu = potential_Sigma @ (rpm_factor_precision @ rpm_factor['mu'] - prior_precision @ prior_params['Ex'])

        #     return 

        potential = {}
        # potential['J'] = RPM_batch["J"][batch_id] - MM_prior["J"]
        # potential['h'] = RPM_batch["h"][batch_id] - MM_prior["h"]
        # potential['Sigma'] = vmap(lambda J, I: psd_solve(J + np.eye(3) * (abs(np.linalg.eigvals(J).min()) + 1e-4), I), in_axes=(0, None))(potential["J"], np.eye(self.prior.latent_dims))
        # potential['mu'] = np.einsum("ijk,ik->ij", potential['Sigma'], potential['h'])
        potential['Sigma'] = RPM_batch["Sigma"][batch_id]
        potential['mu'] = RPM_batch["mu"][batch_id]

        # Update: it makes more sense that inference is done in the posterior object
        posterior_params = self.posterior.infer(prior_params, potential, u)

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
        auxillary_J = optimal_prior_params["J"] - posterior_J
        auxillary_h = optimal_prior_params["h"] - posterior_h
        E_log_aux = np.einsum("ij,ij->i", auxillary_h, post_Ex) - 0.5 * (np.einsum("ij,ij->i", post_Ex, np.einsum("ijk,ik->ij", auxillary_J, post_Ex)) + (auxillary_J * post_Sigma).sum(axis = (1, 2)))

        # use the current plus one random data point
        batch_size = RPM_batch["J"].shape[0]
        timepoints = RPM_batch["J"].shape[1]
        probs = np.ones(batch_size) / (batch_size - 1)
        probs = probs.at[batch_id].set(0)
        sample_datapoint = jr.choice(key, np.concatenate([np.arange(batch_size)]), p = probs, shape=(timepoints,))

        RPM_J = np.concatenate((RPM_batch["J"][batch_id][None], RPM_batch["J"][sample_datapoint, np.arange(timepoints)][None]))
        RPM_h = np.concatenate((RPM_batch["h"][batch_id][None], RPM_batch["h"][sample_datapoint, np.arange(timepoints)][None]))

        # normalised_auxillary_J = posterior_J[None] + RPM_J - MM_prior["J"][None]
        # normalised_auxillary_h = posterior_h[None] + RPM_h - MM_prior["h"][None]
        normalised_auxillary_J = posterior_J[None] + RPM_J
        normalised_auxillary_h = posterior_h[None] + RPM_h

        normalised_auxillary_log_normaliser = self.log_normaliser(normalised_auxillary_J, normalised_auxillary_h)
        # RPM_log_normaliser = self.log_normaliser(RPM_J, RPM_h)
        RPM_log_normaliser = self.log_normaliser(optimal_prior_params["J"] + RPM_J, optimal_prior_params["h"] + RPM_h)

        log_Gamma = logsumexp(normalised_auxillary_log_normaliser - RPM_log_normaliser, axis=0, b=np.array([1, batch_size - 1])[:, None].repeat(timepoints, axis = 1)/batch_size)

        # use all datapoints
        # normalised_auxillary_J = posterior_J[None] + RPM_batch["J"] - MM_prior["J"][None]
        # normalised_auxillary_h = posterior_h[None] + RPM_batch["h"] - MM_prior["h"][None]

        # normalised_auxillary_log_normaliser = self.log_normaliser(normalised_auxillary_J, normalised_auxillary_h)
        # RPM_log_normaliser = self.log_normaliser(RPM_batch["J"], RPM_batch["h"])

        # log_Gamma = logsumexp(normalised_auxillary_log_normaliser - RPM_log_normaliser, axis=0, b=1/normalised_auxillary_log_normaliser.shape[0])

        free_energy = - kl_qp - kl_qf - E_log_aux.sum() - log_Gamma.sum()
        free_energy /= target.size

        # kl /= target.size
        # ell /= target.size
        # elbo = ell - kl

        return {
            "free_energy": free_energy,
            # "elbo": elbo,
            "ell": 0.,
            "kl": 0.,
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
    def kl_posterior_rpmfactors(self, optimal_prior_params, RPM_batch, batch_id, posterior, posterior_entropy):

        rpm_J = optimal_prior_params["J"] + RPM_batch["J"][batch_id]
        rpm_h = optimal_prior_params["h"] + RPM_batch["h"][batch_id]
        rpm_Sigma = psd_solve(rpm_J, np.eye(self.prior.latent_dims)[None])
        rpm_mu = rpm_Sigma @ rpm_h

        cross_entropy = 0.5 * np.einsum("tij,tij->", rpm_J, posterior.smoothed_covariances)
        cross_entropy -= MVN(loc=rpm_mu, covariance_matrix=rpm_Sigma).log_prob(posterior.expected_states).sum()

        # cross_entropy = 0.5 * np.einsum("tij,tij->", RPM_batch["J"][batch_id], posterior.smoothed_covariances)
        # cross_entropy -= MVN(loc=RPM_batch["mu"][batch_id], covariance_matrix=RPM_batch["Sigma"][batch_id]).log_prob(posterior.expected_states).sum()

        return cross_entropy - posterior_entropy

    def kl_posterior_prior(self, prior, prior_params, posterior, posterior_entropy, samples=None):

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

            return np.mean(posterior.log_prob(samples) - prior.log_prob(samples))

    def kl_terms(self, posterior, prior_params, u, RPM_batch, batch_id, optimal_prior_params, samples=None):

        prior = self.prior.distribution(prior_params)
        D = self.prior.latent_dims
        T = prior._expected_states.shape[0]
        posterior_entropy = 0.5 * D * T - posterior.log_prob(posterior.expected_states, u=u)

        kl_qp = self.kl_posterior_prior(prior, prior_params, posterior, posterior_entropy, samples)

        kl_qf = self.kl_posterior_rpmfactors(optimal_prior_params, RPM_batch, batch_id, posterior, posterior_entropy)

        return kl_qp, kl_qf

    def log_normaliser(self, J, h, diagonal_boost = 1e-9):

        # https://en.wikipedia.org/wiki/Exponential_family

        # https://math.stackexchange.com/questions/3158303/using-cholesky-decomposition-to-compute-covariance-matrix-determinant
        L = np.linalg.cholesky(J + diagonal_boost * np.eye(J.shape[-1])[None])
        half_log_det_covariance = np.log(np.diagonal(L, axis1 = -2, axis2 = -1)).sum(-1)

        return vmap(vmap(lambda h, J, y: 0.5 * h @ psd_solve(J, h) - y))(h, J, half_log_det_covariance)