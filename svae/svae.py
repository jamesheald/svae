from jax import vmap
from jax.tree_util import tree_map
import jax.numpy as np
import jax.random as jr

import numpy as onp

# from jax.scipy.linalg import block_diag
# from jax.numpy.linalg import solve
# import tensorflow_probability.substrates.jax.distributions as tfd
# MVN = tfd.MultivariateNormalFullCovariance

class SVAE:
    def __init__(self,
                 recognition=None, decoder=None, prior=None, posterior=None,
                 input_dummy=None, latent_dummy=None, u_dummy=None):
        """
        rec_net, dec_net, prior are all objects that take in parameters
        rec_net.apply(params, data) returns Gaussian potentials (parameters)
        dec_net.apply(params, latents) returns probability distributions
        prior : SVAEPrior
        """
        self.recognition = recognition
        self.decoder = decoder
        self.prior = prior
        self.posterior = posterior
        self.input_dummy = input_dummy
        self.latent_dummy = latent_dummy
        self.u_dummy = u_dummy

    def init(self, key=None):
        if key is None:
            key = jr.PRNGKey(0)

        rec_key, dec_key, prior_key, post_key = jr.split(key, 4)

        return {
            "rec_params": self.recognition.init(rec_key, self.input_dummy),
            "dec_params": self.decoder.init(dec_key, self.latent_dummy),
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
             u, model_params, sample_kl=False, **params):
        rec_params = model_params["rec_params"]
        dec_params = model_params["dec_params"]
        prior_params = self.prior.get_constrained_params(model_params["prior_params"], u)

        # Mask out a large window of states
        mask_size = params.get("mask_size")
        T = data.shape[0]
        D = self.prior.latent_dims
        # mask = onp.ones((T,))
        mask = np.ones((T,))
        key, dropout_key = jr.split(key)
        if mask_size:
            # Potential dropout...!
            # Use a trick to generate the mask without indexing with a tracer
            # mask contiguous timepoints
            # start_id = jr.choice(dropout_key, T - mask_size + 1)
            # mask = np.array(np.arange(T) >= start_id) \
                 # * np.array(np.arange(T) < start_id + mask_size)
            # mask = 1 - mask
            # mask potentially non-contiguous timepoints
            mask_idx = jr.choice(dropout_key, T, (mask_size,), replace = False)
            mask = mask.at[mask_idx].set(0)
            if params.get("mask_type") == "potential":
                # This only works with svaes
                potential = self.recognition.apply(rec_params, data)
                # Uninformative potential
                infinity = 1e5
                uninf_potential = {"mu": np.zeros((T, D)), 
                                   "Sigma": np.tile(np.eye(D) * infinity, (T, 1, 1))}
                # Replace masked parts with uninformative potentials
                potential = tree_map(
                    lambda t1, t2: np.einsum("i,i...->i...", mask[:t1.shape[0]], t1) 
                                 + np.einsum("i,i...->i...", 1-mask[:t2.shape[0]], t2), 
                    potential, 
                    uninf_potential)
            else:
                potential = self.recognition.apply(rec_params, 
                                                   np.einsum("t...,t->t...", data, mask))
        else:
            # Don't do any masking
            potential = self.recognition.apply(rec_params, data)

        # Update: it makes more sense that inference is done in the posterior object
        posterior_params = self.posterior.infer(prior_params, potential, u)
        
        # Take samples under the posterior
        num_samples = params.get("obj_samples") or 1
        samples = self.posterior.sample(posterior_params, u, (num_samples,), key)
        # and compute average ll
        def likelihood_outputs(dec_params, latent, target):
            likelihood_dist = self.decoder.apply(dec_params, latent)
            return likelihood_dist.mean(), likelihood_dist.log_prob(target), likelihood_dist.covariance()

        mean, ells, covv = vmap(likelihood_outputs, in_axes = (None, 0, None))(dec_params, samples, target)

        # Take average over samples then sum the rest
        ell = np.sum(np.mean(ells, axis=0))
        # Compute kl from posterior to prior
        if sample_kl:
            kl = self.kl_posterior_prior(posterior_params, prior_params, u,
                                         samples=samples)
        else:
            kl = self.kl_posterior_prior(posterior_params, prior_params, u)

        kl /= target.size
        ell /= target.size
        elbo = ell - kl

        return {
            "elbo": elbo,
            "ell": ell,
            "kl": kl,
            "posterior_params": posterior_params,
            "posterior_samples": samples,
            "reconstruction": mean,
            # "mask": mask,
            "mean": mean,
            "cov": covv,
            "ells": ells,
            "sample_kl": sample_kl,
        }

    def compute_objective(self, key, data, target, u, model_params, **params):
        results = self.elbo(key, data, target, u, model_params, **params)
        results["objective"] = results["elbo"]
        return results

class DeepLDS(SVAE):
    def kl_posterior_prior(self, posterior_params, prior_params, u,
                           samples=None):
        posterior = self.posterior.distribution(posterior_params)
        prior = self.prior.distribution(prior_params)
        if samples is None:

            # # construct prior precision matrix (one big D x T squared matrix covering all time points)
            # D = self.prior.latent_dims
            # T = prior._expected_states.shape[0]
            # prior_precision_upper_diagonal = block_diag(np.zeros((0, D)), *prior_params["L"].transpose((0, 2, 1)), np.zeros((D, 0)))
            # prior_precision_main_diagonal = block_diag(*prior_params["J"])
            # prior_precision_lower_diagonal = block_diag(np.zeros((D, 0)), *prior_params["L"], np.zeros((0, D)))
            # prior_precision = prior_precision_upper_diagonal + prior_precision_main_diagonal + prior_precision_lower_diagonal

            # # construct posterior precision matrix (one big D x T squared matrix covering all time points)
            # posterior_precision = prior_precision + block_diag(*solve(posterior._emissions_covariances, np.eye(D)[None]))
            # mean_difference = (posterior.smoothed_means - prior._expected_states).reshape(D * T)

            # # construct prior mean (one big D x T vector covering all time points)
            # prior_mean = prior._expected_states.reshape(D * T)

            # # construct posterior mean (one big D x T vector covering all time points)
            # posterior_mean = posterior.smoothed_means.reshape(D * T)

            # cross_entropy = 0.5 * np.trace(solve(posterior_precision, prior_precision).T) - prior.log_prob(posterior.smoothed_means) 
            # cross_entropy = 0.5 * np.trace(solve(posterior_precision, prior_precision).T) - MVN(loc=prior_mean, covariance_matrix=solve(prior_precision, np.eye(D * T))).log_prob(posterior_mean)
            # posterior_entropy = 0.5 * (D * T + D * T * np.log(2 * np.pi) - np.log(np.linalg.det(posterior_precision)))
            # posterior_entropy = 0.5 * D * T - MVN(loc=posterior_mean, covariance_matrix=solve(posterior_precision, np.eye(D * T))).log_prob(posterior_mean)
            # mykl1 = cross_entropy - posterior_entropy

            # det_term = np.log(np.linalg.det(posterior_precision)) - np.log(np.linalg.det(prior_precision))
            # mean_diff_term = mean_difference @ prior_precision @ mean_difference
            # tr_term = np.trace(solve(posterior_precision, prior_precision).T)
            # mykl2 = 0.5 * (det_term + mean_diff_term + tr_term - D * T)

            # tr(prior_precision @ posterior_covariance)
            # the trace of a matrix product is the sum of the elements of the Hadamard product
            # subblocks with zero prior precision (i.e. those not on the tridiagonal) can therefore be ignored
            cross_entropy = 0.5 * np.einsum("tij,tij->", prior_params["J"], posterior.smoothed_covariances)
            Sigmatnt = posterior.expected_states_next_states.transpose((0, 2, 1)) - np.einsum("ti,tj->tji", posterior.expected_states[:-1], posterior.expected_states[1:])
            cross_entropy += np.einsum("tij,tij->", prior_params["L"], Sigmatnt) # no 0.5 weighting because this term is counted twice (once for the lower diagonal and once for the upper diagonal)
            cross_entropy -= prior.log_prob(posterior.expected_states)

            D = self.prior.latent_dims
            T = prior._expected_states.shape[0]
            posterior_entropy = 0.5 * D * T - posterior.log_prob(posterior.expected_states, u=u)

            return cross_entropy - posterior_entropy
            
        else:
            return np.mean(posterior.log_prob(samples) - prior.log_prob(samples))