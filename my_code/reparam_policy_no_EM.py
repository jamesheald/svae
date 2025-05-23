from jax import numpy as np
from jax import random, jit, vmap, value_and_grad
from utils import kl_qp_natural_parameters, batch_expected_log_F, entropy, initialise_LDS_params, generate_LDS_params, batch_generate_data, R2_true_model, construct_covariance_matrix, update_prior, marginal_u_integrated_out, marginal, policy_loss, truncate_singular_values, R2_inferred_vs_actual_z, scale_y, load_pendulum_control_data, moment_match_RPM, get_marginals_of_joint, log_to_wandb, batch_perform_Kalman_smoothing, closed_form_LDS_updates, dynamics_to_tridiag, get_beta_schedule, batch_perform_Kalman_smoothing_true_params, get_constrained_prior_params, initialise_LDS_params_via_M_step, log_prob_under_posterior, create_tf_dataset, batch_get_prior_marginal_means
from jax.lax import scan, stop_gradient
from flax import linen as nn

from flax.training.orbax_utils import restore_args_from_target
from flax.training import train_state
from orbax.checkpoint import AsyncCheckpointer, Checkpointer, PyTreeCheckpointHandler, CheckpointManager, CheckpointManagerOptions
import optax as opt

from dynamax.utils.utils import psd_solve
from jax.scipy.linalg import block_diag

from tqdm import trange

from functools import partial

import tensorflow_probability.substrates.jax.distributions as tfd
MVN = tfd.MultivariateNormalFullCovariance

import tensorflow_datasets as tfds

from networks import control_network, GRU_RPM, delta_q_params, emission_potential

def compute_free_energy(prior_params, prior_JL, J_RPM, mu_RPM, Sigma_RPM, smoothed, emission_potentials, u, key, batch_id, options):

    B, T, D = mu_RPM.shape[0], mu_RPM.shape[1], mu_RPM.shape[2]

    posterior_entropy = 0.5 * D * T - log_prob_under_posterior(prior_params, emission_potentials, smoothed, u)
    cross_entropy = 0.5 * np.einsum("tij,tij->", prior_JL["J"], smoothed['smoothed_covariances'])
    Sigmatnt = smoothed['smoothed_cross_covariances'] - np.einsum("ti,tj->tij", smoothed['smoothed_means'][:-1], smoothed['smoothed_means'][1:])
    cross_entropy += np.einsum("tij,tij->", prior_JL["L"], Sigmatnt) # no 0.5 weighting because this term is counted twice (once for the lower diagonal and once for the upper diagonal)
    # cross_entropy -= log_prob_under_prior(prior_params, smoothed['smoothed_means'], u) # commented out along with call in log_prob_under_posterior, as they cancel
    kl_qp = cross_entropy - posterior_entropy

    ce_qf = 0.5 * np.einsum("tij,tij->", J_RPM, smoothed['smoothed_covariances'])
    ce_qf -= MVN(loc=mu_RPM[batch_id], covariance_matrix=Sigma_RPM[batch_id]).log_prob(smoothed['smoothed_means']).sum()
    
    keys = random.split(key, options['num_MC_samples'])
    ce_qF = - batch_expected_log_F(smoothed['smoothed_means'], smoothed['smoothed_covariances'], mu_RPM, Sigma_RPM, keys).mean()

    T_log_B = T * np.log(B)

    kl_qp /= (T * D)
    ce_qf /= (T * D)
    ce_qF /= (T * D)
    T_log_B /= (T * D)

    return kl_qp, ce_qf, ce_qF, T_log_B

def get_RPM_factors(params, opt_states, y, options):

    rpm_opt_state, _, _, _ = opt_states

    # RPM_constant = rpm_opt_state.apply_fn(params["rpm_params"], y)

    # T = y.shape[1]
    # if options['use_LDS_for_F_in_q']:

    #     RPM_time_varying = marginal(params['prior_params'], T) # treat parameters of p(z'|z) as free (implicit) parameters of the q distribution

    # elif options['use_GRU_for_F_in_q']:

    #     RPM_time_varying = F_approx_opt_state.apply_fn(params["F_approx_params"])

    # RPM = {}
    # RPM['J'] = RPM_time_varying['J'][None] + RPM_constant["J"]
    # RPM['h'] = RPM_time_varying['h'][None] + RPM_constant["h"]
    # RPM['Sigma'] = vmap(vmap(lambda S: psd_solve(S, np.eye(S.shape[-1]))))(RPM['J'])
    # RPM['mu'] = np.einsum("hijk,hik->hij", RPM['Sigma'], RPM['h'])

    RPM = rpm_opt_state.apply_fn(params["rpm_params"], y)

    return RPM, RPM

def get_posterior(params, prior_params, opt_states, y, u, RPM_constant):

    _, delta_q_opt, _, _ = opt_states

    # delta_q_potentials = vmap(delta_q_opt.apply_fn, in_axes=(None,0))(params["delta_q_params"], y)

    # emission_potentials = {}
    # emission_potentials['Sigma'] = vmap(vmap(lambda J1, J2: psd_solve(J1 + J2, np.eye(J1.shape[-1]))))(RPM_constant['J'], delta_q_potentials['J'])
    # emission_potentials['mu'] = vmap(vmap(lambda h1, h2, S: S @ (h1 + h2)))(RPM_constant['h'], delta_q_potentials['h'], emission_potentials['Sigma'])

    # emission_potentials = vmap(delta_q_opt.apply_fn, in_axes=(None,0))(params["delta_q_params"], y)
    emission_potentials = delta_q_opt.apply_fn(params["delta_q_params"], y)

    smoothed = batch_perform_Kalman_smoothing(prior_params, emission_potentials, u)

    return smoothed, emission_potentials

def get_free_energy(params, opt_states, y, u, key, beta, options):

    RPM_constant, RPM = get_RPM_factors(params, opt_states, y, options)

    if options['embed_u']:

        _, _, _, control_state = opt_states

        u = control_state.apply_fn(params["u_emb_params"], u)

    T, U = y.shape[1], u.shape[-1]
    prior_params = get_constrained_prior_params(params['prior_params'], U)

    prior_JL = dynamics_to_tridiag(prior_params, T)

    smoothed, emission_potentials = get_posterior(params, prior_params, opt_states, y, u, RPM_constant)

    B = y.shape[0]
    keys = random.split(key, B)
    kl_qp, ce_qf, ce_qF, T_log_B = vmap(compute_free_energy, in_axes=(None,None,0,None,None,0,0,0,0,0,None))(prior_params, prior_JL, RPM['J'], RPM['mu'], RPM['Sigma'], smoothed, emission_potentials, u, keys, np.arange(B), options)
    free_energy = - beta * kl_qp - ce_qf + ce_qF - T_log_B

    return -free_energy.mean(), (kl_qp, ce_qf, ce_qF, smoothed)

get_value_and_grad = value_and_grad(get_free_energy, has_aux=True)

def params_update(grads, opt_states):
    
    rpm_opt_state, delta_q_opt, prior_opt_state, control_state = opt_states

    rpm_opt_state = rpm_opt_state.apply_gradients(grads = grads["rpm_params"])

    delta_q_opt = delta_q_opt.apply_gradients(grads = grads["delta_q_params"])

    prior_opt_state = prior_opt_state.apply_gradients(grads = grads["prior_params"])
    prior_opt_state.params["A"] = truncate_singular_values(prior_opt_state.params["A"])

    control_state = control_state.apply_gradients(grads = grads["u_emb_params"])

    params = {}
    params["rpm_params"] = rpm_opt_state.params
    params["delta_q_params"] = delta_q_opt.params
    params["prior_params"] = prior_opt_state.params
    params["u_emb_params"] = control_state.params

    return params, [rpm_opt_state, delta_q_opt, prior_opt_state, control_state]

def train_step(params, opt_states, y, u, key, beta, options):

    (loss, (kl_qp, ce_qf, ce_qF, smoothed)), grads = get_value_and_grad(params, opt_states, y, u, key, beta, options)
    params, opt_states = params_update(grads, opt_states)

    return params, opt_states, loss, kl_qp, ce_qf, ce_qF, smoothed['smoothed_means']

def get_train_state(ckpt_metrics_dir, all_models, all_optimisers=[], all_params=[]):

    options = CheckpointManagerOptions(max_to_keep=3, best_fn=lambda metrics: metrics, best_mode='min')
    mngr = CheckpointManager(ckpt_metrics_dir,  
                             {'rpm_model_state': AsyncCheckpointer(PyTreeCheckpointHandler()),
                              'delta_q_state': AsyncCheckpointer(PyTreeCheckpointHandler()),
                              'prior_model_state': AsyncCheckpointer(PyTreeCheckpointHandler()),
                              'u_emb_model_state': AsyncCheckpointer(PyTreeCheckpointHandler())},
                             options)

    states = []
    for params, optimiser, model in zip(all_params, all_optimisers, all_models):

        states.append(train_state.TrainState.create(apply_fn = model.apply, params = params, tx = optimiser))
        
    return states, mngr

B = 1000
T = 100
D = 2
U = 1
h_dims_q = [50, 50, 50]
h_dims_rpm = [50, 50, 50]
h_dims_u_emb = [50, 50, 50]
carry_dim_GRU_RPM = 50
prior_lr = 1e-3
learning_rate = 1e-3
max_grad_norm = 10
n_epochs = 1000
log_every = 250

options = {}
options['normalise_y'] = True
options['diagonal_covariance_RPM'] = True # setting this to True speeds things up a lot
options['diagonal_covariance_q_potentials'] = True
options['layer_norm_RPM'] = True
options['layer_norm_q_potentials'] = True
options['layer_norm_control'] = True
options['embed_u'] = False
options['f_time_dependent'] = False
options['initialise_via_M_step'] = False
options['num_MC_samples'] = 1
options["beta_init_value"] = 1.
options["beta_end_value"] = 1.
options["beta_transition_begin"] = 1000
options["beta_transition_steps"] = 1000
options['fraction_for_validation'] = 0.
options['tfds_shuffle_data'] = True
options['tfds_seed'] = 0
options['batch_size_train'] = 250
options['batch_size_validate'] = 50
options['fit_LDS'] = True
options['save_dir'] = "/nfs/nhome/live/jheald/svae/my_code/runs"
options['project_name'] = 'RPM-mycode'

seed = 0
subkey1, subkey2, subkey3, subkey4, subkey5, subkey6, subkey7, subkey8, key = random.split(random.PRNGKey(seed), 9)

if options['fit_LDS']:

    true_prior_params = generate_LDS_params(D, U, subkey3)
    keys = random.split(subkey4, B)
    true_z, y, u = batch_generate_data(true_prior_params, T, D, U, keys)

    batch_get_prior_marginal_means_jit = jit(batch_get_prior_marginal_means)
    gt_mu_no_u = batch_get_prior_marginal_means_jit(true_prior_params, u[:3]*0.)
    gt_mu_u = batch_get_prior_marginal_means_jit(true_prior_params, u[:3])
    
    gt_smoothed = batch_perform_Kalman_smoothing_true_params(true_prior_params, y, u)

else:

    y, u = load_pendulum_control_data()

if options['f_time_dependent']:

    time = np.arange(T)
    time = (time - time.mean()) / time.std() 
    y = np.concatenate((y[:B,:,:], time[None, :, None].repeat(B, axis=0)),axis=2)

if options['normalise_y']:

    y = scale_y(y)

train_dataset, validate_dataset = create_tf_dataset(y, u, gt_smoothed['smoothed_means'], options)

# RPM = rpm_network(z_dim=D, h_dim=h_dim_rpm)
RPM = GRU_RPM(carry_dim=carry_dim_GRU_RPM, h_dims=h_dims_rpm, z_dim=D, T=T, diagonal_covariance=options['diagonal_covariance_RPM'], layer_norm=options['layer_norm_RPM'])
# delta_q = delta_q_params(carry_dim=carry_dim, h_dims=[], z_dim=D, diagonal_covariance=options['diagonal_covariance_q_potentials'])
delta_q = emission_potential(z_dim=D, h_dims=h_dims_q, diagonal_covariance=options['diagonal_covariance_q_potentials'], layer_norm=options['layer_norm_q_potentials'])
# delta_q = rpm_network(h_dim=h_dim_rpm, z_dim=D)
params = {}
if options['f_time_dependent']:
    params["rpm_params"] = RPM.init(x = np.ones((D+1,)), rngs = {'params': subkey1})
    params["delta_q_params"] = delta_q.init(y = np.ones((T,D+1)), rngs = {'params': subkey7})
else:
    params["rpm_params"] = RPM.init(x = np.ones((B,T,D)), rngs = {'params': subkey1})
    # params["rpm_params"] = RPM.init(y = np.ones((D,)), rngs = {'params': subkey1})
    # params["delta_q_params"] = delta_q.init(y = np.ones((B,T,D,)), rngs = {'params': subkey7})
    params["delta_q_params"] = delta_q.init(x = np.ones((B,T,D,)), rngs = {'params': subkey7})

if options['initialise_via_M_step']:
    params["prior_params"] = initialise_LDS_params_via_M_step(RPM, params["rpm_params"], y, u, subkey2, options, closed_form_M_Step=False)
else:
    params["prior_params"] = initialise_LDS_params(D, U, subkey2, closed_form_M_Step=False)

u_emb = control_network(u_emb_dim=U, h_dims=h_dims_u_emb, layer_norm=options['layer_norm_control'])
params["u_emb_params"] = u_emb.init(u = np.ones((U,)), rngs = {'params': subkey5})

rpm_opt = opt.chain(opt.adam(learning_rate=learning_rate), opt.clip_by_global_norm(max_grad_norm))
delta_q_opt = opt.chain(opt.adam(learning_rate=learning_rate), opt.clip_by_global_norm(max_grad_norm))
prior_opt = opt.chain(opt.adam(learning_rate=prior_lr), opt.clip_by_global_norm(max_grad_norm))
u_emb_opt = opt.chain(opt.adam(learning_rate=learning_rate), opt.clip_by_global_norm(max_grad_norm))

all_optimisers = (rpm_opt, delta_q_opt, prior_opt, u_emb_opt)
all_params = (params["rpm_params"], params["delta_q_params"], params["prior_params"], params["u_emb_params"])
all_models = (RPM, delta_q, RPM, u_emb)
opt_states, mngr = get_train_state(options['save_dir'], all_models, all_optimisers, all_params)

beta_schedule = get_beta_schedule(options)

train_step_jit = jit(partial(train_step, options=options))

get_RPM_factors_jit = jit(partial(get_RPM_factors, options=options))










train_datagen = iter(tfds.as_numpy(train_dataset))

y_batch, u_batch, gt_mu_batch = next(train_datagen)

beta = beta_schedule(0)

loss, (kl_qp, ce_qf, ce_qF, mu) = get_free_energy(params, opt_states, y_batch, u_batch, key, beta, options)

mu_no_u = batch_get_prior_marginal_means_jit(params['prior_params'], u[:3]*0.)
mu_u = batch_get_prior_marginal_means_jit(params['prior_params'], u[:3])

RPM, _ = get_RPM_factors_jit(params, opt_states, y_batch[:3])

R2, projected_mu = R2_inferred_vs_actual_z(mu['smoothed_means'], gt_mu_batch)

log_to_wandb(loss, kl_qp, ce_qf, ce_qF, y_batch[:3], mu_no_u, gt_mu_no_u, mu_u, gt_mu_u, projected_mu, RPM['mu'], gt_mu_batch, options)








print("log mean metrics over all batches, not just last batch of epoch")
print("pass params around via opt_states not separately")




n_batches_train = len(train_dataset)
pbar = trange(n_epochs)
for itr in pbar:

    subkey, key = random.split(key, 2)

    beta = beta_schedule(itr)

    # convert the tf.data.Dataset train_dataset into an iterable that is shuffled differently each epoch
    train_datagen = iter(tfds.as_numpy(train_dataset))

    for batch in range(1, n_batches_train + 1):

        y_batch, u_batch, gt_mu_batch = next(train_datagen)

        if options['fit_LDS']:

            params, opt_states, loss, kl_qp, ce_qf, ce_qF, mu = train_step_jit(params, opt_states, y_batch, u_batch, subkey, beta)

        else:

            params, opt_states, loss, kl_qp, ce_qf, ce_qF, mu = train_step_jit(params, opt_states, y[:B,:,:], u[:B,:,None], subkey, beta)

        pbar.set_description("train loss: {:.3f},  kl_qp: {:.3f}, ce_qf: {:.3f}, ce_qF: {:.3f}, R2 train states: {:.3f}".format(loss, kl_qp.mean(), ce_qf.mean(), ce_qF.mean(), R2))

    if (itr + 1) % log_every == 0:

        mu_no_u = batch_get_prior_marginal_means_jit(params['prior_params'], u[:3]*0.)
        mu_u = batch_get_prior_marginal_means_jit(params['prior_params'], u[:3])

        RPM, _ = get_RPM_factors_jit(params, opt_states, y_batch[:3])

        R2, projected_mu = R2_inferred_vs_actual_z(mu, gt_mu_batch)

        log_to_wandb(loss, kl_qp, ce_qf, ce_qF, y_batch, mu_no_u, gt_mu_no_u, mu_u, gt_mu_u, projected_mu, RPM['mu'], gt_mu_batch, options)

breakpoint()