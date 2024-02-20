from jax import numpy as np
from jax import random, jit, vmap, value_and_grad
from utils import kl_qp_natural_parameters, batch_expected_log_F, entropy, initialise_LDS_params, generate_LDS_params, batch_generate_data, R2_true_model, construct_covariance_matrix, update_prior, marginal_u_integrated_out, marginal, policy_loss, truncate_singular_values, R2_inferred_vs_actual_z, scale_y, load_pendulum_control_data, moment_match_RPM, get_marginals_of_joint, log_to_wandb, batch_perform_Kalman_smoothing, closed_form_LDS_updates, dynamics_to_tridiag, get_beta_schedule, batch_perform_Kalman_smoothing_true_params, get_constrained_prior_params, initialise_LDS_params_via_M_step, log_prob_under_posterior, sample_from_MVN, batch_expected_log_f_over_F, batch_get_prior_samples
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

from networks import GRU_RPM, implicit_q, discriminate, estimate_log_density_ratio, sample_from_q, GRUCell_LN, control_network

def get_RPM_factors(params, opt_states, y, options):

    rpm_opt_state, _, _, _, _ = opt_states

    RPM = rpm_opt_state.apply_fn(params["rpm_params"], y)

    return RPM

def get_posterior_samples(params, opt_states, y, u, key):

    _, _, _, _, q_sample_opt = opt_states

    B = u.shape[0]
    keys = random.split(key, B)
    z, y_embed = vmap(q_sample_opt.apply_fn, in_axes=(None,0,0,0))(params["q_sample_params"], y, u, keys)

    return z, y_embed

def get_free_energy(params, LDRE, opt_states, y, u, beta, key, options):

    RPM = get_RPM_factors(params, opt_states, y, options)

    B, T, D, U = y.shape[0], y.shape[1], y.shape[2], u.shape[-1]
    prior_params = get_constrained_prior_params(params['prior_params'], U)

    subkey1, subkey2, key = random.split(key, 3)
    z_q, y_embed = get_posterior_samples(params, opt_states, y, u, subkey1)

    B = u.shape[0]
    keys = random.split(subkey2, B)
    z_p = batch_get_prior_samples(prior_params, u, keys)

    _, _, _, LDRE_opt_state, _ = opt_states

    kl_qp, LDRE_loss, LDRE_opt_state = LDRE(z_p, z_q, y_embed, u, LDRE_opt_state)

    log_f_over_F = batch_expected_log_f_over_F(RPM['mu'], RPM['Sigma'], z_q, np.arange(B))
    T_log_B = T * np.log(B)

    opt_states[3] = LDRE_opt_state

    kl_qp /= (T * D)
    log_f_over_F /= (T * D)
    T_log_B /= (T * D)

    free_energy = - beta * kl_qp + log_f_over_F - T_log_B

    return -free_energy.mean(), (kl_qp, log_f_over_F, z_q, LDRE_loss, opt_states)

get_value_and_grad = value_and_grad(get_free_energy, has_aux=True)

def params_update_model(grads, opt_states):
    
    rpm_opt_state, prior_opt_state, control_state, LDRE_opt, q_sample_opt = opt_states

    rpm_opt_state = rpm_opt_state.apply_gradients(grads = grads["rpm_params"])

    prior_opt_state = prior_opt_state.apply_gradients(grads = grads["prior_params"])
    prior_opt_state.params["A"] = truncate_singular_values(prior_opt_state.params["A"])

    # LDRE_opt = LDRE_opt.apply_gradients(grads = grads["LDRE_params"])

    q_sample_opt = q_sample_opt.apply_gradients(grads = grads["q_sample_params"])

    params = {}
    params["rpm_params"] = rpm_opt_state.params
    params["prior_params"] = prior_opt_state.params
    params["u_emb_params"] = control_state.params
    params["LDRE_params"] = LDRE_opt.params
    params["q_sample_params"] = q_sample_opt.params

    return params, [rpm_opt_state, prior_opt_state, control_state, LDRE_opt, q_sample_opt]

def train_step(params, opt_states, y, u, key, beta, LDRE, options):

    (loss, (kl_qp, log_f_over_F, z_q, LDRE_loss, opt_states)), grads = get_value_and_grad(params, LDRE, opt_states, y, u, beta, key, options)
    params, opt_states = params_update_model(grads, opt_states)

    return params, opt_states, loss, kl_qp, log_f_over_F, z_q, LDRE_loss

def get_train_state(ckpt_metrics_dir, all_models, all_optimisers=[], all_params=[]):

    options = CheckpointManagerOptions(max_to_keep=3, best_fn=lambda metrics: metrics, best_mode='min')
    mngr = CheckpointManager(ckpt_metrics_dir,  
                             {'rpm_model_state': AsyncCheckpointer(PyTreeCheckpointHandler()),
                              'prior_model_state': AsyncCheckpointer(PyTreeCheckpointHandler()),
                              'u_emb_model_state': AsyncCheckpointer(PyTreeCheckpointHandler()),
                              'LDRE_model_state': AsyncCheckpointer(PyTreeCheckpointHandler()),
                              'implicitQ_model_state': AsyncCheckpointer(PyTreeCheckpointHandler())},
                             options)

    states = []
    for params, optimiser, model in zip(all_params, all_optimisers, all_models):

        states.append(train_state.TrainState.create(apply_fn = model.apply, params = params, tx = optimiser))
        
    return states, mngr

# import os
# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
# os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.5'

B = 250
T = 100
D = 2
U = 1
h_dim_rpm = 50
h_dim_LDRE = 50
h_dim_gen = 50
h_dim_u_emb = 50
carry_dim = 50
prior_lr = 1e-2
learning_rate = 1e-2
max_grad_norm = 10
n_epochs = 5000
log_every = 250

options = {}
options['normalise_y'] = True
options['embed_u'] = False
options['f_time_dependent'] = False
options['initialise_via_M_step'] = False
options['num_MC_samples'] = 1
options["beta_init_value"] = 0.
options["beta_end_value"] = 1.
options["beta_transition_begin"] = 1000
options["beta_transition_steps"] = 1000
options['fit_LDS'] = True
options['save_dir'] = "/nfs/nhome/live/jheald/svae/my_code/runs"
options['project_name'] = 'RPM-mycode'

seed = 5
subkey1, subkey2, subkey3, subkey4, subkey5, subkey6, subkey7, subkey8, subkey9, key = random.split(random.PRNGKey(seed), 10)

if options['fit_LDS']:

    true_prior_params = generate_LDS_params(D, U, subkey3)
    keys = random.split(subkey4, B)
    true_z, y, u = batch_generate_data(true_prior_params, T, D, U, keys)

    smoothed_true_params = batch_perform_Kalman_smoothing_true_params(true_prior_params, y, u)

else:

    y, u = load_pendulum_control_data()

if options['f_time_dependent']:

    time = np.arange(T)
    time = (time - time.mean())/time.std() 
    y = np.concatenate((y[:B,:,:], time[None, :, None].repeat(B, axis=0)),axis=2)

if options['normalise_y']:

    y = scale_y(y)

# RPM = rpm_network(z_dim=D, h_dim=h_dim_rpm)
RPM = GRU_RPM(carry_dim=carry_dim, h_dim=h_dim_rpm, z_dim=D, T=T)
params = {}
if options['f_time_dependent']:
    params["rpm_params"] = RPM.init(y = np.ones((D+1,)), rngs = {'params': subkey1})
else:
    params["rpm_params"] = RPM.init(y = np.ones((B,T,D)), rngs = {'params': subkey1})
    # params["rpm_params"] = RPM.init(y = np.ones((D,)), rngs = {'params': subkey1})

if options['initialise_via_M_step']:
    params["prior_params"] = initialise_LDS_params_via_M_step(RPM, params["rpm_params"], y, u, subkey2, options, closed_form_M_Step=False)
else:
    params["prior_params"] = initialise_LDS_params(D, U, subkey2, closed_form_M_Step=False)

u_emb = control_network(u_emb_dim=U, h_dim=h_dim_u_emb)
params["u_emb_params"] = u_emb.init(u = np.ones((U,)), rngs = {'params': subkey5})

discriminator = discriminate(h_dim=h_dim_LDRE)
params["LDRE_params"] = discriminator.init(z=np.ones((B,T,D)), z_prev=np.ones((B,T,D)), u_prev=np.ones((B,T,U)), y_embed=np.ones((B,T,carry_dim)), rngs = {'params': subkey8})

LDRE = estimate_log_density_ratio(h_dim=h_dim_LDRE)

q_sample = implicit_q(carry_dim=carry_dim, z_dim=D, h_dim=h_dim_gen)
params["q_sample_params"] = q_sample.init(y = np.ones((T,D)), u = np.ones((T,U)), key = random.PRNGKey(0), rngs = {'params': subkey9})

rpm_opt = opt.chain(opt.adam(learning_rate=learning_rate), opt.clip_by_global_norm(max_grad_norm))
prior_opt = opt.chain(opt.adam(learning_rate=prior_lr), opt.clip_by_global_norm(max_grad_norm))
u_emb_opt = opt.chain(opt.adam(learning_rate=learning_rate), opt.clip_by_global_norm(max_grad_norm))
LDRE_opt = opt.chain(opt.adam(learning_rate=learning_rate), opt.clip_by_global_norm(max_grad_norm))
q_sample_opt = opt.chain(opt.adam(learning_rate=learning_rate), opt.clip_by_global_norm(max_grad_norm))

all_optimisers = (rpm_opt, prior_opt, u_emb_opt, LDRE_opt, q_sample_opt)
all_params = (params["rpm_params"], params["prior_params"], params["u_emb_params"], params["LDRE_params"], params["q_sample_params"])
all_models = (RPM, RPM, u_emb, discriminator, q_sample)
opt_states, mngr = get_train_state(options['save_dir'], all_models, all_optimisers, all_params)

beta_schedule = get_beta_schedule(options)

train_step_jit = jit(partial(train_step, LDRE=LDRE, options=options))

print("pass params around via opt_states not separately")
print("not sure how to deal with u_embed in EM (not embedding at the moment)")

pbar = trange(n_epochs)
for itr in pbar:

    subkey, key = random.split(key, 2)

    beta = beta_schedule(itr)

    if options['fit_LDS']:

        params, opt_states, loss, kl_qp, log_f_over_F, z_q, LDRE_loss = train_step_jit(params, opt_states, y, u, subkey, beta)
        R2, predicted_z = R2_inferred_vs_actual_z(z_q, smoothed_true_params['smoothed_means'])

    else:

        params, opt_states, loss, log_f_over_F, z_q, LDRE_loss = train_step_jit(params, opt_states, y[:B,:,:], u[:B,:,None], subkey, beta)
        R2 = 0.

    pbar.set_description("train loss: {:.3f},  kl_qp: {:.3f}, log_f_over_F: {:.3f}, LDRE_loss: {:.3f}, R2 train states: {:.3f}".format(loss, kl_qp.mean(), log_f_over_F.mean(), LDRE_loss.mean(), R2))

    if itr % log_every == 0:

        log_to_wandb(loss, kl_qp, log_f_over_F, LDRE_loss, predicted_z, smoothed_true_params['smoothed_means'], options)

breakpoint()