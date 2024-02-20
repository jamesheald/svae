from jax import numpy as np
from jax import random, jit, vmap, value_and_grad
from utils import kl_qp_natural_parameters, batch_expected_log_F, entropy, initialise_LDS_params, generate_LDS_params, batch_generate_data, R2_true_model, construct_covariance_matrix, update_prior, marginal_u_integrated_out, marginal, policy_loss, truncate_singular_values, R2_inferred_vs_actual_z, scale_y, load_pendulum_control_data, moment_match_RPM, get_marginals_of_joint, log_to_wandb, batch_perform_Kalman_smoothing, closed_form_LDS_updates, dynamics_to_tridiag, get_beta_schedule, batch_perform_Kalman_smoothing_true_params, initialise_LDS_params_via_M_step, construct_precision_matrix, log_prob_under_prior, batch_half_log_det
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

from networks import control_network, GRU_RPM, delta_q_params, rpm_network

def compute_free_energy_E_step(prior_params, prior_JL, J_RPM, mu_RPM, Sigma_RPM, smoothed, u, key, batch_id, options):

    B, T, D = mu_RPM.shape[0], mu_RPM.shape[1], mu_RPM.shape[2]

    posterior_entropy = 0.5 * D * T * (1 + np.log(2 * np.pi)) + batch_half_log_det(smoothed['smoothed_covariances']).sum()
    cross_entropy = 0.5 * np.einsum("tij,tij->", prior_JL["J"], smoothed['smoothed_covariances'])
    cross_entropy -= log_prob_under_prior(prior_params, smoothed['smoothed_means'], u)
    kl_qp = cross_entropy - posterior_entropy

    ce_qf = 0.5 * np.einsum("tij,tij->", J_RPM, smoothed['smoothed_covariances'])
    ce_qf -= MVN(loc=mu_RPM[batch_id], covariance_matrix=Sigma_RPM[batch_id]).log_prob(smoothed['smoothed_means']).sum()
    
    n_samples = options['num_MC_samples']
    keys = random.split(key, n_samples)
    ce_qF = - batch_expected_log_F(smoothed['smoothed_means'], smoothed['smoothed_covariances'], mu_RPM, Sigma_RPM, keys).mean()

    T_log_B = T * np.log(B)

    kl_qp /= (T * D)
    ce_qf /= (T * D)
    ce_qF /= (T * D)
    T_log_B /= (T * D)

    return kl_qp, ce_qf, ce_qF, T_log_B

def compute_free_energy_M_step(J_RPM, mu_RPM, Sigma_RPM, smoothed, key, batch_id, options):

    B, T, D = mu_RPM.shape[0], mu_RPM.shape[1], mu_RPM.shape[2]

    ce_qf = 0.5 * np.einsum("tij,tij->", J_RPM, smoothed['smoothed_covariances'])
    ce_qf -= MVN(loc=mu_RPM[batch_id], covariance_matrix=Sigma_RPM[batch_id]).log_prob(smoothed['smoothed_means']).sum()
    
    n_samples = options['num_MC_samples']
    keys = random.split(key, n_samples)
    ce_qF = - batch_expected_log_F(smoothed['smoothed_means'], smoothed['smoothed_covariances'], mu_RPM, Sigma_RPM, keys).mean()

    T_log_B = T * np.log(B)

    ce_qf /= (T * D)
    ce_qF /= (T * D)
    T_log_B /= (T * D)

    return ce_qf, ce_qF, T_log_B

def get_RPM_factors(params, opt_states, y, options):

    rpm_opt_state, _, _, _, RPM_time_varying_opt_state = opt_states

    # RPM_constant = rpm_opt_state.apply_fn(params["rpm_params"], y)

    # T = y.shape[1]
    # if options['use_LDS_for_F_in_q']:

    #     RPM_time_varying = marginal(params['prior_params'], T) # treat parameters of p(z'|z) as free (implicit) parameters of the q distribution

    # elif options['use_GRU_for_F_in_q']:

    #     RPM_time_varying = RPM_time_varying_opt_state.apply_fn(params["RPM_time_varying_params"])

    # RPM = {}
    # RPM['J'] = RPM_time_varying['J'][None] + RPM_constant["J"]
    # RPM['h'] = RPM_time_varying['h'][None] + RPM_constant["h"]
    # RPM['Sigma'] = vmap(vmap(lambda S: psd_solve(S, np.eye(S.shape[-1]))))(RPM['J'])
    # RPM['mu'] = np.einsum("hijk,hik->hij", RPM['Sigma'], RPM['h'])

    # RPM = rpm_opt_state.apply_fn(params["rpm_params"], y)

    RPM = RPM_time_varying_opt_state.apply_fn(params["RPM_time_varying_params"], y)

    return RPM

def get_posterior(params, opt_states, y, u):

    _, delta_q_opt, _, _, _ = opt_states

    smoothed = vmap(delta_q_opt.apply_fn, in_axes=(None,0))(params["delta_q_params"], np.concatenate((y, u), axis=2))

    return smoothed

def get_free_energy_E_step(params, prior_params, prior_JL, opt_states, y, u, RPM, key, beta, options):

    smoothed = get_posterior(params, opt_states, y, u)

    B = y.shape[0]
    keys = random.split(key, B)
    kl_qp, ce_qf, ce_qF, T_log_B = vmap(compute_free_energy_E_step, in_axes=(None,None,0,None,None,0,0,0,0,None))(prior_params, prior_JL, RPM['J'], RPM['mu'], RPM['Sigma'], smoothed, u, keys, np.arange(B), options)
    free_energy = - beta * kl_qp - ce_qf + ce_qF - T_log_B

    return -free_energy.mean(), (kl_qp, ce_qf, ce_qF)

get_value_and_grad_E_step = value_and_grad(get_free_energy_E_step, has_aux=True)

def one_E_Step(carry, inputs, options):

    params, prior_JL, opt_states, y, u, RPM, beta = carry
    key = inputs

    (loss, (kl_qp, ce_qf, ce_qF)), grads = get_value_and_grad_E_step(params, params['prior_params'], prior_JL, opt_states, y, u, RPM, key, beta, options)
    params, opt_states = params_update_E_step(grads, opt_states)

    carry = params, prior_JL, opt_states, y, u, RPM, beta
    outputs = loss, kl_qp, ce_qf, ce_qF

    return carry, outputs

def E_step(params, opt_states, y, u, key, beta, options):

    RPM = get_RPM_factors(params, opt_states, y, options)

    T = y.shape[1]
    prior_JL = dynamics_to_tridiag(params['prior_params'], T)

    # perform multiple gradient ascent steps on the q (while keeping parameters fixed)
    carry = params, prior_JL, opt_states, y, u, RPM, beta
    keys = random.split(key, options['num_E_steps'])
    (params, _, opt_states, _, _, _, _), (loss, kl_qp, ce_qf, ce_qF) = scan(partial(one_E_Step, options=options), carry, keys)

    smoothed = get_posterior(params, opt_states, y, u)

    return params, opt_states, smoothed, (loss, kl_qp, ce_qf, ce_qF)

def get_free_energy_M_step(params, opt_states, y, key, smoothed, options):

    RPM = get_RPM_factors(params, opt_states, y, options)

    keys = random.split(key, B)
    ce_qf, ce_qF, T_log_B = vmap(compute_free_energy_M_step, in_axes=(0,None,None,0,0,0,None))(RPM['J'], RPM['mu'], RPM['Sigma'], smoothed, keys, np.arange(B), options)
    free_energy = - ce_qf + ce_qF - T_log_B

    return -free_energy.mean(), (ce_qf, ce_qF)

get_value_and_grad_M_step = value_and_grad(get_free_energy_M_step, has_aux=True)

def one_M_Step(carry, inputs, options):

    params, opt_states, y, u, smoothed = carry
    key = inputs

    (loss, (ce_qf, ce_qF)), grads = get_value_and_grad_M_step(params, opt_states, y, key, smoothed, options)
    params, opt_states = params_update_M_step(grads, opt_states)

    carry = params, opt_states, y, u, smoothed
    outputs = loss, ce_qf, ce_qF

    return carry, outputs

def M_step(params, opt_states, y, u, smoothed, key, options):

    # perform multiple gradient ascent steps on the RPM parameters (while keeping q fixed)
    carry = params, opt_states, y, u, smoothed
    keys = random.split(key, options['num_M_steps'])
    (params, opt_states, _, _, _), (loss, ce_qf, ce_qF) = scan(partial(one_M_Step, options=options), carry, keys)

    return params, opt_states, (loss, ce_qf, ce_qF)

def params_update_M_step(grads, opt_states):
    
    rpm_opt_state, delta_q_opt, prior_opt_state, control_state, RPM_time_varying_opt_state = opt_states

    rpm_opt_state = rpm_opt_state.apply_gradients(grads = grads["rpm_params"])

    prior_opt_state = prior_opt_state.apply_gradients(grads = grads["prior_params"])

    RPM_time_varying_opt_state = RPM_time_varying_opt_state.apply_gradients(grads = grads["RPM_time_varying_params"])

    params = {}
    params["rpm_params"] = rpm_opt_state.params
    params["delta_q_params"] = delta_q_opt.params
    params["prior_params"] = prior_opt_state.params
    params["u_emb_params"] = control_state.params
    params["RPM_time_varying_params"] = RPM_time_varying_opt_state.params

    return params, [rpm_opt_state, delta_q_opt, prior_opt_state, control_state, RPM_time_varying_opt_state]

def params_update_E_step(grads, opt_states):
    
    rpm_opt_state, delta_q_opt, prior_opt_state, control_state, RPM_time_varying_opt_state = opt_states

    delta_q_opt = delta_q_opt.apply_gradients(grads = grads["delta_q_params"])

    params = {}
    params["rpm_params"] = rpm_opt_state.params
    params["delta_q_params"] = delta_q_opt.params
    params["prior_params"] = prior_opt_state.params
    params["u_emb_params"] = control_state.params
    params["RPM_time_varying_params"] = RPM_time_varying_opt_state.params

    return params, [rpm_opt_state, delta_q_opt, prior_opt_state, control_state, RPM_time_varying_opt_state]

def train_step(params, opt_states, y, u, key, beta, options):

    key_E_step, key_M_step = random.split(key, 2)

    # E step (variational EM)
    params, opt_states, smoothed, (loss, kl_qp, ce_qf, ce_qF) = E_step(params, opt_states, y, u, key_E_step, beta, options)

    # closed form update for prior parameters
    params = closed_form_LDS_updates(params, smoothed, u, mean_field_q=True)

    # M step (generalised EM)
    params, opt_states, (loss, ce_qf, ce_qF) = M_step(params, opt_states, y, u, smoothed, key_M_step, options)

    return params, opt_states, loss, ce_qf, ce_qF, smoothed['smoothed_means']

def get_train_state(ckpt_metrics_dir, all_models, all_optimisers=[], all_params=[]):

    options = CheckpointManagerOptions(max_to_keep=3, best_fn=lambda metrics: metrics, best_mode='min')
    mngr = CheckpointManager(ckpt_metrics_dir,  
                             {'rpm_model_state': AsyncCheckpointer(PyTreeCheckpointHandler()),
                              'delta_q_state': AsyncCheckpointer(PyTreeCheckpointHandler()),
                              'prior_model_state': AsyncCheckpointer(PyTreeCheckpointHandler()),
                              'u_emb_model_state': AsyncCheckpointer(PyTreeCheckpointHandler()),
                              'RPM_time_varying_model_state': AsyncCheckpointer(PyTreeCheckpointHandler())},
                             options)

    states = []
    for params, optimiser, model in zip(all_params, all_optimisers, all_models):

        states.append(train_state.TrainState.create(apply_fn = model.apply, params = params, tx = optimiser))
        
    return states, mngr

B = 250
T = 100
D = 2
U = 1
h_dim_rpm = 50
h_dim_u_emb = 50
carry_dim = 50
prior_lr = 1e-3
learning_rate = 1e-3
max_grad_norm = 10
n_epochs = 5000
log_every = 250

# from jax import config
# config.update("jax_enable_x64", True)

# yesterday did well with 1e-3 LR and 5 E/M steps, or 1e-2 LR and 1 E/M steps

options = {}
options['normalise_y'] = True
options['embed_u'] = False
options['f_time_dependent'] = False
options['initialise_via_M_step'] = False
options['num_MC_samples'] = 1
options['num_E_steps'] = 1
options['num_M_steps'] = 1
options["beta_init_value"] = 0.
options["beta_end_value"] = 1.
options["beta_transition_begin"] = 1000
options["beta_transition_steps"] = 1000
options['fit_LDS'] = True
options['save_dir'] = "/nfs/nhome/live/jheald/svae/my_code/runs"
options['project_name'] = 'RPM-mycode'

seed = 5
subkey1, subkey2, subkey3, subkey4, subkey5, subkey6, subkey7, key = random.split(random.PRNGKey(seed), 8)

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

RPM = rpm_network(z_dim=D, h_dim=h_dim_rpm)
# RPM = delta_q_params(carry_dim=carry_dim, z_dim=D)
delta_q = delta_q_params(carry_dim=carry_dim, z_dim=D)
params = {}
if options['f_time_dependent']:
    params["rpm_params"] = RPM.init(y = np.ones((D+1,)), rngs = {'params': subkey1})
    params["delta_q_params"] = delta_q.init(y = np.ones((T,D+U+1)), rngs = {'params': subkey7})
else:
    params["rpm_params"] = RPM.init(y = np.ones((D,)), rngs = {'params': subkey1})
    params["delta_q_params"] = delta_q.init(y = np.ones((T,D+U,)), rngs = {'params': subkey7})

if options['initialise_via_M_step']:
    params["prior_params"] = initialise_LDS_params_via_M_step(RPM, params["rpm_params"], y, u, subkey2, options, closed_form_M_Step=True)
else:
    params["prior_params"] = initialise_LDS_params(D, U, subkey2, closed_form_M_Step=True)

u_emb = control_network(u_emb_dim=U, h_dim=h_dim_u_emb)
params["u_emb_params"] = u_emb.init(u = np.ones((U,)), rngs = {'params': subkey5})

# RPM_time_varying = GRU_base(carry_dim=carry_dim, z_dim=D, T=T)
# params["RPM_time_varying_params"] = RPM_time_varying.init(rngs = {'params': subkey6})
RPM_time_varying = GRU_RPM(carry_dim=carry_dim, h_dim=h_dim_rpm, z_dim=D, T=T)
params["RPM_time_varying_params"] = RPM_time_varying.init(y = np.zeros((B,T,D)),rngs = {'params': subkey6})

rpm_opt = opt.chain(opt.adam(learning_rate=learning_rate), opt.clip_by_global_norm(max_grad_norm))
delta_q_opt = opt.chain(opt.adam(learning_rate=learning_rate), opt.clip_by_global_norm(max_grad_norm))
prior_opt = opt.chain(opt.adam(learning_rate=prior_lr), opt.clip_by_global_norm(max_grad_norm))
u_emb_opt = opt.chain(opt.adam(learning_rate=learning_rate), opt.clip_by_global_norm(max_grad_norm))
RPM_time_varying_opt = opt.chain(opt.adam(learning_rate=learning_rate), opt.clip_by_global_norm(max_grad_norm))

all_optimisers = (rpm_opt, delta_q_opt, prior_opt, u_emb_opt, RPM_time_varying_opt)
all_params = (params["rpm_params"], params["delta_q_params"], params["prior_params"], params["u_emb_params"], params["RPM_time_varying_params"])
all_models = (RPM, delta_q, RPM, u_emb, RPM_time_varying)
opt_states, mngr = get_train_state(options['save_dir'], all_models, all_optimisers, all_params)

beta_schedule = get_beta_schedule(options)

train_step_jit = jit(partial(train_step, options=options))

print("pass params around via opt_states not separately")
print("not sure how to deal with u_embed in EM (not embedding at the moment)")
print("commented out RPM_time_varying option to use f_time_dependent")

pbar = trange(n_epochs)
for itr in pbar:

    subkey, key = random.split(key, 2)

    beta = beta_schedule(itr)

    if options['fit_LDS']:

        params, opt_states, loss, ce_qf, ce_qF, mu_posterior = train_step_jit(params, opt_states, y, u, subkey, beta)
        R2, predicted_z = R2_inferred_vs_actual_z(mu_posterior, smoothed_true_params['smoothed_means'])

    else:

        params, opt_states, loss, ce_qf, ce_qF, mu_posterior = train_step_jit(params, opt_states, y[:B,:,:], u[:B,:,None], subkey, beta)
        R2 = 0.

    pbar.set_description("train loss: {:.3f},  ce_qf: {:.3f}, ce_qF: {:.3f}, R2 train states: {:.3f}".format(loss[-1], ce_qf[-1,:].mean(), ce_qF[-1,:].mean(), R2))

    if itr % log_every == 0:

        log_to_wandb(loss, np.array(0.), ce_qf, ce_qF, predicted_z, smoothed_true_params['smoothed_means'], options)

        # from matplotlib import pyplot as plt
        # import seaborn as sns

        # mu_r = mu_posterior.reshape(100,100,3)

        # palette = sns.color_palette(None, D)

        # cnt = 1
        # n_rows = 3
        # n_cols = 2
        # for row in range(n_rows):
        #     for col in range(n_cols):
        #         plt.subplot(n_rows, n_cols, cnt)
        #         for d in range(D):
        #             if col == 0:
        #                 plt.plot(y[row,:,d],'--', c=palette[d])
        #             else:
        #                 plt.plot(mu_r[row,:,d],'--', c=palette[d])
        #         cnt += 1
        # plt.show()

breakpoint()

# from matplotlib import pyplot as plt
# # # plt.plot((true_prior_params['C'] @ true_z[0,:,:].T + true_prior_params['d'][None].T).T,'r')
# plt.plot(y[0,:,:])
# plt.show()
# breakpoint()