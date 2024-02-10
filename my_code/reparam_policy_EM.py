from jax import numpy as np
from jax import random, jit, vmap, value_and_grad
from utils import kl_qp_natural_parameters, batch_expected_log_F, entropy, initialise_LDS_params, generate_LDS_params, batch_generate_data, R2_true_model, construct_covariance_matrix, update_prior, marginal_u_integrated_out, marginal, policy_loss, truncate_singular_values, R2_inferred_vs_actual_z, scale_y, load_pendulum_control_data, moment_match_RPM, get_marginals_of_joint, log_to_wandb, batch_perform_Kalman_smoothing, closed_form_LDS_updates, dynamics_to_tridiag
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

class delta_q_params(nn.Module):
    carry_dim: int
    z_dim: int

    def setup(self):
        
        self.BiGRU = nn.Bidirectional(nn.RNN(nn.GRUCell(self.carry_dim)), nn.RNN(nn.GRUCell(self.carry_dim)))

        self.dense = nn.Dense(self.z_dim + self.z_dim * (self.z_dim + 1) // 2)

    def __call__(self, y):

        def get_natural_parameters(x):

            mu, var_output_flat = np.split(x, [self.z_dim])

            Sigma = construct_covariance_matrix(var_output_flat, self.z_dim)

            J = psd_solve(Sigma, np.eye(self.z_dim))
            h = J @ mu

            return Sigma, mu, J, h

        concatenated_carry = self.BiGRU(y)

        out = self.dense(concatenated_carry)

        Sigma, mu, J, h = vmap(get_natural_parameters)(out)

        return {'Sigma': Sigma, 'mu': mu, 'J': J, 'h': h}

class F_for_q(nn.Module):
    carry_dim: int
    z_dim: int

    def setup(self):
        
        self.GRU = nn.RNN(nn.GRUCell(self.carry_dim))

        self.dense = nn.Dense(self.z_dim + self.z_dim * (self.z_dim + 1) // 2)

    def __call__(self, inputs):

        def get_natural_parameters(x):

            mu, var_output_flat = np.split(x, [self.z_dim])

            Sigma = construct_covariance_matrix(var_output_flat, self.z_dim)

            J = psd_solve(Sigma, np.eye(self.z_dim))
            h = J @ mu

            return Sigma, mu, J, h

        carry = self.GRU(inputs)

        out = self.dense(carry)

        Sigma, mu, J, h = vmap(get_natural_parameters)(out)

        return {'Sigma': Sigma, 'mu': mu, 'J': J, 'h': h}

class control_network(nn.Module):
    u_emb_dim: int
    h_dim: int

    def setup(self):

        self.dense_1 = nn.Dense(features=self.h_dim)
        self.dense_2 = nn.Dense(features=self.h_dim)
        self.dense_3 = nn.Dense(features=self.h_dim)
        self.dense_4 = nn.Dense(features=self.u_emb_dim)

    def __call__(self, u):

        def embed_u(u):

            # x = nn.LayerNorm()(y)
             # x = self.dense_1(x)
            x = self.dense_1(u)
            x = nn.relu(x)
            x = self.dense_2(x)
            x = nn.relu(x)
            x = self.dense_3(x)
            x = nn.relu(x)
            x = self.dense_4(x)

            return x

        if u.ndim == 1:
            x = embed_u(u)
        elif u.ndim == 2:
            x = vmap(embed_u)(u)
        elif u.ndim == 3:
            x = vmap(vmap(embed_u))(u)

        return x

class rpm_network(nn.Module):
    z_dim: int
    h_dim: int

    def setup(self):

        self.dense_1 = nn.Dense(features=self.h_dim)
        self.dense_2 = nn.Dense(features=self.h_dim)
        self.dense_3 = nn.Dense(features=self.h_dim)
        self.dense_4 = nn.Dense(features=self.z_dim + self.z_dim * (self.z_dim + 1) // 2)

    def __call__(self, y):

        def get_natural_parameters(y):

            # x = nn.LayerNorm()(y)
             # x = self.dense_1(x)
            x = self.dense_1(y)
            
            x = nn.relu(x)
            x = self.dense_2(x)
            x = nn.relu(x)
            x = self.dense_3(x)
            x = nn.relu(x)
            x = self.dense_4(x)

            h, var_output_flat = np.split(x, [self.z_dim])

            Sigma = construct_covariance_matrix(var_output_flat, self.z_dim)
            J = psd_solve(Sigma, np.eye(self.z_dim))
            mu = Sigma @ h

            return Sigma, mu, J, h

        if y.ndim == 1:
            Sigma, mu, J, h = get_natural_parameters(y)
        elif y.ndim == 2:
            Sigma, mu, J, h = vmap(get_natural_parameters)(y)
        elif y.ndim == 3:
            Sigma, mu, J, h = vmap(vmap(get_natural_parameters))(y)

        return {'Sigma': Sigma, 'mu': mu, 'J': J, 'h': h}

def log_prob_under_prior(prior_params, x, u):

    def log_prop_one_transition(A, x_prev, B, u, Q, x):

        return MVN(loc=A @ x_prev + B @ u, covariance_matrix=Q).log_prob(x)

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

    # ll = log_prob_under_prior(prior_params, x, u)
    # ll += log_prop_all_emissions(emission_potentials["mu"], emission_potentials["Sigma"], smoothed['smoothed_means']).sum()
    ll = log_prop_all_emissions(emission_potentials["mu"], emission_potentials["Sigma"], smoothed['smoothed_means']).sum()
    ll -= log_normalizer(prior_params, smoothed, emission_potentials, u)

    return ll

def compute_free_energy_E_step(prior_params, prior_JL, J_RPM, mu_RPM, Sigma_RPM, smoothed, emission_potentials, u, key, batch_id):

    B, T, D = mu_RPM.shape[0], mu_RPM.shape[1], mu_RPM.shape[2]

    posterior_entropy = 0.5 * D * T - log_prob_under_posterior(prior_params, emission_potentials, smoothed, u)
    cross_entropy = 0.5 * np.einsum("tij,tij->", prior_JL["J"], smoothed['smoothed_covariances'])
    Sigmatnt = smoothed['smoothed_cross_covariances'] - np.einsum("ti,tj->tij", smoothed['smoothed_means'][:-1], smoothed['smoothed_means'][1:])
    cross_entropy += np.einsum("tij,tij->", prior_JL["L"], Sigmatnt) # no 0.5 weighting because this term is counted twice (once for the lower diagonal and once for the upper diagonal)
    # cross_entropy -= log_prob_under_prior(prior_params, smoothed['smoothed_means'], u) # commented out along with call in log_prob_under_posterior, as they cancel
    kl_qp = cross_entropy - posterior_entropy

    ce_qf = 0.5 * np.einsum("tij,tij->", J_RPM, smoothed['smoothed_covariances'])
    ce_qf -= MVN(loc=mu_RPM[batch_id], covariance_matrix=Sigma_RPM[batch_id]).log_prob(smoothed['smoothed_means']).sum()
    
    n_samples = 10
    keys = random.split(key, n_samples)
    ce_qF = - batch_expected_log_F(smoothed['smoothed_means'], smoothed['smoothed_covariances'], mu_RPM, Sigma_RPM, keys).mean()

    T_log_B = T * np.log(B)

    kl_qp /= (T * D)
    ce_qf /= (T * D)
    ce_qF /= (T * D)
    T_log_B /= (T * D)

    return kl_qp, ce_qf, ce_qF, T_log_B

def compute_free_energy_M_step(J_RPM, mu_RPM, Sigma_RPM, smoothed, key, batch_id):

    B, T, D = mu_RPM.shape[0], mu_RPM.shape[1], mu_RPM.shape[2]

    ce_qf = 0.5 * np.einsum("tij,tij->", J_RPM, smoothed['smoothed_covariances'])
    ce_qf -= MVN(loc=mu_RPM[batch_id], covariance_matrix=Sigma_RPM[batch_id]).log_prob(smoothed['smoothed_means']).sum()
    
    n_samples = 10
    keys = random.split(key, n_samples)
    ce_qF = - batch_expected_log_F(smoothed['smoothed_means'], smoothed['smoothed_covariances'], mu_RPM, Sigma_RPM, keys).mean()

    T_log_B = T * np.log(B)

    ce_qf /= (T * D)
    ce_qF /= (T * D)
    T_log_B /= (T * D)

    return ce_qf, ce_qF, T_log_B

# def E_step(params, opt_states, y, u, key):

#     rpm_opt_state, delta_q_opt, _, _, _ = opt_states

#     RPM_norm = rpm_opt_state.apply_fn(params["rpm_params"], y)

#     smoothed = batch_perform_Kalman_smoothing(params['prior_params'], RPM_norm, u)

#     return smoothed

def get_RPM_factors(params, opt_states, y, options):

    rpm_opt_state, _, _, _, F_approx_opt_state = opt_states

    RPM_constant = rpm_opt_state.apply_fn(params["rpm_params"], y)

    T = y.shape[1]
    if options['use_LDS_for_F_in_q']:

        RPM_time_varying = marginal(params['prior_params'], T) # treat parameters of p(z'|z) as free (implicit) parameters of the q distribution

    elif options['use_GRU_for_F_in_q']:

        RPM_time_varying = F_approx_opt_state.apply_fn(params["F_approx_params"], np.zeros((T,1)))

    RPM = {}
    RPM['J'] = RPM_time_varying['J'][None] + RPM_constant["J"]
    RPM['h'] = RPM_time_varying['h'][None] + RPM_constant["h"]
    RPM['Sigma'] = vmap(vmap(lambda S: psd_solve(S, np.eye(S.shape[-1]))))(RPM['J'])
    RPM['mu'] = np.einsum("hijk,hik->hij", RPM['Sigma'], RPM['h'])

    return RPM_constant, RPM

def get_posterior(params, prior_params, opt_states, y, u, RPM_constant):

    _, delta_q_opt, _, _, _ = opt_states

    delta_q_potentials = vmap(delta_q_opt.apply_fn, in_axes=(None,0))(params["delta_q_params"], y)

    emission_potentials = {}
    emission_potentials['Sigma'] = vmap(vmap(lambda J1, J2: psd_solve(J1 + J2, np.eye(J1.shape[-1]))))(RPM_constant['J'], delta_q_potentials['J'])
    emission_potentials['mu'] = vmap(vmap(lambda h1, h2, S: S @ (h1 + h2)))(RPM_constant['h'], delta_q_potentials['h'], emission_potentials['Sigma'])

    smoothed = batch_perform_Kalman_smoothing(prior_params, emission_potentials, u)

    return smoothed, emission_potentials

def get_free_energy_E_step(params, prior_params, prior_JL, opt_states, y, u, RPM_constant, RPM, key):

    smoothed, emission_potentials = get_posterior(params, prior_params, opt_states, y, u, RPM_constant)

    B = y.shape[0]
    keys = random.split(key, B)
    kl_qp, ce_qf, ce_qF, T_log_B = vmap(compute_free_energy_E_step, in_axes=(None,None,0,None,None,0,0,0,0,0))(prior_params, prior_JL, RPM['J'], RPM['mu'], RPM['Sigma'], smoothed, emission_potentials, u, keys, np.arange(B))
    free_energy = - kl_qp - ce_qf + ce_qF - T_log_B

    return -free_energy.mean(), (kl_qp, ce_qf, ce_qF)

get_value_and_grad_E_step = value_and_grad(get_free_energy_E_step, has_aux=True)

def one_E_Step(carry, inputs):

    params, prior_JL, opt_states, y, u, RPM_constant, RPM = carry
    key = inputs

    (loss, (kl_qp, ce_qf, ce_qF)), grads = get_value_and_grad_E_step(params, params['prior_params'], prior_JL, opt_states, y, u, RPM_constant, RPM, key)
    params, opt_states = params_update_E_step(grads, opt_states)

    carry = params, prior_JL, opt_states, y, u, RPM_constant, RPM
    outputs = loss, kl_qp, ce_qf, ce_qF

    return carry, outputs

def E_step(params, opt_states, y, u, key, options):

    RPM_constant, RPM = get_RPM_factors(params, opt_states, y, options)

    T = y.shape[1]
    prior_JL = dynamics_to_tridiag(params['prior_params'], T)

    # perform multiple gradient ascent steps on the q (while keeping parameters fixed)
    carry = params, prior_JL, opt_states, y, u, RPM_constant, RPM
    keys = random.split(key, options['num_E_steps'])
    (params, _, opt_states, _, _, _, _), (loss, kl_qp, ce_qf, ce_qF) = scan(one_E_Step, carry, keys)

    smoothed, _ = get_posterior(params, params['prior_params'], opt_states, y, u, RPM_constant)

    return params, opt_states, smoothed, (loss, kl_qp, ce_qf, ce_qF)

def get_free_energy_M_step(params, opt_states, y, key, smoothed, options):

    _, RPM = get_RPM_factors(params, opt_states, y, options)

    keys = random.split(key, B)
    ce_qf, ce_qF, T_log_B = vmap(compute_free_energy_M_step, in_axes=(0,None,None,0,0,0))(RPM['J'], RPM['mu'], RPM['Sigma'], smoothed, keys, np.arange(B))
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
    
    rpm_opt_state, delta_q_opt, prior_opt_state, control_state, F_approx_opt_state = opt_states

    rpm_opt_state = rpm_opt_state.apply_gradients(grads = grads["rpm_params"])

    prior_opt_state = prior_opt_state.apply_gradients(grads = grads["prior_params"])
    prior_opt_state.params["A_F"] = truncate_singular_values(prior_opt_state.params["A_F"])

    F_approx_opt_state = F_approx_opt_state.apply_gradients(grads = grads["F_approx_params"])

    params = {}
    params["rpm_params"] = rpm_opt_state.params
    params["delta_q_params"] = delta_q_opt.params
    params["prior_params"] = prior_opt_state.params
    params["u_emb_params"] = control_state.params
    params["F_approx_params"] = F_approx_opt_state.params

    return params, [rpm_opt_state, delta_q_opt, prior_opt_state, control_state, F_approx_opt_state]

def params_update_E_step(grads, opt_states):
    
    rpm_opt_state, delta_q_opt, prior_opt_state, control_state, F_approx_opt_state = opt_states

    delta_q_opt = delta_q_opt.apply_gradients(grads = grads["delta_q_params"])

    params = {}
    params["rpm_params"] = rpm_opt_state.params
    params["delta_q_params"] = delta_q_opt.params
    params["prior_params"] = prior_opt_state.params
    params["u_emb_params"] = control_state.params
    params["F_approx_params"] = F_approx_opt_state.params

    return params, [rpm_opt_state, delta_q_opt, prior_opt_state, control_state, F_approx_opt_state]

def train_step(params, opt_states, y, u, key, options):

    key_E_step, key_M_step = random.split(key, 2)

    # E step
    params, opt_states, smoothed, (loss, kl_qp, ce_qf, ce_qF) = E_step(params, opt_states, y, u, key_E_step, options)

    # closed form update for prior parameters
    params = closed_form_LDS_updates(params, smoothed, u, mean_field_q=False)

    # M step
    params, opt_states, (loss, ce_qf, ce_qF) = M_step(params, opt_states, y, u, smoothed, key_M_step, options)

    return params, opt_states, loss, ce_qf, ce_qF, smoothed['smoothed_means']

def get_train_state(ckpt_metrics_dir, all_models, all_optimisers=[], all_params=[]):

    options = CheckpointManagerOptions(max_to_keep=3, best_fn=lambda metrics: metrics, best_mode='min')
    mngr = CheckpointManager(ckpt_metrics_dir,  
                             {'rpm_model_state': AsyncCheckpointer(PyTreeCheckpointHandler()),
                              'delta_q_state': AsyncCheckpointer(PyTreeCheckpointHandler()),
                              'prior_model_state': AsyncCheckpointer(PyTreeCheckpointHandler()),
                              'u_emb_model_state': AsyncCheckpointer(PyTreeCheckpointHandler()),
                              'F_approx_model_state': AsyncCheckpointer(PyTreeCheckpointHandler())},
                             options)

    states = []
    for params, optimiser, model in zip(all_params, all_optimisers, all_models):

        states.append(train_state.TrainState.create(apply_fn = model.apply, params = params, tx = optimiser))
        
    return states, mngr

B = 100
T = 100
D = 2
U = 1
h_dim_rpm = 50
h_dim_u_emb = 5
carry_dim = 10
prior_lr = 1e-3
learning_rate = 1e-3
max_grad_norm = 10
n_epochs = 5000
log_every = 250

options = {}
options['normalise_y'] = True
options['embed_u'] = False
options['use_LDS_for_F_in_q'] = True
# options['explicitly_integrate_out_u'] = True
options['use_GRU_for_F_in_q'] = False
# options['use_MM_for_F_in_q'] = False
# options['use_policy_loss'] = False
options['f_time_dependent'] = False
options['num_E_steps'] = 1
options['num_M_steps'] = 1
options['fit_LDS'] = True
options['save_dir'] = "/nfs/nhome/live/jheald/svae/my_code/runs"
options['project_name'] = 'RPM-mycode'

seed = 5
subkey1, subkey2, subkey3, subkey4, subkey5, subkey6, subkey7, key = random.split(random.PRNGKey(seed), 8)

RPM = rpm_network(z_dim=D, h_dim=h_dim_rpm)
delta_q = delta_q_params(carry_dim=carry_dim, z_dim=D)
params = {}
if options['f_time_dependent']:
    params["rpm_params"] = RPM.init(y = np.ones((D+1,)), rngs = {'params': subkey1})
    params["delta_q_params"] = delta_q.init(y = np.ones((T,D+1)), rngs = {'params': subkey7})
else:
    params["rpm_params"] = RPM.init(y = np.ones((D,)), rngs = {'params': subkey1})
    params["delta_q_params"] = delta_q.init(y = np.ones((T,D,)), rngs = {'params': subkey7})

params["prior_params"] = initialise_LDS_params(D, U, subkey2, closed_form_M_Step=True)

u_emb = control_network(u_emb_dim=U, h_dim=h_dim_u_emb)
params["u_emb_params"] = u_emb.init(u = np.ones((U,)), rngs = {'params': subkey5})

F_approx = F_for_q(carry_dim=carry_dim, z_dim=D)
params["F_approx_params"] = F_approx.init(inputs = np.ones((T,1)), rngs = {'params': subkey6})

rpm_opt = opt.chain(opt.adam(learning_rate=learning_rate), opt.clip_by_global_norm(max_grad_norm))
delta_q_opt = opt.chain(opt.adam(learning_rate=learning_rate), opt.clip_by_global_norm(max_grad_norm))
prior_opt = opt.chain(opt.adam(learning_rate=prior_lr), opt.clip_by_global_norm(max_grad_norm))
u_emb_opt = opt.chain(opt.adam(learning_rate=learning_rate), opt.clip_by_global_norm(max_grad_norm))
F_approx_opt = opt.chain(opt.adam(learning_rate=learning_rate), opt.clip_by_global_norm(max_grad_norm))

all_optimisers = (rpm_opt, delta_q_opt, prior_opt, u_emb_opt, F_approx_opt)
all_params = (params["rpm_params"], params["delta_q_params"], params["prior_params"], params["u_emb_params"], params["F_approx_params"])
all_models = (RPM, delta_q, RPM, u_emb, F_approx)
opt_states, mngr = get_train_state(options['save_dir'], all_models, all_optimisers, all_params)

if options['fit_LDS']:

    true_prior_params = generate_LDS_params(D, U, subkey3)
    keys = random.split(subkey4, B)
    true_z, y, u = batch_generate_data(true_prior_params, T, D, U, keys)

    R2, predicted_z = R2_true_model(true_prior_params, y, u, true_z)
    print('R2 true model' , R2)

else:

    y, u = load_pendulum_control_data()

if options['f_time_dependent']:

    y = np.concatenate((y[:B,:,:], np.arange(T)[None, :, None].repeat(B, axis=0)),axis=2)

if options['normalise_y']:

    y = scale_y(y)

train_step_jit = jit(partial(train_step, options=options))

print("pass params around via opt_states not separately")
print("not sure how to deal with u_embed in EM (not embedding at the moment)")
print("Q1 and Q were same parameters, bug in get_constrained_prior_params - not sure how long it's been there for")

pbar = trange(n_epochs)
for itr in pbar:

    subkey, key = random.split(key, 2)

    if options['fit_LDS']:

        params, opt_states, loss, ce_qf, ce_qF, mu_posterior = train_step_jit(params, opt_states, y, u, subkey)
        R2, _ = R2_inferred_vs_actual_z(mu_posterior, true_z)

    else:

        params, opt_states, loss, ce_qf, ce_qF, mu_posterior = train_step_jit(params, opt_states, y[:B,:,:], u[:B,:,None], subkey)
        R2 = 0.

    pbar.set_description("train loss: {:.3f},  ce_qf: {:.3f}, ce_qF: {:.3f}, R2 train states: {:.3f}".format(loss[-1], ce_qf[-1,:].mean(), ce_qF[-1,:].mean(), R2))

    if itr % log_every == 0:

        log_to_wandb(loss, np.array(0.), ce_qf, ce_qF, mu_posterior.reshape(B,T,D), y, options)

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