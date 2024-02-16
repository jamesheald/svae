from jax import numpy as np
from jax import random, jit, vmap, value_and_grad
from utils import kl_qp_natural_parameters, batch_expected_log_F, entropy, initialise_LDS_params, generate_LDS_params, batch_generate_data, R2_true_model, construct_covariance_matrix, update_prior, marginal_u_integrated_out, marginal, policy_loss, truncate_singular_values, R2_inferred_vs_actual_z, scale_y, load_pendulum_control_data, moment_match_RPM, get_marginals_of_joint, log_to_wandb, batch_perform_Kalman_smoothing, closed_form_LDS_updates, dynamics_to_tridiag, get_beta_schedule, batch_perform_Kalman_smoothing_true_params, get_constrained_prior_params, initialise_LDS_params_via_M_step, log_prob_under_posterior
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

from networks import GRUCell_LN

class delta_q_params(nn.Module):
    carry_dim: int
    z_dim: int

    def setup(self):
        
        # self.BiGRU = nn.Bidirectional(nn.RNN(nn.GRUCell(self.carry_dim)), nn.RNN(nn.GRUCell(self.carry_dim)))
        self.BiGRU = nn.Bidirectional(nn.RNN(GRUCell_LN(self.carry_dim)), nn.RNN(GRUCell_LN(self.carry_dim)))

        # self.dense = nn.Dense(self.z_dim + self.z_dim * (self.z_dim + 1) // 2)
        self.dense = nn.Dense(self.z_dim*2)

    def __call__(self, y):

        def get_natural_parameters(x):

            mu, var_output_flat = np.split(x, [self.z_dim])

            # Sigma = construct_covariance_matrix(var_output_flat, self.z_dim)

            Sigma = np.diag(np.exp(var_output_flat)+1e-6)

            J = np.diag(1/np.exp(var_output_flat))

            # J = psd_solve(Sigma, np.eye(self.z_dim))
            h = J @ mu

            return Sigma, mu, J, h

        concatenated_carry = self.BiGRU(y)

        out = self.dense(concatenated_carry)

        Sigma, mu, J, h = vmap(get_natural_parameters)(out)

        return {'Sigma': Sigma, 'mu': mu, 'J': J, 'h': h}

class GRUCell_LN_NoInput(nn.RNNCellBase):

    features: int
    gate_fn: Callable[..., Any] = sigmoid
    activation_fn: Callable[..., Any] = tanh
    kernel_init: Initializer = default_kernel_init
    recurrent_kernel_init: Initializer = initializers.orthogonal()
    bias_init: Initializer = initializers.zeros_init()
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = np.float32
    carry_init: Initializer = initializers.zeros_init()

    @compact
    def __call__(self, carry, inputs):

        h = carry
        hidden_features = h.shape[-1]
        # input and recurrent layers are summed so only one needs a bias.
        dense_h = partial(
          Dense,
          features=hidden_features,
          use_bias=True,
          dtype=self.dtype,
          param_dtype=self.param_dtype,
          kernel_init=self.recurrent_kernel_init,
          bias_init=self.bias_init,
        )

        r = self.gate_fn(nn.LayerNorm()(dense_h(name='hr')(h)))
        z = self.gate_fn(nn.LayerNorm()(dense_h(name='hz')(h)))
        n = self.activation_fn(r * nn.LayerNorm()(dense_h(name='hn')(h))
          )

        new_h = (1.0 - z) * n + z * h
        return new_h, new_h

    @nowrap
    def initialize_carry(self, rng: PRNGKey, input_shape: Tuple[int, ...]):
        """Initialize the RNN cell carry.

        Args:
          rng: random number generator passed to the init_fn.
          input_shape: a tuple providing the shape of the input to the cell.

        Returns:
          An initialized carry for the given RNN cell.
        """
        batch_dims = input_shape[:-1]
        mem_shape = batch_dims + (self.features,)
        return self.carry_init(rng, mem_shape, self.param_dtype)

    @property
    def num_feature_axes(self) -> int:
        return 1

# network for creating time-dependent RPM factors f_t(z_t | x_t), where the time dependence is provided by a GRU
class GRU_RPM(nn.Module):
    carry_dim: int
    h_dim: int
    z_dim: int
    T: int

    def setup(self):
        
        self.GRU = nn.RNN(GRUCell_LN_NoInput(self.carry_dim))
        # self.GRU = nn.RNN(nn.GRUCell(self.carry_dim))

        self.carry_init = self.param('carry_init', lambda rng, shape: nn.initializers.normal(1.0)(rng, shape), (self.carry_dim,))

        self.dense_1 = nn.Dense(features=self.h_dim)
        self.LN_1 = nn.LayerNorm()
        self.dense_2 = nn.Dense(features=self.h_dim)
        self.LN_2 = nn.LayerNorm()
        self.dense_3 = nn.Dense(features=self.h_dim)
        self.LN_3 = nn.LayerNorm()
        # self.dense_4 = nn.Dense(features=self.z_dim + self.z_dim * (self.z_dim + 1) // 2)
        self.dense_4 = nn.Dense(features=2*self.z_dim)

    def __call__(self, y):

        def get_natural_parameters(x):

            mu, var_output_flat = np.split(x, [self.z_dim])

            # Sigma = construct_covariance_matrix(var_output_flat, self.z_dim)
            Sigma = np.diag(np.exp(var_output_flat))

            # J = psd_solve(Sigma, np.eye(self.z_dim))
            J = np.diag(1/np.exp(var_output_flat))
            h = J @ mu

            return Sigma, mu, J, h

        carry = self.GRU(np.zeros((self.T,1)), initial_carry=self.carry_init)

        # x = self.LN_1(self.dense_1(np.concatenate((carry[None].repeat(y.shape[0], axis=0), y), axis=2)))
        # x = nn.relu(x)
        # x = self.LN_2(self.dense_2(x))
        # x = nn.relu(x)
        # x = self.LN_3(self.dense_3(x))
        # x = nn.relu(x)
        # x = self.dense_4(x)

        x = self.LN_1(self.dense_1(y))
        x = nn.relu(x)
        x = self.LN_2(self.dense_2(x))
        x = nn.relu(x)
        x = self.LN_3(self.dense_3(x))
        x = nn.relu(x)
        x = self.dense_4(np.concatenate((carry[None].repeat(x.shape[0], axis=0), x), axis=2))

        Sigma, mu, J, h = vmap(vmap(get_natural_parameters))(x)

        return {'Sigma': Sigma, 'mu': mu, 'J': J, 'h': h}

class F_for_q(nn.Module):
    carry_dim: int
    z_dim: int
    T: int

    def setup(self):
        
        self.GRU = nn.RNN(GRUCell_LN_NoInput(self.carry_dim))
        # self.GRU = nn.RNN(nn.GRUCell(self.carry_dim))

        self.carry_init = self.param('carry_init', lambda rng, shape: nn.initializers.normal(1.0)(rng, shape), (self.carry_dim,))

        self.dense = nn.Dense(self.z_dim + self.z_dim * (self.z_dim + 1) // 2)

    def __call__(self):

        def get_natural_parameters(x):

            mu, var_output_flat = np.split(x, [self.z_dim])

            Sigma = construct_covariance_matrix(var_output_flat, self.z_dim)

            J = psd_solve(Sigma, np.eye(self.z_dim))
            h = J @ mu

            return Sigma, mu, J, h

        carry = self.GRU(np.zeros((self.T,1)) ,initial_carry=self.carry_init)

        out = self.dense(carry)

        Sigma, mu, J, h = vmap(get_natural_parameters)(out)

        return {'Sigma': Sigma, 'mu': mu, 'J': J, 'h': h}
        # def get_natural_parameters(x):

        #     h, var_output_flat = np.split(x, [self.z_dim])

        #     J = construct_precision_matrix(var_output_flat, self.z_dim)

        #     return J, h

        # carry = self.GRU(np.zeros((self.T,1)) ,initial_carry=self.carry_init)

        # out = self.dense(carry)

        # J, h = vmap(get_natural_parameters)(out)

        # return {'J': J, 'h': h}

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
        self.LN_1 = nn.LayerNorm()
        self.dense_2 = nn.Dense(features=self.h_dim)
        self.LN_2 = nn.LayerNorm()
        self.dense_3 = nn.Dense(features=self.h_dim)
        self.LN_3 = nn.LayerNorm()
        # self.dense_4 = nn.Dense(features=self.z_dim + self.z_dim * (self.z_dim + 1) // 2)
        self.dense_4 = nn.Dense(features=self.z_dim*2)

    def __call__(self, y):

        def get_natural_parameters(y):

            x = self.LN_1(self.dense_1(y))
            x = nn.relu(x)
            x = self.LN_2(self.dense_2(x))
            x = nn.relu(x)
            x = self.LN_3(self.dense_3(x))
            x = nn.relu(x)
            x = self.dense_4(x)

            h, var_output_flat = np.split(x, [self.z_dim])

            # Sigma = construct_covariance_matrix(var_output_flat, self.z_dim)
            Sigma = np.diag(np.exp(var_output_flat))
            # J = psd_solve(Sigma, np.eye(self.z_dim))
            J = np.diag(1/np.exp(var_output_flat))
            mu = Sigma @ h

            return Sigma, mu, J, h

        if y.ndim == 1:
            Sigma, mu, J, h = get_natural_parameters(y)
        elif y.ndim == 2:
            Sigma, mu, J, h = vmap(get_natural_parameters)(y)
        elif y.ndim == 3:
            Sigma, mu, J, h = vmap(vmap(get_natural_parameters))(y)

        return {'Sigma': Sigma, 'mu': mu, 'J': J, 'h': h}

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
    
    n_samples = options['num_MC_samples']
    keys = random.split(key, n_samples)
    ce_qF = - batch_expected_log_F(smoothed['smoothed_means'], smoothed['smoothed_covariances'], mu_RPM, Sigma_RPM, keys).mean()

    T_log_B = T * np.log(B)

    kl_qp /= (T * D)
    ce_qf /= (T * D)
    ce_qF /= (T * D)
    T_log_B /= (T * D)

    return kl_qp, ce_qf, ce_qF, T_log_B

def get_RPM_factors(params, opt_states, y, options):

    rpm_opt_state, _, _, _, F_approx_opt_state = opt_states

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

    _, delta_q_opt, _, _, _ = opt_states

    # delta_q_potentials = vmap(delta_q_opt.apply_fn, in_axes=(None,0))(params["delta_q_params"], y)

    # emission_potentials = {}
    # emission_potentials['Sigma'] = vmap(vmap(lambda J1, J2: psd_solve(J1 + J2, np.eye(J1.shape[-1]))))(RPM_constant['J'], delta_q_potentials['J'])
    # emission_potentials['mu'] = vmap(vmap(lambda h1, h2, S: S @ (h1 + h2)))(RPM_constant['h'], delta_q_potentials['h'], emission_potentials['Sigma'])

    emission_potentials = vmap(delta_q_opt.apply_fn, in_axes=(None,0))(params["delta_q_params"], y)

    smoothed = batch_perform_Kalman_smoothing(prior_params, emission_potentials, u)

    return smoothed, emission_potentials

def get_free_energy(params, opt_states, y, u, key, beta, options):

    RPM_constant, RPM = get_RPM_factors(params, opt_states, y, options)

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
    
    rpm_opt_state, delta_q_opt, prior_opt_state, control_state, F_approx_opt_state = opt_states

    rpm_opt_state = rpm_opt_state.apply_gradients(grads = grads["rpm_params"])

    delta_q_opt = delta_q_opt.apply_gradients(grads = grads["delta_q_params"])

    prior_opt_state = prior_opt_state.apply_gradients(grads = grads["prior_params"])
    prior_opt_state.params["A"] = truncate_singular_values(prior_opt_state.params["A"])
    # prior_opt_state.params["A_F"] = truncate_singular_values(prior_opt_state.params["A_F"])

    F_approx_opt_state = F_approx_opt_state.apply_gradients(grads = grads["F_approx_params"])

    params = {}
    params["rpm_params"] = rpm_opt_state.params
    params["delta_q_params"] = delta_q_opt.params
    params["prior_params"] = prior_opt_state.params
    params["u_emb_params"] = control_state.params
    params["F_approx_params"] = F_approx_opt_state.params

    return params, [rpm_opt_state, delta_q_opt, prior_opt_state, control_state, F_approx_opt_state]

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
                              'u_emb_model_state': AsyncCheckpointer(PyTreeCheckpointHandler()),
                              'F_approx_model_state': AsyncCheckpointer(PyTreeCheckpointHandler())},
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
h_dim_u_emb = 50
carry_dim = 50
prior_lr = 1e-3
learning_rate = 1e-3
max_grad_norm = 10
n_epochs = 5000
log_every = 250

options = {}
options['normalise_y'] = True
options['embed_u'] = False
options['use_LDS_for_F_in_q'] = False
# options['explicitly_integrate_out_u'] = True
options['use_GRU_for_F_in_q'] = True
# options['use_MM_for_F_in_q'] = False
# options['use_policy_loss'] = False
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

seed = 0
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

# RPM = rpm_network(z_dim=D, h_dim=h_dim_rpm)
RPM = GRU_RPM(carry_dim=carry_dim, h_dim=h_dim_rpm, z_dim=D, T=T)
delta_q = delta_q_params(carry_dim=carry_dim, z_dim=D)
# delta_q = rpm_network(h_dim=h_dim_rpm, z_dim=D)
params = {}
if options['f_time_dependent']:
    params["rpm_params"] = RPM.init(y = np.ones((D+1,)), rngs = {'params': subkey1})
    params["delta_q_params"] = delta_q.init(y = np.ones((T,D+1)), rngs = {'params': subkey7})
else:
    params["rpm_params"] = RPM.init(y = np.ones((B,T,D)), rngs = {'params': subkey1})
    # params["rpm_params"] = RPM.init(y = np.ones((D,)), rngs = {'params': subkey1})
    params["delta_q_params"] = delta_q.init(y = np.ones((T,D,)), rngs = {'params': subkey7})

if options['initialise_via_M_step']:
    params["prior_params"] = initialise_LDS_params_via_M_step(RPM, params["rpm_params"], y, u, subkey2, options, closed_form_M_Step=False)
else:
    params["prior_params"] = initialise_LDS_params(D, U, subkey2, closed_form_M_Step=False)

u_emb = control_network(u_emb_dim=U, h_dim=h_dim_u_emb)
params["u_emb_params"] = u_emb.init(u = np.ones((U,)), rngs = {'params': subkey5})

F_approx = F_for_q(carry_dim=carry_dim, z_dim=D, T=T)
params["F_approx_params"] = F_approx.init(rngs = {'params': subkey6})

rpm_opt = opt.chain(opt.adam(learning_rate=learning_rate), opt.clip_by_global_norm(max_grad_norm))
delta_q_opt = opt.chain(opt.adam(learning_rate=learning_rate), opt.clip_by_global_norm(max_grad_norm))
prior_opt = opt.chain(opt.adam(learning_rate=prior_lr), opt.clip_by_global_norm(max_grad_norm))
u_emb_opt = opt.chain(opt.adam(learning_rate=learning_rate), opt.clip_by_global_norm(max_grad_norm))
F_approx_opt = opt.chain(opt.adam(learning_rate=learning_rate), opt.clip_by_global_norm(max_grad_norm))

all_optimisers = (rpm_opt, delta_q_opt, prior_opt, u_emb_opt, F_approx_opt)
all_params = (params["rpm_params"], params["delta_q_params"], params["prior_params"], params["u_emb_params"], params["F_approx_params"])
all_models = (RPM, delta_q, RPM, u_emb, F_approx)
opt_states, mngr = get_train_state(options['save_dir'], all_models, all_optimisers, all_params)

beta_schedule = get_beta_schedule(options)

train_step_jit = jit(partial(train_step, options=options))

print("pass params around via opt_states not separately")
print("not sure how to deal with u_embed in EM (not embedding at the moment)")

pbar = trange(n_epochs)
for itr in pbar:

    subkey, key = random.split(key, 2)

    beta = beta_schedule(itr)

    if options['fit_LDS']:

        params, opt_states, loss, kl_qp, ce_qf, ce_qF, mu_posterior = train_step_jit(params, opt_states, y, u, subkey, beta)
        R2, predicted_z = R2_inferred_vs_actual_z(mu_posterior, smoothed_true_params['smoothed_means'])

    else:

        params, opt_states, loss, kl_qp, ce_qf, ce_qF, mu_posterior = train_step_jit(params, opt_states, y[:B,:,:], u[:B,:,None], subkey, beta)
        R2 = 0.

    pbar.set_description("train loss: {:.3f},  kl_qp: {:.3f}, ce_qf: {:.3f}, ce_qF: {:.3f}, R2 train states: {:.3f}".format(loss, kl_qp.mean(), ce_qf.mean(), ce_qF.mean(), R2))

    # prior_params = get_constrained_prior_params(params['prior_params'], U)
    # smoothed = batch_perform_Kalman_smoothing(prior_params, emission_potentials, u)

    if itr % log_every == 0:

        log_to_wandb(loss, kl_qp, ce_qf, ce_qF, predicted_z, smoothed_true_params['smoothed_means'], options)

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