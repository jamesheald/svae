rofrom jax import numpy as np
from jax import random, jit, vmap, value_and_grad
from utils import kl_qp_natural_parameters, batch_expected_log_F, entropy, get_constrained_prior_params, initialise_LDS_params, generate_LDS_params, batch_generate_data, R2_true_model, construct_covariance_matrix, update_prior, marginal_u_integrated_out, marginal, policy_loss, truncate_singular_values, R2_inferred_vs_actual_z, scale_y, load_pendulum_control_data, moment_match_RPM, get_marginals_of_joint, log_to_wandb, initialise_LDS_params_M_step
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

def compute_free_energy(prior_params, prior, J_RPM, h_RPM, mu_RPM, Sigma_RPM, posterior, u, key, batch_id):

    B, T, D = mu_RPM.shape[0], h_RPM.shape[0], h_RPM.shape[1]

    kl_qp = kl_qp_natural_parameters(posterior['J'], posterior['h'], prior['J'], prior['h'])

    Sigma_posterior = psd_solve(posterior['J'], np.eye(posterior['J'].shape[-1]))
    mu_posterior = Sigma_posterior @ posterior['h']
    
    mu_posterior_marginal, Sigma_posterior_marginal = get_marginals_of_joint(mu_posterior, Sigma_posterior, T, D)

    ce_qf = 0.5 * np.einsum("tij,tij->", J_RPM, Sigma_posterior_marginal)
    ce_qf -= MVN(loc=mu_RPM[batch_id], covariance_matrix=Sigma_RPM[batch_id]).log_prob(mu_posterior_marginal).sum()
    
    n_samples = 10
    keys = random.split(key, n_samples)
    ce_qF = - batch_expected_log_F(mu_posterior, Sigma_posterior, mu_RPM, Sigma_RPM, keys).mean()

    # loss_policy = vmap(policy_loss, in_axes=(None,0,0,0))(prior_params, u, mu_posterior_marginal, Sigma_posterior_marginal).sum()
    loss_policy = 0.

    T_log_B = T * np.log(B)

    kl_qp /= h_RPM.size
    ce_qf /= h_RPM.size
    ce_qF /= h_RPM.size
    loss_policy /= h_RPM.size
    T_log_B /= h_RPM.size

    return loss_policy, kl_qp, ce_qf, ce_qF, T_log_B, mu_posterior

def train_step(params, opt_states, y, u_raw, key, options):

    rpm_opt_state, _, control_state, F_approx_opt_state = opt_states

    if options['embed_u']:

        u = control_state.apply_fn(params["u_emb_params"], u_raw)

    else:

        u = np.copy(u_raw)

    B, T, U = y.shape[0], y.shape[1], u.shape[-1]
    if options['f_time_dependent']:

        D = y.shape[2] - 1

    else:

        D = y.shape[2]

    prior_params = get_constrained_prior_params(params['prior_params'], D, U)

    if options['use_LDS_for_F_in_q']:

        RPM_norm = rpm_opt_state.apply_fn(params["rpm_params"], y)

        # latent_dims = 3 ######## TO CHANGE
        # u_dims = 1 ######## TO CHANGE
        # Q_lqr = np.eye(latent_dims) ######## TO CHANGE
        # R_lqr = np.eye(u_dims) * 1e-3 ######## TO CHANGE
        # x_goal = (np.linalg.solve(params['prior_params']["A"] - np.eye(latent_dims), params['prior_params']["B"])).squeeze()
        # x_goal /= np.linalg.norm(x_goal)
        # # x_goal *= p["goal_norm"] ######## don't make goal unit norm away from origin
        # # (u_eq, _, _, _) = np.linalg.lstsq(params['prior_params']["B"], (np.eye(latent_dims) - params['prior_params']["A"]) @ x_goal)

        # RPM_norm_goal = rpm_opt_state.apply_fn(params["rpm_params"], np.array([1., 0., 0.]))
        # delta_mu = x_goal - RPM_norm_goal['mu']
        # RPM_norm['mu'] = vmap(vmap(lambda mu, goal_mu: mu + goal_mu, in_axes=(0, None)), in_axes=(0, None))(RPM_norm['mu'], delta_mu)
        # RPM_norm['h'] = vmap(vmap(lambda J, h, goal_mu: h + J @ goal_mu, in_axes=(0, 0, None)), in_axes=(0, 0, None))(RPM_norm['J'], RPM_norm['h'], delta_mu)

        if options['explicitly_integrate_out_u']:

            prior_marginal = marginal_u_integrated_out(prior_params, T)

        else:

            prior_marginal = marginal(prior_params, D, T) # treat parameters of p(z'|z) as free (implicit) parameters of the q distribution

        RPM = {}
        RPM['J'] = prior_marginal['J'][None] + RPM_norm["J"]
        RPM['h'] = prior_marginal['h'][None] + RPM_norm["h"]
        RPM['Sigma'] = vmap(vmap(lambda S, I: psd_solve(S, I), in_axes=(0, None)), in_axes=(0, None))(RPM['J'], np.eye(D))
        RPM['mu'] = np.einsum("hijk,hik->hij", RPM['Sigma'], RPM['h'])

    elif options['use_MM_for_F_in_q']:

        RPM = rpm_opt_state.apply_fn(params["rpm_params"], y)

        MM = moment_match_RPM(RPM)
        RPM_norm = {}
        RPM_norm["J"] = RPM["J"] - MM["J"][None]
        RPM_norm["J"] = vmap(vmap(lambda J: np.diag(np.clip(np.diag(J), a_min=0, a_max=None))))(RPM_norm["J"])
        RPM_norm["h"] = RPM["h"] - MM["h"][None]

    elif options['use_GRU_for_F_in_q']:

        RPM_norm = rpm_opt_state.apply_fn(params["rpm_params"], y)

        prior_marginal = F_approx_opt_state.apply_fn(params["F_approx_params"], np.zeros((T,1)))

        RPM = {}
        RPM['J'] = prior_marginal['J'][None] + RPM_norm["J"]
        RPM['h'] = prior_marginal['h'][None] + RPM_norm["h"]
        RPM['Sigma'] = vmap(vmap(lambda S, I: psd_solve(S, I), in_axes=(0, None)), in_axes=(0, None))(RPM['J'], np.eye(D))
        RPM['mu'] = np.einsum("hijk,hik->hij", RPM['Sigma'], RPM['h'])

    prior, likelihood, posterior = vmap(update_prior, in_axes=(None,0,0,0))(prior_params, u, RPM_norm["J"], RPM_norm["h"])

    keys = random.split(key, B)
    loss_policy, kl_qp, ce_qf, ce_qF, T_log_B, mu_posterior = vmap(compute_free_energy, in_axes=(None,0,0,0,None,None,0,0,0,0))(prior_params, prior, RPM['J'], RPM['h'], RPM['mu'], RPM['Sigma'], posterior, u, keys, np.arange(B))

    if options['use_policy_loss']:

        free_energy = loss_policy - kl_qp - ce_qf + ce_qF - T_log_B

    else:
        
        free_energy = - kl_qp - ce_qf + ce_qF - T_log_B

    return -free_energy.mean(), (loss_policy, kl_qp, ce_qf, ce_qF, mu_posterior)

def get_train_state(ckpt_metrics_dir, all_models, all_optimisers=[], all_params=[]):

    options = CheckpointManagerOptions(max_to_keep=3, best_fn=lambda metrics: metrics, best_mode='min')
    mngr = CheckpointManager(ckpt_metrics_dir,  
                             {'rpm_model_state': AsyncCheckpointer(PyTreeCheckpointHandler()),
                              'prior_model_state': AsyncCheckpointer(PyTreeCheckpointHandler()),
                              'u_emb_model_state': AsyncCheckpointer(PyTreeCheckpointHandler()),
                              'F_approx_model_state': AsyncCheckpointer(PyTreeCheckpointHandler())},
                             options)

    states = []
    for params, optimiser, model in zip(all_params, all_optimisers, all_models):

        states.append(train_state.TrainState.create(apply_fn = model.apply, params = params, tx = optimiser))
        
    return states, mngr

def params_update(opt_states, grads):
    
    rpm_opt_state, prior_opt_state, control_state, F_approx_opt_state = opt_states

    rpm_opt_state = rpm_opt_state.apply_gradients(grads = grads["rpm_params"])

    prior_opt_state = prior_opt_state.apply_gradients(grads = grads["prior_params"])
    prior_opt_state.params["A"] = truncate_singular_values(prior_opt_state.params["A"])
    prior_opt_state.params["A_F"] = truncate_singular_values(prior_opt_state.params["A_F"])

    control_state = control_state.apply_gradients(grads = grads["u_emb_params"])

    F_approx_opt_state = F_approx_opt_state.apply_gradients(grads = grads["F_approx_params"])

    params = {}
    params["rpm_params"] = rpm_opt_state.params
    params["prior_params"] = prior_opt_state.params
    params["u_emb_params"] = control_state.params
    params["F_approx_params"] = F_approx_opt_state.params

    return params, (rpm_opt_state, prior_opt_state, control_state, F_approx_opt_state)

B = 100
T = 100
D = 2
U = 1
h_dim_rpm = 5
h_dim_u_emb = 5
carry_dim = 10
prior_lr = 1e-3
learning_rate = 1e-3
max_grad_norm = 10
n_epochs = 5000
log_every = 250

options = {}
options['initialise_LDS_M_step'] = True
options['normalise_y'] = True
options['embed_u'] = False
options['use_LDS_for_F_in_q'] = True
options['explicitly_integrate_out_u'] = True
options['use_MM_for_F_in_q'] = False
options['use_GRU_for_F_in_q'] = False
options['use_policy_loss'] = False
options['f_time_dependent'] = False
options['fit_LDS'] = True
options['save_dir'] = "/nfs/nhome/live/jheald/svae/my_code/runs"
options['project_name'] = 'RPM-mycode'

seed = 5
subkey1, subkey2, subkey3, subkey4, subkey5, subkey6, key = random.split(random.PRNGKey(seed), 7)

RPM = rpm_network(z_dim=D, h_dim=h_dim_rpm)
params = {}
if options['f_time_dependent']:
    params["rpm_params"] = RPM.init(y = np.ones((D+1,)), rngs = {'params': subkey1})
else:
    params["rpm_params"] = RPM.init(y = np.ones((D,)), rngs = {'params': subkey1})

# if options['initialise_LDS_M_step']:
#     params["prior_params"] = initialise_LDS_params_M_step(D, U, subkey2)
# else:
params["prior_params"] = initialise_LDS_params(D, U, subkey2, closed_form_M_Step = False)

u_emb = control_network(u_emb_dim=U, h_dim=h_dim_u_emb)
params["u_emb_params"] = u_emb.init(u = np.ones((U,)), rngs = {'params': subkey5})

F_approx = F_for_q(carry_dim=carry_dim, z_dim=D)
params["F_approx_params"] = F_approx.init(inputs = np.ones((T,1)), rngs = {'params': subkey6})

rpm_opt = opt.chain(opt.adam(learning_rate=learning_rate), opt.clip_by_global_norm(max_grad_norm))
prior_opt = opt.chain(opt.adam(learning_rate=prior_lr), opt.clip_by_global_norm(max_grad_norm))
u_emb_opt = opt.chain(opt.adam(learning_rate=learning_rate), opt.clip_by_global_norm(max_grad_norm))
F_approx_opt = opt.chain(opt.adam(learning_rate=learning_rate), opt.clip_by_global_norm(max_grad_norm))

all_optimisers = (rpm_opt, prior_opt, u_emb_opt, F_approx_opt)
all_params = (params["rpm_params"], params["prior_params"], params["u_emb_params"], params["F_approx_params"])
all_models = (RPM, RPM, u_emb, F_approx)
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

train_step_jit = jit(partial(value_and_grad(train_step, has_aux=True), options=options))

print("policy loss not implemented!")

if options['initialise_LDS_M_step']:
    params = initialise_LDS_params_M_step(params, opt_states, y, u, subkey2, options, closed_form_M_Step = False)

pbar = trange(n_epochs)
for itr in pbar:

    subkey, key = random.split(key, 2)

    if options['fit_LDS']:

        (loss, (loss_policy, kl_qp, ce_qf, ce_qF, mu_posterior)), grads = train_step_jit(params, opt_states, y, u, subkey)
        R2, _ = R2_inferred_vs_actual_z(mu_posterior, true_z)

    else:

        (loss, (loss_policy, kl_qp, ce_qf, ce_qF, mu_posterior)), grads = train_step_jit(params, opt_states, y[:B,:,:], u[:B,:,None], subkey)
        R2 = 0.

    params, opt_states = params_update(opt_states, grads)

    pbar.set_description("train loss: {:.3f}, policy loss: {:.3f}, kl_qp: {:.3f}, ce_qf: {:.3f}, ce_qF: {:.3f}, R2 train states: {:.3f}".format(loss, loss_policy.mean(), kl_qp.mean(), ce_qf.mean(), ce_qF.mean(), R2))

    if itr % log_every == 0:

        log_to_wandb(loss, kl_qp, ce_qf, ce_qF, mu_posterior.reshape(B,T,D), y, options)

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

from matplotlib import pyplot as plt


from matplotlib import pyplot as plt
# # plt.plot((true_prior_params['C'] @ true_z[0,:,:].T + true_prior_params['d'][None].T).T,'r')
plt.plot(y[0,:,:])
plt.show()
# breakpoint()