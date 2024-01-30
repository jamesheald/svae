import jax
from jax import jit, vmap
from jax.lax import scan
from jax import random as jr
from jax import numpy as np
from jax.tree_util import tree_map

import optax as opt
from copy import deepcopy

from functools import partial

from tqdm import trange

from time import time
import wandb, traceback
from pprint import pprint

from svae.logging import summarize_pendulum_run, predict_multiple, save_params_to_wandb, log_to_wandb, validation_log_to_wandb
from svae.posteriors import DKFPosterior, CDKFPosterior, LDSSVAEPosterior, PlaNetPosterior
from svae.priors import LinearGaussianChainPrior, LieParameterizedLinearGaussianChainPrior
from svae.networks import PlaNetRecognitionWrapper
from svae.utils import truncate_singular_values, get_train_state, R2_inferred_vs_actual_states
from svae.svae import DeepLDS

from flax.training.early_stopping import EarlyStopping
from flax.training.orbax_utils import save_args_from_target
from orbax.checkpoint import SaveArgs

from dynamax.utils.utils import psd_solve

from scipy.linalg import solve_discrete_are, solve
import copy

# @title Experiment scheduler
LINE_SEP = "#" * 42

def dict_len(d):
    if (type(d) == list):
        return len(d)
    else:
        return dict_len(d[list(d.keys())[0]])

def dict_map(d, func):
    if type(d) == list:
        return func(d)
    elif type(d) == dict:
        r = deepcopy(d)
        for key in d.keys():
            r[key] = dict_map(r[key], func)
            # Ignore all the Nones
            if r[key] is None:
                r.pop(key)
        if len(r.keys()) == 0:
            # There's no content
            return None
        else:
            return r
    else:
        return None

def dict_product(d1, d2):
    l1, l2 = dict_len(d1), dict_len(d2)
    def expand_list(d):
        result = []
        for item in d:
            result.append(item)
            result.extend([None] * (l2-1))
        return result
    def multiply_list(d):
        return d * l1
    result = dict_map(d1, expand_list)
    additions = dict_map(d2, multiply_list)
    return dict_update(result, additions)

def dict_get(d, id):
    return dict_map(d, lambda l: l[id])

def dict_update(d, u):
    if d is None:
        d = dict()
    for key in u.keys():
        if type(u[key]) == dict:
            d.update({
                key: dict_update(d.get(key), u[key])
            })
        else:
            d.update({key: u[key]})
    return d

# A standardized function that structures and schedules experiments
# Can chain multiple variations of experiment parameters together
def experiment_scheduler(run_params, dataset_getter, model_getter, train_func,
                         logger_func=None, err_logger_func=None,
                         run_variations=None, params_expander=None,
                         on_error=None, continue_on_error=True, use_wandb=True):
    """
    Arguments:
        run_params: dict{"dataset_params"}
            A large dictionary containing all relevant parameters to the run
        dataset_getter: run_params -> dict{"train_data", ["generative_model"]}
            A function that loads/samples a dataset
        model_getter: run_params, data_dict -> model
            A function that creates a model given parameters. Note that the model
            could depend on the specifics of the dataset/generative model as well
        train_func: model, data, run_params -> results
            A function that contains the training loop.
            TODO: later we might wanna open up this pipeline and customize further!
        (optional) logger_func: results, run_params -> ()
            A function that logs the current run.
        (optional) err_logger_func: message, run_params -> ()
            A function that is called when the run fails.
        (optional) run_variations: dict{}
            A nested dictionary where the leaves are lists of different parameters.
            None means no change from parameters of the last run.
        (optional) params_expander: dict{} -> dict{}
            Turns high level parameters into specific low level parameters.
    returns:
        all_results: List<result>
            A list containing results from all runs. Failed runs are indicated
            with a None value.
    """
    params_expander = params_expander or (lambda d: d)

    num_runs = dict_len(run_variations) if run_variations else 1
    params = deepcopy(run_params)
    print("Total number of runs: {}".format(num_runs))
    print("Base paramerters:")
    pprint(params)

    global data_dict
    all_results = []
    all_models = []

    def _single_run(data_out, model_out):
        print("Loading dataset!")
        data_dict = dataset_getter(curr_params)
        data_out.append(data_dict)
        # Make a new model
        model_dict = model_getter(curr_params, data_dict)
        model_out.append(model_dict)
        all_models.append(model_dict)
        results = train_func(model_dict, data_dict, curr_params)
        all_results.append(results)
        if logger_func:
            logger_func(results, curr_params, data_dict)

    for run in range(num_runs):
        print(LINE_SEP)
        print("Starting run #{}".format(run))
        print(LINE_SEP)
        curr_variation = dict_get(run_variations, run)
        if curr_variation is None:
            if (run != 0):
                print("Variation #{} is a duplicate, skipping run.".format(run))
                continue
            curr_params = params_expander(params)
        else:
            print("Current parameter variation:")
            pprint(curr_variation)
            curr_params = dict_update(params, curr_variation)
            curr_params = params_expander(curr_params)
            print("Current full parameters:")
            pprint(curr_params)
            if curr_variation.get("dataset_params"):
                reload_data = True
        # Hack to get the values even when they err out
        data_out = []
        model_out = []
        if not continue_on_error:
            _single_run(data_out, model_out)
        else:
            try:
                _single_run(data_out, model_out)
                if use_wandb: wandb.finish()
            except:
                all_results.append(None)
                if (on_error):
                    try:
                        on_error(data_out[0], model_out[0])
                    except:
                        pass # Oh well...
                print("Run errored out due to some the following reason:")
                traceback.print_exc()
                if use_wandb: wandb.finish(exit_code=1)
    return all_results, all_models

class Trainer:
    """
    model: a pytree node
    loss (key, params, model, data, **train_params) -> (loss, aux)
        Returns a loss (a single float) and an auxillary output (e.g. posterior)
    init (key, model, data, **train_params) -> (params, opts)
        Returns the initial parameters and optimizers to go with those parameters
    update (params, grads, opts, model, aux, **train_params) -> (params, opts)
        Returns updated parameters, optimizers
    """
    def __init__(self, model, 
                 train_params=None, 
                 init=None, 
                 loss=None, 
                 val_loss=None,
                 update=None,
                 initial_params=None):
        # Trainer state
        self.params = initial_params
        self.model = model
        self.past_params = []
        self.time_spent = []

        if train_params is None:
            train_params = dict()

        self.train_params = train_params

        if init is not None:
            self.init = init
        if loss is not None:
            self.loss = loss

        self.val_loss = val_loss or self.loss
        if update is not None: 
            self.update = update

    def train_step(self, key, params, data, target, u, opt_states, itr, goal_obs):
        model = self.model
        results = \
            jax.value_and_grad(
                lambda params: partial(self.loss, goal_obs=goal_obs, itr=itr, **self.train_params)\
                (key, model, data, target, u, params), has_aux=True)(params)
        (loss, aux), grads = results
        params, opts = self.update(params, grads, self.opts, opt_states, model, aux, **self.train_params)
        return params, opts, (loss, aux), grads

    def val_step(self, key, params, data, target, u, goal_obs):
        return partial(self.val_loss, goal_obs=goal_obs, **self.train_params)(key, self.model, data, target, u, params)

    def val_epoch(self, key, params, data, target, u, goal_obs, val_states):

        batch_size = self.train_params.get("val_batch_size") or data.shape[0]

        num_batches = data.shape[0] // batch_size

        loss_sum = 0
        aux_sum = None
        posterior_means = np.zeros(val_states.shape)

        for batch_id in range(num_batches):
            batch_start = batch_id * batch_size
            loss, aux = self.val_step_jitted(key, params, 
                                        data[batch_start:batch_start+batch_size], 
                                        target[batch_start:batch_start+batch_size],
                                        u[batch_start:batch_start+batch_size], goal_obs)

            posterior_means = posterior_means.at[batch_start:batch_start+batch_size].set(aux['posterior_means'])

            loss_sum += loss
            if aux_sum is None:
                aux_sum = aux
            else:
                aux_sum = jax.tree_map(lambda a,b: a+b, aux_sum, aux)
            key, _ = jr.split(key)
        
        loss_avg = loss_sum / num_batches
        aux_avg = jax.tree_map(lambda a: a / num_batches, aux_sum)

        # collapse trials and timepoints into one sequence
        val_states = val_states.reshape(-1, val_states.shape[-1])
        posterior_means = posterior_means.reshape(-1, posterior_means.shape[-1])

        aux_avg['R2_val_states'], predicted_states = R2_inferred_vs_actual_states(posterior_means, val_states)

        return loss_avg, aux_avg

    """
    Callback: a function that takes training iterations and relevant parameter
        And logs to WandB
    """
    def train(self, data_dict, run_type, max_iters, 
              callback=None, val_callback=None, 
              summary=None, key=None,
              min_delta=1e-3, patience=2,
              early_stop_start=10000, 
              max_lose_streak=2000):

        early_stop = EarlyStopping(min_delta=min_delta, patience=patience)

        if key is None:
            key = jr.PRNGKey(0)

        model = self.model
        train_data = data_dict["train_data"]
        train_targets = data_dict.get("train_targets")
        if (train_targets is None): train_targets = train_data
        train_u = data_dict["train_u"]
        train_states = data_dict["train_states"]
        val_data = data_dict.get("val_data")
        val_targets = data_dict.get("val_targets")
        if (val_targets is None): val_targets = val_targets = val_data
        val_u = data_dict["val_u"]
        val_states = data_dict["val_states"]
        batch_size = self.train_params.get("train_batch_size") or train_data.shape[0]
        num_batches = train_data.shape[0] // batch_size

        init_key, key = jr.split(key, 2)

        # Initialize optimizer
        self.params, self.opts, self.opt_states, self.mngr = self.init(init_key, model, 
                                                                  train_data[:batch_size],
                                                                  self.params,
                                                                  **self.train_params)

        self.train_losses = []
        self.test_losses = []
        self.val_losses = []
        self.past_params = []
        self.R2_train_states = []
        self.R2_train_states_rpm = []
        self.R2_val_states = []

        pbar = trange(max_iters)
        pbar.set_description("[jit compling...]")
        
        mask_start = self.train_params.get("mask_start")
        if (mask_start):
            mask_size = self.train_params["mask_size"]
            self.train_params["mask_size"] = 0

        train_step = jit(self.train_step)
        self.val_step_jitted = jit(self.val_step)

        best_loss = None
        best_itr = 0
        val_loss = None
        R2_val_states = None

        indices = np.arange(train_data.shape[0], dtype=int)

        # import gym
        # import numpy as onp
        # from jax.lax import scan
        # from svae.utils import lie_params_to_constrained, construct_dynamics_matrix, scale_matrix_by_norm
        # env = gym.make('Pendulum-v0')
        # def get_previous_P(carry, inputs):

        #     P, A, B, Q, R = carry

        #     prev_P = Q + A.T @ P @ A - (A.T @ P @ B) @ np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)

        #     carry = prev_P, A, B, Q, R
        #     outputs = None

        #     return carry, outputs

        # def get_optimal_feedback_gain(A, B, Q, R):

        #     carry = Q, A, B, Q, R
        #     (P, _, _, _, _), _ = scan(get_previous_P, carry, None, length=100)
        #     K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)

        #     return K

        # def filter_observation(models, params, m, P, y, delta_mu):

        #     rec = models.recognition.apply(params['rec_params'], y)
        #     e = rec['mu'] + delta_mu - m
        #     S = P + rec['Sigma']
        #     K = np.linalg.solve(S, P).T
        #     m += K @ e
        #     P = (np.eye(m.size) - K) @ P

        #     return m, P

        # def get_control(m, K):

        #     u = - K @ m

        #     return u

        # def predict_next_state(m, P, A, B, u, Q):

        #     m = A @ m + B @ u
        #     P = A @ P @ A.T + Q

        #     return m, P

        for itr in pbar:

            # n_rollouts = 1
            # z_dim = 3
            # n_timepoints = 100
            # m1 = self.params['prior_params']['m1']
            # Q1 = lie_params_to_constrained(self.params['prior_params']['Q1'], z_dim)
            # # Q1 = np.diag(np.exp(all_results[0][1]['prior_params']['Q1']))
            # A = construct_dynamics_matrix(self.params['prior_params']["A_u"], self.params['prior_params']["A_v"], self.params['prior_params']["A_s"], z_dim)
            # B = scale_matrix_by_norm(self.params['prior_params']['B'])
            # Q = lie_params_to_constrained(self.params['prior_params']['Q'], z_dim)
            # # Q = np.diag(np.exp(all_results[0][1]['prior_params']['Q']))

            # latent_dims = 3 ######## TO CHANGE
            # u_dims = 1 ######## TO CHANGE
            # Q_lqr = np.eye(latent_dims) ######## TO CHANGE
            # R_lqr = np.eye(u_dims) * 1e-3 ######## TO CHANGE
            # K = get_optimal_feedback_gain(A, B, Q_lqr, R_lqr)

            # # compute optimal feedback gain matrix K
            # prior_params = self.model.prior.get_constrained_params(self.params["prior_params"], np.empty((n_timepoints,1)))
            # p = copy.deepcopy(prior_params)
            # latent_dims = 3 ######## TO CHANGE
            # u_dims = 1 ######## TO CHANGE
            # Q_lqr = np.eye(latent_dims) ######## TO CHANGE
            # R_lqr = np.eye(u_dims) * 1e-3 ######## TO CHANGE
            # x_goal = (np.linalg.solve(p["A"] - np.eye(latent_dims), p["B"])).squeeze()
            # x_goal /= np.linalg.norm(x_goal)
            # x_goal *= p["goal_norm"] ######## don't make goal unit norm away from origin
            # (u_eq, _, _, _) = np.linalg.lstsq(p["B"], (np.eye(latent_dims) - p["A"]) @ x_goal)

            # # shift the mean/precision-weighted mean of all RPM potentials so that the mean of the inferred hidden state for the goal is at x_goal
            # RPM_goal = model.recognition.apply(self.params["rec_params"], data_dict["scaled_goal"])
            # delta_mu = x_goal - RPM_goal['mu']

            # m = onp.zeros((n_rollouts, z_dim, n_timepoints))
            # P = onp.zeros((n_rollouts, z_dim, z_dim, n_timepoints))
            # u = onp.zeros((n_rollouts, n_timepoints, 1))
            # rew = onp.zeros((n_rollouts, n_timepoints))
            # m[:, :, 0] = m1[None].repeat(n_rollouts, axis = 0)
            # P[:, :, :, 0] = Q1[None].repeat(n_rollouts, axis = 0)
            # for r in range(n_rollouts):
            #     obs = env.reset()
            #     for t in range(n_timepoints - 1):
            #         obs = data_dict['scaler_obs'].transform(obs[None]).squeeze()
            #         m[r, :, t], P[r, :, :, t] = filter_observation(self.model, self.params, m[r, :, t], P[r, :, :, t], obs, delta_mu)
            #         u[r, t, :] = get_control(m[r, :, t], K)
            #         m[r, :, t + 1], P[r, :, :, t + 1] = predict_next_state(m[r, :, t], P[r, :, :, t], A, B, u[r, t], Q)
            #         action = data_dict['scaler_u'].inverse_transform(u[r, t, :][None])[:, 0]
            #         obs, reward, done, info = env.step(action)
            #         rew[r, t] = reward
            #     print("cumulative reward", rew[r, :].sum())

            train_key, val_key, key = jr.split(key, 3)

            batch_id = itr % num_batches
            batch_start = batch_id * batch_size
            epoch = itr // num_batches

            # Uncomment this to time the execution
            # t = time()
            # Training step
            # ----------------------------------------
            batch_indices = indices[batch_start:batch_start+batch_size]

            step_results = train_step(train_key, self.params, 
                           train_data[batch_indices],
                           train_targets[batch_indices], 
                           train_u[batch_indices], 
                           self.opt_states, itr, data_dict["scaled_goal"])
            self.params, self.opt_states, loss_out, grads = step_results#\
                # jax.tree_map(lambda x: x.block_until_ready(), step_results)
            # ----------------------------------------
            # dt = time() - t
            # self.time_spent.append(dt)

            loss, aux = loss_out

            # collapse trials and timepoints into one sequence

            train_states_batch = train_states[batch_indices].reshape(-1, train_states[batch_indices].shape[-1])
            posterior_means = aux['posterior_means'].reshape(-1, aux['posterior_means'].shape[-1])
            aux['R2_train_states'], predicted_states = R2_inferred_vs_actual_states(posterior_means, train_states_batch)
            self.R2_train_states.append(aux["R2_train_states"])
            for idim in range(3):
                str_i = '_dim' + str(idim)
                aux['R2_train_states' + str_i], _ = R2_inferred_vs_actual_states(posterior_means, train_states_batch[:,idim])

            rpm_means = aux['rpm_means'].reshape(-1, aux['rpm_means'].shape[-1])
            aux['R2_train_states_rpm'], predicted_states_rpm = R2_inferred_vs_actual_states(rpm_means, train_states_batch)
            self.R2_train_states_rpm.append(aux["R2_train_states_rpm"])
            for idim in range(3):
                str_i = '_dim' + str(idim)
                aux['R2_train_states_rpm' + str_i], _ = R2_inferred_vs_actual_states(rpm_means, train_states_batch[:,idim])

            # cov_diag = vmap(vmap(lambda x: np.diag(x)))(aux['cov'][:,0,:,:])
            # min_cov_idx = np.unravel_index(cov_diag.argmin(), cov_diag.shape)
            # print("min cov", cov_diag.min())
            # print("min cov episode", min_cov_idx[0])
            # print("min cov time point", min_cov_idx[1])
            # print("min cov dim", min_cov_idx[2])
            # print("kl mean", aux['kl'].mean())
            # print("min kl", aux['kl'].min())
            # print("min kl episode", aux['kl'].argmin())
            # max_ells = np.unravel_index(aux['ells'][:, 0, :].argmax(), aux['ells'].shape)
            # print("max ells", aux['ells'][:, 0, :].max())
            # print("max ells episode", max_ells[0])
            # print("max ells time point", max_ells[1])

            print("policy_loss mean", aux['policy_loss'].mean())

            # print("kl_correction mean", aux['kl_correction'].mean())
            # print("log_normaliser_implied mean", aux['log_normaliser_implied'].mean())
            # print("log_normaliser_implied2 mean", aux['log_normaliser_implied2'].mean())

            # print("kl_qp mean", aux['kl_qp'].mean())
            # print("kl_qf mean", aux['kl_qf'].mean())
            # print("log_Gamma mean", aux['log_Gamma'].mean())
            # print("E_log_aux mean", aux['E_log_aux'].mean())
            # print("E_log_aux1 mean", aux['E_log_aux1'].mean())
            # print("E_log_aux2 mean", aux['E_log_aux2'].mean())
            # print("E_log_aux3 mean", aux['E_log_aux3'].mean())

            # print("A", self.params['prior_params']['A'])
            # print("B", self.params['prior_params']['B'])
            # print("Q", self.params['prior_params']['Q'])
            # print("Q1", self.params['prior_params']['Q1'])
            # print("m1", self.params['prior_params']['m1'])

            self.train_losses.append(loss)
            if epoch == 0:
                pbar.set_description("train loss: {:.3f}, kl_qp: {:.3f}, kl_qf: {:.3f}, log_Gamma: {:.3f}, R2 train states: {:.3f}, R2 train states rpm: {:.3f}".format(loss, aux['kl_qp'].mean(), aux['kl_qf'].mean(), aux['log_Gamma'].mean(), self.R2_train_states[-1], self.R2_train_states_rpm[-1]))
            else:
                pbar.set_description("train loss: {:.3f}, kl_qp: {:.3f}, kl_qf: {:.3f}, log_Gamma: {:.3f}, R2 train states: {:.3f}, R2 train states rpm: {:.3f}, val loss: {:.3f}, R2 val states: {:.3f}, best metric: {:.3f}, has improved: {}".format(loss, aux['kl_qp'].mean(), aux['kl_qf'].mean(), aux['log_Gamma'].mean(), self.R2_train_states[-1], self.R2_train_states_rpm[-1], val_loss, R2_val_states, early_stop.best_metric, early_stop.has_improved))
            if (callback) and epoch % self.train_params.get("log_every_n_epochs") == 0: 
                callback(self, (loss, aux, predicted_states, predicted_states_rpm, train_states_batch), data_dict, grads)

            # evaluate the model on validation data to either log progress or (potentially) perform a checkpoint
            if batch_id == num_batches - 1 and (epoch % self.train_params.get("checkpoint_every_n_epochs") == 0 or epoch % self.train_params.get("log_every_n_epochs") == 0):
                # We're at the end of an epoch
                # We could randomly shuffle the data
                indices = jr.permutation(key, indices)
                if (self.train_params.get("use_validation")):
                    val_loss_out = self.val_epoch(val_key, self.params, val_data, val_targets, val_u, data_dict["scaled_goal"], val_states)
                    if (val_callback) and epoch % self.train_params.get("log_every_n_epochs") == 0: 
                        val_callback(self, val_loss_out, data_dict)
                    val_loss, val_aux = val_loss_out
                    R2_val_states = val_aux['R2_val_states']
                    self.R2_val_states.append(R2_val_states)
                    self.val_losses.append(val_loss)

                    if epoch % self.train_params.get("checkpoint_every_n_epochs") == 0:

                        if self.train_params.get("inference_method") != "rpm":
                            self.mngr.save(epoch,
                                           items={'recognition_model_state': self.opt_states[0],
                                                  'decoder_model_state': self.opt_states[1],
                                                  'prior_model_state': self.opt_states[2]},
                                           # save_kwargs={'recognition_model_state': {'save_args': tree_map(lambda _: SaveArgs(), self.opt_states[0])},
                                           # 'decoder_model_state': {'save_args': tree_map(lambda _: SaveArgs(), self.opt_states[1])},
                                           # 'prior_model_state': {'save_args': tree_map(lambda _: SaveArgs(), self.opt_states[2])}
                                           # },
                                           metrics=float(val_loss))
                        else:
                            self.mngr.save(epoch,
                                           items={'recognition_model_state': self.opt_states[0],
                                                  'prior_model_state': self.opt_states[1],
                                                  'delta_q_state': self.opt_states[2],
                                                  'delta_f_tilde_state': self.opt_states[3],
                                                  },
                                           # save_kwargs={'recognition_model_state': {'save_args': tree_map(lambda _: SaveArgs(), self.opt_states[0])},
                                           # 'decoder_model_state': {'save_args': tree_map(lambda _: SaveArgs(), self.opt_states[1])},
                                           # 'prior_model_state': {'save_args': tree_map(lambda _: SaveArgs(), self.opt_states[2])}
                                           # },
                                           metrics=float(val_loss))

                        # only query early_stop at a checkpoint
                        if itr > self.train_params.get("early_stop_start"):
                            early_stop = early_stop.update(val_loss)
                            if early_stop.should_stop:
                                # if early stopping criteria met, break
                                print('Early stopping criteria met, breaking...')
                                
                                break

                                                
            # if not self.train_params.get("use_validation") or val_loss is None:
            #     curr_loss = loss
            # else:
            #     curr_loss = val_loss

            # if itr >= early_stop_start:
            #     if best_loss is None or curr_loss < best_loss:
            #         best_itr = itr
            #         best_loss = curr_loss
                # if curr_loss > best_loss and itr - best_itr > max_lose_streak:
                #     print("Early stopping!")
                #     break

            # # Record parameters
            # record_params = self.train_params.get("record_params")
            # if record_params and record_params(itr):
            #     curr_params = deepcopy(self.params)
            #     curr_params["iteration"] = itr
            #     self.past_params.append(curr_params)

            if (mask_start and itr == mask_start):
                self.train_params["mask_size"] = mask_size
                train_step = jit(self.train_step)
                val_step = jit(self.val_step)

        if summary:
            summary(self, data_dict)

        self.mngr.wait_until_finished()

def svae_init(key, model, data, initial_params=None, **train_params):
    init_params = model.init(key)
    if (initial_params): init_params.update(initial_params)
    
    if (train_params["inference_method"] == "planet"):
        init_params["rec_params"] = {
            "rec_params": init_params["rec_params"],
            "post_params": init_params["post_params"]["network_params"]
        }
    # Expand the posterior parameters by batch size
    init_params["post_params"] = vmap(lambda _: init_params["post_params"])(data)
    init_params["post_samples"] = np.zeros((data.shape[0], 
                                            train_params.get("obj_samples") or 1) 
                                             + model.posterior.shape)
    # If we are in VAE mode, set the dynamics matrix to be 0
    if (train_params.get("run_type") == "vae_baseline"):
        A = init_params["prior_params"]["A"]
        init_params["prior_params"]["A"] = np.zeros_like(A)

    learning_rate = train_params["learning_rate"]
    rec_opt = opt.chain(opt.adamw(learning_rate=learning_rate, weight_decay=train_params.get("weight_decay")), opt.clip_by_global_norm(train_params.get("max_grad_norm")))
    # rec_opt_state = rec_opt.init(init_params["rec_params"])
    # rec_opt_state, rec_opt_ckptr =  get_train_state(rec_opt, model, init_params["rec_params"], train_params, "recognition_model")
    if train_params["inference_method"] != "rpm":
        dec_opt = opt.chain(opt.adamw(learning_rate=learning_rate, weight_decay=train_params.get("weight_decay")), opt.clip_by_global_norm(train_params.get("max_grad_norm")))
        # dec_opt_state = dec_opt.init(init_params["dec_params"])
        # dec_opt_state, dec_opt_ckptr =  get_train_state(dec_opt, model, init_params["dec_params"], train_params, "decoder_model")

    if (train_params.get("use_natural_grad")):
        prior_lr = None
        prior_opt = None
        prior_opt_state = None
    else:
        # Add the option of using an gradient optimizer for prior parameters
        prior_lr = train_params.get("prior_learning_rate") or learning_rate
        prior_opt = opt.chain(opt.adamw(learning_rate=prior_lr, weight_decay=train_params.get("weight_decay")), opt.clip_by_global_norm(train_params.get("max_grad_norm")))
        # prior_opt_state = prior_opt.init(init_params["prior_params"])
        # prior_opt_state, prior_opt_ckptr =  get_train_state(prior_opt, model, init_params["prior_params"], train_params, "prior_model")

    if train_params["zero_last_layers"]:
        
        init_params["delta_q_params"]['params']['dense']['kernel'] *= 0.
        init_params["delta_q_params"]['params']['dense']['bias'] *= 0.

        init_params["delta_f_tilde_params"]['params']['head_mean_fn']['Dense_0']['kernel'] *= 0.
        init_params["delta_f_tilde_params"]['params']['head_mean_fn']['Dense_0']['bias'] *= 0.
        init_params["delta_f_tilde_params"]['params']['head_log_var_fn']['Dense_0']['kernel'] *= 0.
        init_params["delta_f_tilde_params"]['params']['head_log_var_fn']['Dense_0']['bias'] *= 0.

    learning_rate = train_params["learning_rate"]
    delta_q_opt = opt.chain(opt.adamw(learning_rate=learning_rate, weight_decay=train_params.get("weight_decay")), opt.clip_by_global_norm(train_params.get("max_grad_norm")))
    
    learning_rate = train_params["delta_nat_f_tilde_lr"]
    delta_f_tilde_opt = opt.chain(opt.adamw(learning_rate=learning_rate, weight_decay=train_params.get("weight_decay")), opt.clip_by_global_norm(train_params.get("max_grad_norm")))

    if train_params["inference_method"] == "rpm" or train_params["inference_method"] == "lds":
        all_optimisers = (rec_opt, prior_opt, delta_q_opt, delta_f_tilde_opt)
        all_params = (init_params["rec_params"], init_params["prior_params"], init_params["delta_q_params"], init_params["delta_f_tilde_params"])
        all_opt_states, mngr = get_train_state(train_params, all_optimisers, all_params)
    else:
        all_optimisers = (rec_opt, dec_opt, prior_opt)
        all_params = (init_params["rec_params"], init_params["dec_params"], init_params["prior_params"])
        all_opt_states, mngr = get_train_state(train_params, all_optimisers, all_params)


    return (init_params, 
            all_optimisers, 
            all_opt_states,
            mngr)
    
def svae_loss(key, model, data_batch, target_batch, u_batch, model_params, goal_obs, itr=0, **train_params):
    batch_size = data_batch.shape[0]
    n_timepoints = data_batch.shape[1]
    # Axes specification for vmap
    # We're just going to ignore this for now
    RPM_batch = model.recognition.apply(model_params["rec_params"], data_batch)

    # compute optimal feedback gain matrix K
    prior_params = model.prior.get_constrained_params(model_params["prior_params"], np.empty((n_timepoints,1)))
    p = copy.deepcopy(prior_params)
    latent_dims = 3 ######## TO CHANGE
    u_dims = 1 ######## TO CHANGE
    Q_lqr = np.eye(latent_dims) ######## TO CHANGE
    R_lqr = np.eye(u_dims) * 1e-3 ######## TO CHANGE
    x_goal = (np.linalg.solve(p["A"] - np.eye(latent_dims), p["B"])).squeeze()
    x_goal /= np.linalg.norm(x_goal)
    x_goal *= p["goal_norm"] ######## don't make goal unit norm away from origin
    (u_eq, _, _, _) = np.linalg.lstsq(p["B"], (np.eye(latent_dims) - p["A"]) @ x_goal)

    # shift the mean/precision-weighted mean of all RPM potentials so that the mean of the inferred hidden state for the goal is at x_goal
    # RPM_goal = model.recognition.apply(model_params["rec_params"], goal_obs)
    # delta_mu = x_goal - RPM_goal['mu']
    # RPM_batch['mu'] = vmap(vmap(lambda mu, delta_mu: mu + delta_mu, in_axes=(0, None)), in_axes=(0, None))(RPM_batch['mu'], delta_mu)
    # RPM_batch['h'] = vmap(vmap(lambda J, h, delta_mu: h + J @ delta_mu, in_axes=(0, 0, None)), in_axes=(0, 0, None))(RPM_batch['J'], RPM_batch['h'], delta_mu)

    # moment matched approximation to F
    # https://math.stackexchange.com/questions/195911/calculation-of-the-covariance-of-gaussian-mixtures
    # MM_prior = {}
    # MM_prior['mu'] = RPM_batch['mu'].mean(axis = 0)
    # mu_diff = RPM_batch['mu'] - MM_prior['mu'][None]
    # MM_prior['Sigma'] = RPM_batch['Sigma'].mean(axis = 0) + np.einsum("ijk,ijl->ijkl", mu_diff, mu_diff).mean(axis = 0)
    # MM_prior['J'] = vmap(lambda S, I: psd_solve(S, I), in_axes=(0, None))(MM_prior['Sigma'], np.eye(MM_prior['mu'].shape[-1]))
    # MM_prior['h'] = np.einsum("ijk,ik->ij", MM_prior['J'], MM_prior['mu'])

    # # x_goal = model.recognition.apply(model_params["rec_params"], np.array([1., 0., 0.])) # cos(theta), sin(theta), theta_dot (theta = 0 is upright)
    # prior_params = model.prior.get_constrained_params(model_params["prior_params"], np.empty((n_timepoints,1)))
    # # optimal_prior_params = model.prior.get_marginals_under_optimal_control(prior_params, x_goal['mu'])
    # K = model.prior.get_optimal_feedback_gain(p["A"], p["B"], Q_lqr, R_lqr)
    # optimal_prior_params = model.prior.get_marginals_under_optimal_control(prior_params, x_goal, u_eq, K)

    # prior_params_batch = vmap(model.prior.get_constrained_params, in_axes=(None,0))(model_params["prior_params"], u_batch)

    prior_marg_params = model.prior.get_constrained_params(model_params["prior_params"], None)

    # if using nonparametric delta_f_tilde
    # delta_f_tilde = model.delta_nat_f_tilde.apply(model_params["delta_f_tilde_params"])

    result = vmap(partial(model.compute_objective, **train_params), 
                  in_axes=(0, 0, 0, 0, 0, None, None, None))\
                  (jr.split(key, batch_size), data_batch, target_batch, u_batch, np.arange(batch_size), prior_marg_params, RPM_batch, model_params)
    # Need to compute sufficient stats if we want the natural gradient update
    if (train_params.get("use_natural_grad")):
        post_params = result["posterior_params"]
        post_samples = result["posterior_samples"]
        post_suff_stats = vmap(model.posterior.sufficient_statistics)(post_params)
        expected_post_suff_stats = tree_map(
            lambda l: np.mean(l,axis=0), post_suff_stats)
        result["sufficient_statistics"] = expected_post_suff_stats

    if train_params.get("inference_method") != "rpm" and train_params.get("inference_method") != "lds":
        # objs = result["objective"]
        if (train_params.get("beta") is None): # only apply beta < 1 during training (no longer applies as i pass train_params during val too)
            beta = 1
        else:
            beta = train_params["beta"](itr)
        objs = result["ell"] - beta * result["kl"]
    else:
        objs = result["free_energy"]

    # result['delta_mu'] = delta_mu

    return -np.mean(objs), result

def predict_forward(x, A, b, T):
    def _step(carry, t):
        carry = A @ carry + b
        return carry, carry
    return scan(_step, x, np.arange(T))[1]

def svae_pendulum_val_loss(key, model, data_batch, target_batch, model_params, **train_params):  
    N, T = data_batch.shape[:2]
    # We only care about the first 100 timesteps
    T = T // 2
    D = model.prior.latent_dims

    # obs_data, pred_data = data_batch[:,:T//2], data_batch[:,T//2:]
    obs_data = data_batch[:,:T]
    obj, out_dict = svae_loss(key, model, obs_data, obs_data, model_params, **train_params)
    # Compute the prediction accuracy
    prior_params = model_params["prior_params"] 
    # Instead of this, we want to evaluate the expected log likelihood of the future observations
    # under the posterior given the current set of observations
    # So E_{q(x'|y)}[p(y'|x')] where the primes represent the future
    post_params = out_dict["posterior_params"]
    horizon = train_params["prediction_horizon"] or 5

    _, _, _, pred_lls = vmap(predict_multiple, in_axes=(None, None, None, 0, None, 0, None))\
        (train_params, model_params, model, obs_data, T-horizon, jr.split(key, N), 10)
    # pred_lls = vmap(_prediction_lls)(np.arange(N), jr.split(key, N))
    out_dict["prediction_ll"] = pred_lls
    return obj, out_dict

def svae_update(params, grads, opts, opt_states, model, aux, **train_params):
    if train_params["inference_method"] == "rpm" or train_params["inference_method"] == "lds":
        rec_opt, prior_opt, delta_q_opt, delta_f_tilde_opt = opts
        rec_opt_state, prior_opt_state, delta_q_opt_state, delta_f_tilde_opt_state = opt_states
        rec_grad = grads["rec_params"]
    else:
        rec_opt, dec_opt, prior_opt = opts
        rec_opt_state, dec_opt_state, prior_opt_state = opt_states
        rec_grad, dec_grad = grads["rec_params"], grads["dec_params"]
    # updates, rec_opt_state = rec_opt.update(rec_grad, rec_opt_state, params["rec_params"])
    # params["rec_params"] = opt.apply_updates(params["rec_params"], updates)
    rec_opt_state = rec_opt_state.apply_gradients(grads = rec_grad)
    params["rec_params"] = rec_opt_state.params
    params["post_params"] = aux["posterior_params"]
    # params["post_samples"] = aux["posterior_samples"]
    if train_params["run_type"] == "model_learning":
        if train_params["inference_method"] != "rpm" and train_params["inference_method"] != "lds":
            # Update decoder
            # updates, dec_opt_state = dec_opt.update(dec_grad, dec_opt_state, params["dec_params"])
            # params["dec_params"] = opt.apply_updates(params["dec_params"], updates)
            dec_opt_state = dec_opt_state.apply_gradients(grads = dec_grad)
            params["dec_params"] = dec_opt_state.params

        # Update prior parameters
        if (train_params.get("use_natural_grad")):
            # Here we interpolate the sufficient statistics instead of the parameters
            suff_stats = aux["sufficient_statistics"]
            lr = params.get("prior_learning_rate") or 1
            avg_suff_stats = params["prior_params"]["avg_suff_stats"]
            # Interpolate the sufficient statistics
            params["prior_params"]["avg_suff_stats"] = tree_map(lambda x,y : (1 - lr) * x + lr * y, 
                avg_suff_stats, suff_stats)
            params["prior_params"] = model.prior.m_step(params["prior_params"])
        else:
            # updates, prior_opt_state = prior_opt.update(grads["prior_params"], prior_opt_state, params["prior_params"])
            # params["prior_params"] = opt.apply_updates(params["prior_params"], updates)
            prior_opt_state = prior_opt_state.apply_gradients(grads = grads["prior_params"])
            params["prior_params"] = prior_opt_state.params
            # params["prior_params"]["A"] = truncate_singular_values(params["prior_params"]["A"])
    
    if (train_params.get("run_type") == "vae_baseline"):
        # Zero out the updated dynamics matrix
        params["prior_params"]["A"] = np.zeros_like(params["prior_params"]["A"])
    # else:
    #     if (train_params.get("constrain_dynamics")):
    #         # Scale A so that its maximum singular value does not exceed 1
    #         params["prior_params"]["A"] = truncate_singular_values(params["prior_params"]["A"])
    #         # params["prior_params"]["A"] = scale_singular_values(params["prior_params"]["A"]):

    # # params["prior_params"]["A"] = aux["posterior_expected_states_next_states"].transpose(0,1,3,2).sum(axis=(0,1)) @ inv(aux["posterior_expected_states_squared"])
    # params["prior_params"]["A"] = psd_solve(aux["posterior_expected_states_squared"][:,:-1,:,:].sum(axis=(0,1)), aux["posterior_expected_states_next_states"].sum(axis=(0,1))).T

    # batch_size = aux["posterior_expected_states_squared"].shape[0]
    # n_timesteps = aux["posterior_expected_states_squared"].shape[1]
    # n_dims = aux["posterior_expected_states_squared"].shape[-1]
    # params["prior_params"]["Q"] = (aux["posterior_expected_states_squared"][:,1:,:,:].sum(axis=(0,1)) - params["prior_params"]["A"] @ aux["posterior_expected_states_next_states"].sum(axis=(0,1))) / (batch_size * (n_timesteps - 1))

    # params["prior_params"]["m1"] = aux['posterior_means'][:,0,:].mean(axis=0)
    # # params["prior_params"]["Q1"] = aux["posterior_expected_states_squared"][:,0,:,:] - np.outer(aux['posterior_means'][:,0,:], params["prior_params"]["m1"]) + vmap(lambda x, xbar: np.outer(x-xbar,x-xbar), in_axes=(0,None))(aux['posterior_means'][:,0,:], params["prior_params"]["m1"]).mean(axis=0)
    # params["prior_params"]["Q1"] = (aux["posterior_expected_states_squared"][:,0,:,:] - vmap(lambda x: np.outer(x,x))(aux['posterior_means'][:,0,:])).mean(axis=0) + vmap(lambda x, xbar: np.outer(x-xbar,x-xbar), in_axes=(0,None))(aux['posterior_means'][:,0,:], params["prior_params"]["m1"]).mean(axis=0)

    # # for numerical stability
    # params["prior_params"]["A"] = truncate_singular_values(params["prior_params"]["A"])
    # params["prior_params"]["Q"] = (params["prior_params"]["Q"] + params["prior_params"]["Q"].T)/2
    # params["prior_params"]["Q"] += np.eye(n_dims) * 1e-4
    # params["prior_params"]["Q1"] = (params["prior_params"]["Q1"] + params["prior_params"]["Q1"].T)/2
    # params["prior_params"]["Q1"] += np.eye(n_dims) * 1e-4

    # if params['use_delta_nat_q']:

    delta_q_opt_state = delta_q_opt_state.apply_gradients(grads = grads["delta_q_params"])
    params["delta_q_params"] = delta_q_opt_state.params

    # if params['use_delta_nat_f_tilde']:

    delta_f_tilde_opt_state = delta_f_tilde_opt_state.apply_gradients(grads = grads["delta_f_tilde_params"])
    params["delta_f_tilde_params"] = delta_f_tilde_opt_state.params

    if train_params["inference_method"] != "rpm" and train_params["inference_method"] != "lds":
        return params, (rec_opt_state, dec_opt_state, prior_opt_state)
    else:
        return params, (rec_opt_state, prior_opt_state, delta_q_opt_state, delta_f_tilde_opt_state)

def init_model(run_params, data_dict):
    p = deepcopy(run_params)
    d = p["dataset_params"]
    latent_dims = p["latent_dims"]
    input_shape = data_dict["train_data"].shape[1:]
    num_timesteps = input_shape[0]
    data = data_dict["train_data"]
    seed = p["seed"]
    seed_model, seed_elbo, seed_ems, seed_rec = jr.split(seed, 4)

    run_type = p["run_type"]
    recnet_class = globals()[p["recnet_class"]]
    decnet_class = globals()[p["decnet_class"]]

    if p["inference_method"] == "dkf":
        posterior = DKFPosterior(latent_dims, num_timesteps)
    elif p["inference_method"] in ["cdkf", "conv"]:
        posterior = CDKFPosterior(latent_dims, num_timesteps)
    elif p["inference_method"] == "planet":
        posterior = PlaNetPosterior(p["posterior_architecture"],
                                    latent_dims, num_timesteps)
    elif p["inference_method"] == "svae":
        # The parallel Kalman stuff only applies to SVAE
        # Since RNN based methods are inherently sequential
        posterior = LDSSVAEPosterior(latent_dims, num_timesteps, 
                                     use_parallel=p.get("use_parallel_kf"))
        
    rec_net = recnet_class.from_params(**p["recnet_architecture"])
    dec_net = decnet_class.from_params(**p["decnet_architecture"])
    if p["inference_method"] == "planet":
        # Wrap the recognition network
        rec_net = PlaNetRecognitionWrapper(rec_net)

    if (p.get("use_natural_grad")):
        prior = LinearGaussianChainPrior(latent_dims, num_timesteps)
    else:
        prior = LieParameterizedLinearGaussianChainPrior(latent_dims, num_timesteps, 
                    init_dynamics_noise_scale=p.get("init_dynamics_noise_scale") or 1)

    model = DeepLDS(
        recognition=rec_net,
        decoder=dec_net,
        prior=prior,
        posterior=posterior,
        input_dummy=np.zeros(input_shape),
        latent_dummy=np.zeros((num_timesteps, latent_dims))
    )

    initial_params = None
    svae_val_loss = svae_pendulum_val_loss if run_params["dataset"] == "pendulum" else svae_loss

    # Define the trainer object here
    trainer = Trainer(model, train_params=run_params, init=svae_init, 
                      loss=svae_loss, 
                      val_loss=svae_val_loss, 
                      update=svae_update, initial_params=initial_params)

    return {
        # We don't actually need to include model here
        # 'cause it's included in the trainer object
        "model": model,
        # "emission_params": emission_params
        "trainer": trainer
    }

def start_trainer(model_dict, data_dict, run_params):
    trainer = model_dict["trainer"]
    if run_params.get("log_to_wandb"):
        if run_params["dataset"] == "pendulum":
            summary = summarize_pendulum_run
        else:
            summary = save_params_to_wandb
    else:
        summary = None
    trainer.train(data_dict,
                  max_iters=run_params["max_iters"],
                  key=run_params["seed"],
                  callback=log_to_wandb, val_callback=validation_log_to_wandb,
                  summary=summary)
    return (trainer.model, trainer.params, trainer.train_losses)