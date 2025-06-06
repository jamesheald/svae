import jax
import jax.numpy as np
import jax.random as jr
from jax.numpy.linalg import eigh, cholesky, svd, inv, solve
from jax.scipy.linalg import solve_triangular
from jax import scipy

import matplotlib.pyplot as plt

from flax.linen import softplus, sigmoid

from flax.training.orbax_utils import restore_args_from_target
from flax.training import train_state
from orbax.checkpoint import AsyncCheckpointer, Checkpointer, PyTreeCheckpointHandler, CheckpointManager, CheckpointManagerOptions
from flax.training.early_stopping import EarlyStopping

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

from sklearn import linear_model

def plot_img_grid(recon):
    plt.figure(figsize=(8,8))
    # Show the sequence as a block of images
    stacked = recon.reshape(10, 24 * 10, 24)
    imgrid = stacked.swapaxes(0, 1).reshape(24 * 10, 24 * 10)
    plt.imshow(imgrid, vmin=0, vmax=1)

def R2_inferred_vs_actual_states(posterior_means, true_states):

    # import numpy as np
    # from sklearn.linear_model import LinearRegression
    # X = np.random.rand(100, 3)
    # y = X @ np.random.rand(3, 3)  + 3
    # reg = LinearRegression().fit(X, y)
    # print(reg.predict(X) - (X @ reg.coef_.T + reg.intercept_))

    reg = linear_model.LinearRegression()
    reg.fit(posterior_means, true_states) # reg.predict(posterior_means) = posterior_means @ reg.coef_.T + reg.intercept_
    R2 = reg.score(posterior_means, true_states)
    predicted_states = reg.predict(posterior_means)

    return R2, predicted_states

# @title Math helpers
def softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)

def inv_softplus(x, eps=1e-4):
    return np.log(np.exp(x - eps) - 1)

def vectorize_pytree(*args):
    """
    Flatten an arbitrary PyTree into a vector.
    :param args:
    :return:
    """
    flat_tree, _ = jax.tree_util.tree_flatten(args)
    flat_vs = [x.flatten() for x in flat_tree]
    return np.concatenate(flat_vs, axis=0)

def sample_from_MVN(mu, Sigma, key):

    sqrt_Sigma = np.linalg.cholesky(Sigma)
    x = sqrt_Sigma @ jr.normal(key, mu.shape) + mu

    return x

def construct_covariance_matrix(x, dim, eps=1e-4):

    # create lower triangular matrix
    L = np.zeros((dim, dim))
    L = L.at[np.tril_indices(dim)].set(x)

    # construct covariance matrix via its cholesky decomposition
    Sigma = L @ L.T

    # add scaled identity matrix for stability
    Sigma += eps * np.eye(dim)

    return Sigma

def construct_precision_matrix(x, dim, eps=.0):

    # create lower triangular matrix
    L = np.zeros((dim, dim))
    L = L.at[np.tril_indices(dim)].set(x)

    # construct covariance matrix via its cholesky decomposition
    Sigma = L @ L.T

    # add scaled identity matrix for stability
    Sigma += eps * np.eye(dim)

    return Sigma

# converts an (n(n+1)/2,) vector of Lie parameters
# to an (n, n) matrix
def lie_params_to_constrained(out_flat, dim, eps=1e-4):
    D, A = out_flat[:dim], out_flat[dim:]
    # ATTENTION: we changed this!
    # D = np.maximum(softplus(D), eps)
    D = softplus(D) + eps
    # Build a skew-symmetric matrix
    S = np.zeros((dim, dim))
    i1, i2 = np.tril_indices(dim - 1)
    S = S.at[i1+1, i2].set(A)
    S = S.T
    S = S.at[i1+1, i2].set(-A)

    O = scipy.linalg.expm(S)
    J = O.T @ np.diag(D) @ O
    return J

def scale_singular_values(A):
    _, s, _ = svd(A)
    return A / (np.maximum(1, np.max(s)))

def truncate_singular_values(A):
    eps = 1e-3
    # u, s, vt = svd(A) NotImplementedError: Singular value decomposition JVP not implemented for full matrices
    u, s, vt = np.linalg.svd(A) 
    return u @ np.diag(np.clip(s, eps, 1)) @ vt

def random_rotation(seed, n, theta=None):
    key1, key2 = jr.split(seed)

    if theta is None:
        # Sample a random, slow rotation
        theta = 0.5 * np.pi * jr.uniform(key1)

    if n == 1:
        return jr.uniform(key1) * np.eye(1)

    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    out = np.eye(n)
    out = out.at[:2, :2].set(rot)
    q = np.linalg.qr(jr.uniform(key2, shape=(n, n)))[0]
    return q.dot(out).dot(q.T)

def scale_matrix_by_norm(M):

    # M /= np.linalg.norm(M)

    return M

# def scale_input_weight_matrix(u, v, s, dim, eps = 1e-3):

#     U, _ = np.linalg.qr(u.reshape((dim, dim)))
#     V, _ = np.linalg.qr(v.reshape((dim, dim)))
#     singular_values = sigmoid(s) * (1 - eps) + eps / 2
#     Sigma = np.diag(singular_values)
#     A = U @ Sigma @ V.T
#     # A -= np.eye(dim) * 1e-3

#     return scale_matrix_by_norm(A)

def construct_dynamics_matrix(u, v, s, dim_1, dim_2, eps = 1e-3):

    U, _ = np.linalg.qr(u.reshape((dim_1, dim_1)))
    V, _ = np.linalg.qr(v.reshape((dim_2, dim_2)))
    # U, _ = np.linalg.qr(u.reshape((dim, dim)))
    # V, _ = np.linalg.qr(v.reshape((dim, dim)))
    singular_values = sigmoid(s) * (1 - eps) + eps / 2
    Sigma = np.diag(singular_values)
    A = U @ Sigma @ V.T
    A = scale_matrix_by_norm(A)
    # A -= np.eye(dim) * 1e-3

    # U, _ = np.linalg.qr(u.reshape((dim_1, dim_1)))
    # V, _ = np.linalg.qr(v.reshape((dim_2, dim_2)))
    # Sigma = np.zeros((dim_1, dim_2))
    # mask = np.eye(max(dim_1, dim_2))
    # Sigma = Sigma.at[mask[:dim_1,:dim_2]].set(sigmoid(s) * (1 - eps) + eps / 2)
    # A = U @ Sigma @ V.T
    # A = scale_matrix_by_norm(A)

    return A

# Computes A.T @ Q^{-1} @ A in a way that's guaranteed to be symmetric
def inv_quad_form(Q, A):
    sqrt_Q = np.linalg.cholesky(Q)
    trm = solve_triangular(sqrt_Q, A, lower=True, check_finite=False)
    return trm.T @ trm

def inv_symmetric(Q):
    sqrt_Q = np.linalg.cholesky(Q)
    sqrt_Q_inv = np.linalg.inv(sqrt_Q)
    return sqrt_Q_inv.T @ sqrt_Q_inv

# Converts from (A, b, Q) to (J, L, h)
def dynamics_to_tridiag(dynamics_params, T, D):
    # Q1, m1, A, Q, B = dynamics_params["Q1"], \
    #     dynamics_params["m1"], dynamics_params["A"], \
    #     dynamics_params["Q"], dynamics_params["B"]
    # Q1, m1, A, A_bar, B, Q, U, S = dynamics_params["Q1"], \
    # dynamics_params["m1"], dynamics_params["A"], dynamics_params["Abar"], \
    # dynamics_params["B"], dynamics_params["Q"], \
    # dynamics_params["U"], dynamics_params["S"]
    Q1, A, Q = dynamics_params["Q1"], dynamics_params["A"], dynamics_params["Q"],
    # A_bar = A + B @ U
    # Q_bar = Q + B @ S @ B.T
    # diagonal blocks of precision matrix
    J = np.zeros((T, D, D))
    J = J.at[0].add(inv_symmetric(Q1))
    J = J.at[:-1].add(inv_quad_form(Q, A))
    J = J.at[1:].add(inv_symmetric(Q))
    # lower diagonal blocks of precision matrix
    L = -np.linalg.solve(Q, A)
    L = np.tile(L[None, :, :], (T - 1, 1, 1))
    return { "J": J, "L": L}
    # h = np.zeros((T, D)) 
    # h = h.at[0].add(np.linalg.solve(Q1, m1))
    # h = h.at[:-1].add(-np.dot(A.T, np.linalg.solve(Q, b)))
    # h = h.at[1:].add(np.linalg.solve(Q, b))
    # return { "J": J, "L": L, "h": h }

def get_scaler(scaler):
    scalers = {
        "minmax": MinMaxScaler,
        "standard": StandardScaler,
        "maxabs": MaxAbsScaler,
        "robust": RobustScaler,
    }
    return scalers.get(scaler.lower())()

def get_train_state(train_params, all_optimisers=[], all_params=[]):

    # orbax tutorial 
    # https://colab.research.google.com/github/google/orbax/blob/main/checkpoint/orbax/checkpoint/orbax_checkpoint.ipynb#scrollTo=G4Gv7L4olc_n

    # file describing checkpoint_manager methods and attributes
    # https://github.com/google/orbax/blob/main/checkpoint/orbax/checkpoint/checkpoint_manager.py

    if train_params['reload_state']:

        ckpt_metrics_dir = train_params["reload_dir"] + '/checkpoints/'

    else:

        ckpt_metrics_dir = train_params["save_dir"] + '/checkpoints/'

    # keep only the best 'max_to_keep' checkpoints
    options = CheckpointManagerOptions(max_to_keep=3, best_fn=lambda metrics: metrics, best_mode='min')
    if train_params["inference_method"] != "rpm":
        mngr = CheckpointManager(ckpt_metrics_dir,  
                                 {'recognition_model_state': AsyncCheckpointer(PyTreeCheckpointHandler()),
                                  'decoder_model_state': AsyncCheckpointer(PyTreeCheckpointHandler()),
                                  'prior_model_state': AsyncCheckpointer(PyTreeCheckpointHandler())},
                                 options)
    else:
        mngr = CheckpointManager(ckpt_metrics_dir,  
                                 {'recognition_model_state': AsyncCheckpointer(PyTreeCheckpointHandler()),
                                  'prior_model_state': AsyncCheckpointer(PyTreeCheckpointHandler()),
                                  'delta_q_state': AsyncCheckpointer(PyTreeCheckpointHandler()),
                                  'delta_f_tilde_state': AsyncCheckpointer(PyTreeCheckpointHandler())},
                                 options)

    if train_params['reload_state']:

        # restore the best checkpoint from train_params["reload_dir"]
        items = mngr.restore(mngr.best_step())

        if train_params["inference_method"] != "rpm":
            states = [items['recognition_model_state'],  items['decoder_model_state'],  items['prior_model_state']]
        else:
            states = [items['recognition_model_state'],  items['prior_model_state'], items['delta_q_state'],  items['delta_f_tilde_state']]
        # items['metrics']

        # change ckpt_metrics_dir to train_params["save_dir"]
        from etils import epath
        mngr._directory = epath.Path(train_params["save_dir"] + '/checkpoints/')

    else:

        states = []
        for params, optimiser in zip(all_params, all_optimisers):

            states.append(train_state.TrainState.create(apply_fn = lambda x: x, params = params, tx = optimiser)) # apply_fn unused (normally would be model.apply)

    return states, mngr