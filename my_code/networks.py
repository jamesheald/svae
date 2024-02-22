from jax import numpy as np
import flax.linen as nn
from jax import vmap, random, value_and_grad

from utils import construct_covariance_matrix
from dynamax.utils.utils import psd_solve

from flax.linen import initializers
from flax.linen.activation import sigmoid, tanh
from flax.linen.linear import default_kernel_init, Dense
from flax.linen.module import compact, nowrap

from functools import partial # pylint: disable=g-importing-member
from typing import (
  Any,
  Callable,
  Optional,
  Tuple,
  List,
)

from flax.typing import (
  PRNGKey,
  Dtype,
  Initializer,
)

class GRUCell_LN(nn.RNNCellBase):
    r"""GRU cell.

    The mathematical definition of the cell is as follows

    .. math::

      \begin{array}{ll}
      r = \sigma(W_{ir} x + b_{ir} + W_{hr} h) \\
      z = \sigma(W_{iz} x + b_{iz} + W_{hz} h) \\
      n = \tanh(W_{in} x + b_{in} + r * (W_{hn} h + b_{hn})) \\
      h' = (1 - z) * n + z * h \\
      \end{array}

    where x is the input and h, is the output of the previous time step.

    Example usage::

    >>> import flax.linen as nn
    >>> import jax, jax.numpy as jnp

    >>> x = jax.random.normal(jax.random.key(0), (2, 3))
    >>> layer = nn.GRUCell(features=4)
    >>> carry = layer.initialize_carry(jax.random.key(1), x.shape)
    >>> variables = layer.init(jax.random.key(2), carry, x)
    >>> new_carry, out = layer.apply(variables, carry, x)

    Attributes:
    features: number of output features.
    gate_fn: activation function used for gates (default: sigmoid).
    activation_fn: activation function used for output and memory update
      (default: tanh).
    kernel_init: initializer function for the kernels that transform
      the input (default: lecun_normal).
    recurrent_kernel_init: initializer function for the kernels that transform
      the hidden state (default: initializers.orthogonal()).
    bias_init: initializer for the bias parameters (default: initializers.zeros_init())
    dtype: the dtype of the computation (default: None).
    param_dtype: the dtype passed to parameter initializers (default: float32).
    """

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
        """Gated recurrent unit (GRU) cell.

        Args:
          carry: the hidden state of the GRU cell,
            initialized using ``GRUCell.initialize_carry``.
          inputs: an ndarray with the input for the current time step.
            All dimensions except the final are considered batch dimensions.

        Returns:
          A tuple with the new carry and the output.
        """
        h = carry
        hidden_features = h.shape[-1]
        # input and recurrent layers are summed so only one needs a bias.
        dense_h = partial(
          Dense,
          features=hidden_features,
          use_bias=False,
          dtype=self.dtype,
          param_dtype=self.param_dtype,
          kernel_init=self.recurrent_kernel_init,
          bias_init=self.bias_init,
        )
        dense_i = partial(
          Dense,
          features=hidden_features,
          use_bias=True,
          dtype=self.dtype,
          param_dtype=self.param_dtype,
          kernel_init=self.kernel_init,
          bias_init=self.bias_init,
        )

        # r = self.gate_fn(dense_i(name='ir')(inputs) + dense_h(name='hr')(h))
        # z = self.gate_fn(dense_i(name='iz')(inputs) + dense_h(name='hz')(h))
        # # add bias because the linear transformations aren't directly summed.
        # n = self.activation_fn(
        #   dense_i(name='in')(inputs) + r * dense_h(name='hn', use_bias=True)(h)
        # )

        r = self.gate_fn(nn.LayerNorm()(dense_i(name='ir')(inputs)) + nn.LayerNorm()(dense_h(name='hr')(h)))
        z = self.gate_fn(nn.LayerNorm()(dense_i(name='iz')(inputs)) + nn.LayerNorm()(dense_h(name='hz')(h)))
        # add bias because the linear transformations aren't directly summed.
        n = self.activation_fn(
          nn.LayerNorm()(dense_i(name='in')(inputs)) + r * nn.LayerNorm()(dense_h(name='hn', use_bias=True)(h))
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

class control_network(nn.Module):
    u_emb_dim: int
    h_dims: List
    layer_norm: bool

    def setup(self):

      self.mlp = [[nn.Dense(features=h_dim), nn.LayerNorm()] if self.layer_norm else [nn.Dense(features=h_dim), lambda x: x] for h_dim in self.h_dims]

      self.dense_out = nn.Dense(features=self.u_emb_dim)

    def __call__(self, u):

        def embed_u(x):

          for dense, layer_norm in self.mlp:
            
            x = dense(x)
            x = layer_norm(x)
            x = nn.relu(x)

          x = self.dense_out(x)

          return x

        if u.ndim == 1:
            x = embed_u(u)
        elif u.ndim == 2:
            x = vmap(embed_u)(u)
        elif u.ndim == 3:
            x = vmap(vmap(embed_u))(u)

        return x

class emission_potential(nn.Module):
    z_dim: int
    h_dims: int
    diagonal_covariance: bool
    layer_norm: bool

    def setup(self):

      self.mlp = [[nn.Dense(features=h_dim), nn.LayerNorm()] if self.layer_norm else [nn.Dense(features=h_dim), lambda x: x] for h_dim in self.h_dims]

      if self.diagonal_covariance:
        self.dense_out = nn.Dense(features=self.z_dim * 2)
      else:
        self.dense_out = nn.Dense(features=self.z_dim + self.z_dim * (self.z_dim + 1) // 2)

    def __call__(self, x):

        def get_natural_parameters(x):

            mu, var_output_flat = np.split(x, [self.z_dim])

            if self.diagonal_covariance:

              Sigma = np.diag(np.exp(var_output_flat))
              J = np.diag(1 / np.exp(var_output_flat))
            else:
              Sigma = construct_covariance_matrix(var_output_flat, self.z_dim)
              J = psd_solve(Sigma, np.eye(self.z_dim))

            h = J @ mu

            return Sigma, mu, J, h

        for dense, layer_norm in self.mlp:
            
            x = dense(x)
            x = layer_norm(x)
            x = nn.relu(x)

        x = self.dense_out(x)

        if x.ndim == 1:
            Sigma, mu, J, h = get_natural_parameters(x)
        elif x.ndim == 2:
            Sigma, mu, J, h = vmap(get_natural_parameters)(x)
        elif x.ndim == 3:
            Sigma, mu, J, h = vmap(vmap(get_natural_parameters))(x)

        return {'Sigma': Sigma, 'mu': mu, 'J': J, 'h': h}

# network for creating time-dependent RPM factors f_t(z_t | x_t), where the time dependence is provided by a GRU
class GRU_RPM(nn.Module):
    carry_dim: int
    h_dims: List
    z_dim: int
    T: int
    diagonal_covariance: bool
    layer_norm: bool

    def setup(self):
        
        self.GRU = nn.RNN(GRUCell_LN_NoInput(self.carry_dim))
        # self.GRU = nn.RNN(nn.GRUCell(self.carry_dim))

        self.carry_init = self.param('carry_init', lambda rng, shape: nn.initializers.normal(1.0)(rng, shape), (self.carry_dim,))

        self.mlp = [[nn.Dense(features=h_dim), nn.LayerNorm()] if self.layer_norm else [nn.Dense(features=h_dim), lambda x: x] for h_dim in self.h_dims]

        if self.diagonal_covariance:
          self.dense_out = nn.Dense(features=self.z_dim * 2)
        else:
          self.dense_out = nn.Dense(features=self.z_dim + self.z_dim * (self.z_dim + 1) // 2)

    def __call__(self, x):

        def get_natural_parameters(x):

            mu, var_output_flat = np.split(x, [self.z_dim])

            if self.diagonal_covariance:
              Sigma = np.diag(np.exp(var_output_flat))
              J = np.diag(1 / np.exp(var_output_flat))
            else:
              Sigma = construct_covariance_matrix(var_output_flat, self.z_dim)
              J = psd_solve(Sigma, np.eye(self.z_dim))

            h = J @ mu

            return Sigma, mu, J, h

        carry = self.GRU(np.zeros((self.T,1)), initial_carry=self.carry_init)

        # x = self.LN_1(self.dense_1(np.concatenate((carry[None].repeat(x.shape[0], axis=0), x), axis=2)))

        for dense, layer_norm in self.mlp:
            x = dense(x)
            x = layer_norm(x)
            x = nn.relu(x)

        x = self.dense_out(np.concatenate((carry[None].repeat(x.shape[0], axis=0), x), axis=2))

        Sigma, mu, J, h = vmap(vmap(get_natural_parameters))(x)

        return {'Sigma': Sigma, 'mu': mu, 'J': J, 'h': h}

class delta_q_params(nn.Module):
    carry_dim: int
    h_dims: List
    z_dim: int
    diagonal_covariance: bool
    layer_norm: bool

    def setup(self):
        
        # self.BiGRU = nn.Bidirectional(nn.RNN(nn.GRUCell(self.carry_dim)), nn.RNN(nn.GRUCell(self.carry_dim)))
        self.BiGRU = nn.Bidirectional(nn.RNN(GRUCell_LN(self.carry_dim)), nn.RNN(GRUCell_LN(self.carry_dim)))

        self.mlp = [[nn.Dense(features=h_dim), nn.LayerNorm()] if self.layer_norm else [nn.Dense(features=h_dim), lambda x: x] for h_dim in self.h_dims]

        if self.diagonal_covariance:
          self.dense_out = nn.Dense(self.z_dim * 2)
        else:
          self.dense_out = nn.Dense(self.z_dim + self.z_dim * (self.z_dim + 1) // 2)

    def __call__(self, y):

        def get_natural_parameters(x):

            mu, var_output_flat = np.split(x, [self.z_dim])

            if self.diagonal_covariance:
              Sigma = np.diag(np.exp(var_output_flat) + 1e-6) # Sigma = np.diag(np.exp(var_output_flat)+1e-6)
              J = np.diag(1 / np.exp(var_output_flat))
            else:
              Sigma = construct_covariance_matrix(var_output_flat, self.z_dim)
              J = psd_solve(Sigma, np.eye(self.z_dim))

            h = J @ mu

            return Sigma, mu, J, h

        x = self.BiGRU(y)

        for dense, layer_norm in self.mlp:
            
          x = dense(x)
          x = layer_norm(x)
          x = nn.relu(x)

        x = self.dense_out(x)

        if y.ndim == 2:
          Sigma, mu, J, h = vmap(get_natural_parameters)(x)
        elif y.ndim == 3:
          Sigma, mu, J, h = vmap(vmap(get_natural_parameters))(x)

        return {'Sigma': Sigma, 'mu': mu, 'smoothed_covariances': Sigma, 'smoothed_means': mu, 'J': J, 'h': h}

class sample_from_q(nn.Module):
    h_dim: int
    z_dim: int

    def setup(self):
        
        self.dense_1 = nn.Dense(features=self.h_dim)
        self.LN_1 = nn.LayerNorm()
        self.dense_2 = nn.Dense(features=self.h_dim)
        self.LN_2 = nn.LayerNorm()
        self.dense_3 = nn.Dense(features=self.h_dim)
        self.LN_3 = nn.LayerNorm()
        self.dense_4 = nn.Dense(features=self.z_dim)

    def __call__(self, carry, inputs):

        # sample from q(z_t | z_{t-1}, u_{t-1}, y_{t:T}), where y_{t:T} is represented by y_embed
        # first time point should really be sampled from q(z_0 | y_{0:T}) 

        z_prev = carry
        u_prev, y_embed, key = inputs

        epsilon = random.normal(key, z_prev.shape)

        x = self.LN_1(self.dense_1(np.concatenate((z_prev, u_prev, y_embed, epsilon))))
        x = nn.relu(x)
        x = self.LN_2(self.dense_2(x))
        x = nn.relu(x)
        x = self.LN_3(self.dense_3(x))
        x = nn.relu(x)
        z = self.dense_4(x)

        carry = z
        outputs = z

        return carry, outputs

class implicit_q(nn.Module):
    carry_dim: int
    h_dim: int
    z_dim: int

    def setup(self):
        
        self.GRU = nn.RNN(GRUCell_LN(self.carry_dim))

        self.carry_init = self.param('carry_init', lambda rng, shape: np.zeros(shape), (self.carry_dim,))

        scanner = nn.scan(sample_from_q, variable_broadcast = 'params', split_rngs = {"params": False})
        self.sampler = scanner(h_dim=self.h_dim, z_dim=self.z_dim)

    def __call__(self, y, u, key):

        noise = random.normal(key, y.shape)

        y_embed = self.GRU(y, reverse=True, keep_order=True, initial_carry=self.carry_init)

        # dummy variables for first time point
        z_prev = np.zeros(self.z_dim)
        u_prev = np.zeros(u.shape[-1])
        
        carry = z_prev
        keys = random.split(key, u.shape[0])
        inputs = np.vstack((u_prev, u[:-1,:])), y_embed, keys
        _, z = self.sampler(carry, inputs)

        return z, y_embed

class discriminate(nn.Module):
    h_dim: int

    @nn.compact
    def __call__(self, z, z_prev, u_prev, y_embed):

        x = nn.LayerNorm()(nn.Dense(features = self.h_dim)(np.concatenate((z, z_prev, u_prev, y_embed), axis=2)))
        x = nn.relu(x)
        x = nn.LayerNorm()(nn.Dense(features = self.h_dim)(x))
        x = nn.relu(x)
        x = nn.LayerNorm()(nn.Dense(features = self.h_dim)(x))
        x = nn.relu(x)
        x = nn.Dense(features = 1)(x)

        return x.squeeze()

class estimate_log_density_ratio(nn.Module):
    h_dim: int

    def setup(self):

        None

    def __call__(self, z_p, z_q, y_embed, u, state):

        def bernoulli_logarithmic_loss(params, z_p, z_q, y_embed, u, state):

            # dummy variables for first time point
            B, D, U = z_p.shape[0], z_p.shape[-1], u.shape[-1]
            z_p_prev = np.concatenate((np.zeros((B,1,D)), z_p[:,:-1,:]), axis=1)
            z_q_prev = np.concatenate((np.zeros((B,1,D)), z_q[:,:-1,:]), axis=1)
            u_prev = np.concatenate((np.zeros((B,1,U)), u[:,:-1,:]), axis=1)

            log_density_ratio_prior_sample = state.apply_fn(params, z_p, z_p_prev, u_prev, y_embed)

            log_density_ratio_posterior_sample = state.apply_fn(params, z_q, z_q_prev, u_prev, y_embed)

            # loss to train the log density ratio estimator
            # learn log p(z_t | z_{t-1}, u_{t-1}) - log q(z_t | z_{t-1}, u_{t-1}, y_{t:T})
            loss = - nn.log_sigmoid(log_density_ratio_posterior_sample) - nn.log_sigmoid(-log_density_ratio_prior_sample)

            return loss.mean(), log_density_ratio_posterior_sample.sum(-1) 
            # return loss[:,1:].sum(-1).mean(), log_density_ratio_posterior_sample[:,1:].sum(-1) # ignore first time point

            # logits = np.concatenate((log_density_ratio_posterior_sample,log_density_ratio_prior_sample))
            # labels = np.concatenate((np.ones(B), np.zeros(B)))
            # loss = opt.sigmoid_binary_cross_entropy(logits, labels)

            # return loss, log_density_ratio_posterior_sample

        loss_grad = value_and_grad(bernoulli_logarithmic_loss, has_aux = True)

        batch_size = y_embed.shape[0]
        (loss, log_density_ratio_posterior_sample), grads = loss_grad(state.params, z_p, z_q, y_embed, u, state)

        state = state.apply_gradients(grads = grads)

        loss, log_density_ratio_posterior_sample = bernoulli_logarithmic_loss(state.params, z_p, z_q, y_embed, u, state)

        # batch_size = y.shape[0]
        # loss, log_density_ratio_posterior_sample = bernoulli_logarithmic_loss(z_p.reshape(batch_size, -1), z_q.reshape(batch_size, -1), y.reshape(batch_size, -1), u.reshape(batch_size, -1))

        return log_density_ratio_posterior_sample, loss, state