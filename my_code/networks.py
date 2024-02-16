from functools import partial # pylint: disable=g-importing-member
from typing import (
  Any,
  Callable,
  Optional,
  Tuple,
)

from flax.typing import (
  PRNGKey,
  Dtype,
  Initializer,
)

from flax.linen import initializers
from flax.linen.activation import sigmoid, tanh
from flax.linen.linear import default_kernel_init, Dense
from flax.linen.module import compact, nowrap

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