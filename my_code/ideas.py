# gradient acculumation so you can do RPM properly instead of minibatches?

# cyclic beta

# plot grad norm
# plot inferred latents direct from rpm factor
# use parallel KF and add 'b' to it'

# transformers instead of BiGRU

# nonlinear u
# learn dynamics bias as B /in R 3 x 2 with B @ \tilde{u}, where \tilde{u} = [u, 1] (colummn vector)
# A x + B f(u) (sequential updates in M step, closed form prior parameters and then gradient based f; pass f(u) to BiGRU in E-step with stopgrad(f(u))

# delta emission potentials
#    - delta terms should be FFN not RNN as you are just perturbing emission potentials - this is also necessary to do online inference (unless using RPM factors alone)
#    - could directly model emission terms (i.e. without perturbing f/F neural network, as this is the q distribution and it doesn't have to be connected to the RPM model) - this is more flexible though maybe less efficient

# non-EM version where you don't incorporate prior into q and do message passing (i.e. just model q directly e.g. with BiGRU)

# condition RPM factor at time t on u_t-1?

# interior bound
#    - no EM (could parameterise f_hat)
#    - EM (parameterise f_hat to allow closed form updates to A)

# could RPM factors be modelled as input(observation)-driven GRU or BiGRU?

# transformer tricks for dealing with quadratic cost (of RPM)?

# test mini-batch version

# consider removing layernorm (or applying it more sparingly, if not getting great results)

# EM stuff
# try not doing closed form updates for prior but gradient based with everything else (maybe different effective learning rates for different parameter sets screws things up)
# is closed form M step correct? (initial covariance?)