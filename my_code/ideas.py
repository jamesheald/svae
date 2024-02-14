# plot grad norm
# specify network layer num
# use parallel KF and add 'b' to it'

# fix prior params to true values and try to learn RPM
# represent A via SVD composition
# learn diagonal covariances
# learn dynamics bias as B /in R 3 x 2 with B @ \tilde{u}, where \tilde{u} = [u, 1] (colummn vector)
# set u's to 0
# try linear RPM
# A x + B f(u) (sequential updates in M step, closed form prior parameters and then gradient based f; pass f(u) to BiGRU in E-step with stopgrad(f(u))
# is closed form M step correct? (initial covariance?)

# delta emission potentials
#    - delta terms should be FFN not RNN as you are just perturbing emission potentials
#    - could directly model emission terms (i.e. without perturbing f/F neural network, as this is the q distribution and it doesn't have to be connected to the RPM model) - this is more flexible though maybe less efficient

# interior bound
#    - no EM (could parameterise f_hat)
#    - EM (parameterise f_hat to allow closed form updates to A)

# try not doing closed form updates for prior but gradient based with everything else (maybe different effective learning rates for different parameter sets screws things up)

# could RPM factors be modelled as input(observation)-driven GRU or BiGRU?

# transformer tricks for dealing with quadratic cost (of RPM)?
# gradient acculumation so you can do RPM properly instead of minibatches?

# add noise to prevent converging to local optimum by a) using MC sample for F, b) minibatches with data shuffled each epocjh
# bigger batch size B to get more data to train neural networks