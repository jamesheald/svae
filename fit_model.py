###############
# kalman smoothing is done by passing mean of emission potential as observation - weird?! line 249 distributions
# figure out how to turn back use_parallel_kf to true, line 380 experiments
# constrain_prior seems wierd, line 499 of training


import jax.random as jr
from svae.experiments import run_lds, run_pendulum_control

# A hedious global variable I'm going to get rid of...
from svae.datasets import data_dict

run_params = {
    "inference_method": "svae",
    "latent_dims": 3,
    "emission_dims": 3,
    "input_dims": 1,
    # "rnn_dims": 10,
    "seed": jr.PRNGKey(1),
    # "dataset_size": "medium",
    # "snr": "medium",
    "sample_kl": True,
    "train_batch_size": 800,
    "val_batch_size": 200,
    # "dimensionality": "medium",
    "num_timesteps": 99,
    "constrain_prior": False,
    "base_lr": .001,
    "lr_decay": True,
    "prior_base_lr": .01,
    "prior_lr_warmup": True,
    "max_iters": 20000,
    "log_to_wandb": False,
    "min_delta": 1e-3,
    "patience": 50
}

# results = run_lds(run_params)
# model, learned_params, losses = results
# test_result = model.elbo(jr.PRNGKey(0), data_dict["val_data"][0], data_dict["val_data"][0], learned_params)
# test_result.keys()

all_results, all_models = run_pendulum_control(run_params)
breakpoint()
# all_results, all_models = run_lds(run_params)

# all_results[0][0] # model deepLDS object (? == all_models[0]['model'])
# all_results[0][1] # params dict_keys(['dec_params', 'post_params', 'post_samples', 'prior_params', 'rec_params']), e.g. all_results[0][1]['prior_params'] 
# all_results[0][2] # train_losses list, 
from matplotlib import pyplot as plt
plt.plot(all_results[0][2][:2250])
plt.show()

# all_models[0]['model'] # deepLDS object
# all_models[0]['trainer'] # Trainer object


# all_models[0]['model'].prior.get_dynamics_params()

breakpoint()
