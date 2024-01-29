#determine gradient clipping value by logging gradient norms during training and assessing
# tensorboard - save loss function, save gifs and rewards on a small number (say 3) of examples

# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

# Q and Q currently diagonal
# delta mu comented out in train and below
# using non optimal prior
# layer norm added to trunkMLP

def main():

    # to change an argument via the command line, for example: python main.py --reload_folder_name 'run_1' --reload_state True

    from jax import config
    # config.update("jax_enable_x64", True)
    # config.update("jax_debug_nans", True)
    # config.update("jax_disable_jit", True)

    # from jax.config import config
    # config.update("jax_disable_jit", True)
    # config.update("jax_debug_nans", False)

    # type help at a breakpoint() to see available commands
    # use xeus-python kernel -- Python 3.9 (XPython) -- for debugging

    from jax import devices
    print('devices:', devices())

    import argparse
    from distutils.util import strtobool
    parser = argparse.ArgumentParser(description = 'hyperparameters')

    parser.add_argument('--save_dir',                    default = '/nfs/nhome/live/jheald/svae/runs/NoDeltaQ_NoDeltaF_NoDeltaFNP_NoFTimeDepend_NoAnsatz_MyInference_svae_seed0')
    parser.add_argument('--reload_state',                type = bool, default = False)
    parser.add_argument('--reload_dir',                  default = '/nfs/nhome/live/jheald/svae/runs/first_attempt')    
    # parser.add_argument('--save_dir',                    default = '/Users/james/Dropbox (UCL)/ucgtjhe@ucl.ac.uk’s files/James MacBook/Gatsby/svae/runs/first_attempt')
    # parser.add_argument('--reload_dir',                  default = '/Users/james/Dropbox (UCL)/ucgtjhe@ucl.ac.uk’s files/James MacBook/Gatsby/svae/runs/first_attempt')
    
    parser.add_argument('--log_to_wandb',                type = bool, default = True)
    parser.add_argument('--project_name',                default = 'RPM-LDS')
    parser.add_argument('--log_every_n_epochs',          type = int, default = 100)

    # options
    parser.add_argument('--use_linear_rpm',              type = lambda x: bool(strtobool(x)), default = '0')
    parser.add_argument('--use_delta_nat_q',             type = lambda x: bool(strtobool(x)), default = '0') #### compute entropy with sparse matrices? if q is time independent you can do it block wise! - but maybe less good for learning Q and a?
    # parser.add_argument('--use_delta_nat_q_nonpara',             type = lambda x: bool(strtobool(x)), default = '0')
    parser.add_argument('--stop_grad_q',                 type = lambda x: bool(strtobool(x)), default = '0')
    parser.add_argument('--use_ansatz',                  type = lambda x: bool(strtobool(x)), default = '1')
    parser.add_argument('--use_prior_for_F',             type = lambda x: bool(strtobool(x)), default = '1')
    parser.add_argument('--use_delta_nat_f_tilde',       type = lambda x: bool(strtobool(x)), default = '0')
    parser.add_argument('--use_delta_nat_f_tilde_np',    type = lambda x: bool(strtobool(x)), default = '0')
    parser.add_argument('--zero_last_layers',            type = lambda x: bool(strtobool(x)), default = '0')
    parser.add_argument('--f_time_dependent',            type = lambda x: bool(strtobool(x)), default = '0')
    parser.add_argument('--use_my_inference',            type = lambda x: bool(strtobool(x)), default = '1')
    # parameterise A
    # different learning rate schedules, warmup, cyclic etc as in https://proceedings.mlr.press/v119/dong20e/dong20e-supp.pdf
    # ensure controllability? https://arxiv.org/pdf/2301.09519.pdf
    # plot inferred latents under prior of input-driven system for RPM and the optimal kalman filter

    parser.add_argument('--jax_seed',                    type = int, default = 0)

    parser.add_argument('--latent_dims',                 type = int, default = 3)
    parser.add_argument('--emission_dims',               type = int, default = 3)
    parser.add_argument('--input_dims',                  type = int, default = 1)

    parser.add_argument('--inference_method',            default = 'rpm') # 'svae', lds
    parser.add_argument('--rec_trunk_features',          type = list, default = [50, 50, 50])
    parser.add_argument('--rec_head_mean_features',      type = list, default = [])
    parser.add_argument('--rec_head_var_features',       type = list, default = [])
    parser.add_argument('--rec_diagonal_covariance',     type = bool, default = False)
    parser.add_argument('--GRU_dim',                     type = int, default = 50)
    # parser.add_argument('--dec_features',                type = list, default = [512])
    
    parser.add_argument('--use_parallel_kf',             type = bool, default = True)
    parser.add_argument('--mask_size',                   type = int, default = 0)
    parser.add_argument('--mask_start',                  type = int, default = 0)

    parser.add_argument('--train_size',                  type = int, default = 100) # 800
    parser.add_argument('--val_size',                    type = int, default = 100) # 200
    parser.add_argument('--train_batch_size',            type = int, default = 100)
    parser.add_argument('--val_batch_size',              type = int, default = 100)

    parser.add_argument('--early_stop_start',            type = int, default = 5000)
    parser.add_argument('--min_delta',                   type = float, default = 1e-3)
    parser.add_argument('--patience',                    type = int, default = 25)
    parser.add_argument('--max_iters',                   type = int, default = 5000)
    parser.add_argument('--checkpoint_every_n_epochs',   type = int, default = 100)

    parser.add_argument('--base_lr',                     type = float, default = 1e-3)
    parser.add_argument('--lr_decay',                    type = bool, default = False) # definitely won't want this on when I am learning online
    parser.add_argument('--prior_base_lr',               type = float, default = 1e-2)
    parser.add_argument('--prior_lr_warmup',             type = float, default = False)
    parser.add_argument('--delta_nat_f_tilde_lr',        type = float, default = 1e-3)
    parser.add_argument('--max_grad_norm',               type = float, default = 10.)
    parser.add_argument('--weight_decay',                type = float, default = 0.0000) # 0.0001
    parser.add_argument('--beta_transition_begin',       type = int, default = 1000)
    parser.add_argument('--beta_transition_steps',       type = int, default = 1000)

    # for LDS dataset generation
    parser.add_argument('--num_trials',                  type = int, default = 100)
    parser.add_argument('--num_timesteps',               type = int, default = 100)
    parser.add_argument('--snr',                         default = 'large')
    parser.add_argument('--latent_dims_dataset',         type = int, default = 3)
    parser.add_argument('--emission_dims_dataset',       type = int, default = 3)
    parser.add_argument('--input_dims_dataset',          type = int, default = 1)
    # "dataset_size": "medium", # FOR LDS DATASET GENERATION

    args = parser.parse_args()

    for k, v in vars(args).items():
        print(k, v)

    # save the hyperparameters
    import pickle
    import os
    path = args.save_dir + '/hyperparameters'
    os.makedirs(os.path.dirname(path))
    file = open(path, 'wb') # change 'wb' to 'rb' to load
    pickle.dump(args, file) # change to args = pickle.load(file) to load
    file.close()

    from svae.experiments import run_pendulum_control, run_lds
    all_results, all_models = run_lds(vars(args))
    # all_results, all_models = run_pendulum_control(vars(args))

if __name__ == '__main__':

    main()
