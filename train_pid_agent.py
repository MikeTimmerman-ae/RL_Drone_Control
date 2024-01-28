import torch
import os
import shutil
import matplotlib

from flying_sim.configs.config import Config
from flying_sim.callback import CustomCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from stable_baselines3.ppo.ppo import PPO


matplotlib.use('TkAgg')


def main():

    #################################################
    #### 0. Configuration & Environment set up
    #################################################
    print('#### 0. Configuration & Environment set up')
    config = Config()

    # save policy to output_dir
    if os.path.exists(config.training.model_dir) and config.training.overwrite:  # if I want to overwrite the directory
        shutil.rmtree(config.training.model_dir)  # delete an entire directory tree

    if not os.path.exists(config.training.model_dir):
        os.makedirs(config.training.model_dir)  # create new output directory

    if os.path.exists(os.path.join(config.training.model_dir, 'configs')):
        shutil.rmtree(os.path.join(config.training.model_dir, 'configs'))  # delete configuration subfolder
        shutil.copytree('flying_sim/configs', os.path.join(config.training.model_dir, 'configs'))   # copy current config in directory

    # cuda and pytorch settings
    torch.manual_seed(config.env_config.seed)  # Set seed for generating random numbers
    torch.cuda.manual_seed_all(config.env_config.seed)  # Set seed for generating random numbers on all GPUs
    if config.training.cuda:
        # torch.cuda.set_device(1)
        if config.training.cuda_deterministic:  # reproducible but slower
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        else:  # not reproducible but faster
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False

    torch.set_num_threads(config.training.num_threads)      # Set number of threads used for intraop parallelism on CPU
    device = torch.device("cuda:0" if config.training.cuda else "cpu")

    # Create a wrapped, monitored VecEnv
    envs = make_vec_env(config.env_config.env_train,
                        n_envs=config.training.num_processes,
                        monitor_kwargs={'info_keywords': ["is_success"]})   # Create ShmemVecEnv Object (Wrapper of Vectorized Env)
    eval_env = make_vec_env(config.env_config.env_eval, 3)
    #################################################
    #### 1. RL network (Ego agent)
    #################################################
    print('\n #### 1. Train RL network')

    # 1.1 create RL policy network
    if config.training.continue_training:
        model = PPO.load(config.training.model_dir + config.training.parent_model, envs)
    else:
        model = PPO('MlpPolicy', envs,
                    verbose=1,
                    n_steps=config.ppo.num_steps,
                    tensorboard_log=config.training.model_dir)

    # 1.2 train policy network
    custom_callback = CustomCallback()
    eval_callback = EvalCallback(eval_env,
                                 best_model_save_path=config.training.model_dir + 'best_policy',
                                 # log_path="./logs/"
                                 eval_freq=8192,
                                 deterministic=True,
                                 verbose=1)
    callback = CallbackList([custom_callback, eval_callback])
    model.learn(total_timesteps=config.training.num_env_steps,
                log_interval=config.training.log_interval,
                callback=callback,
                tb_log_name=config.training.new_model[-2:-1],
                reset_num_timesteps=False)

    # 1.3 save policy
    model.save(config.training.model_dir + config.training.new_model)
    model.policy.save(config.training.model_dir + config.training.new_policy)

    # 1.4 close environment
    envs.close()


if __name__ == '__main__':
    main()
