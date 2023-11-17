import gymnasium as gym
import torch as th
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.policies import BasePolicy

# Define a custom Neural Network Policy in accordance to A2C implementation
# using stable_baselines3 package
class A2C_Policy(BasePolicy):
    def __init__(self, *args, actor_net_arch=None, critic_net_arch=None, **kwargs):
        super(A2C_Policy, self).__init__(*args, **kwargs)
        
        if actor_net_arch is None:
            actor_net_arch = [128, 64]   # Default architecture for the actor (policy)
        if critic_net_arch is None:
            critic_net_arch = [264, 128]  # Default architecture for the critic (value function)

        self.net_arch = [dict(pi=actor_net_arch, vf=critic_net_arch)]

##---------------------- Implementation ----------------------##
# Define different parameters for actor and critic networks
# actor_net_arch = [64, 64]
# critic_net_arch = [32, 32]
# actor_activation_fn = th.nn.ReLU
# critic_activation_fn = th.nn.Tanh
# Learning rates and L2 regularization factors for actor and critic
# actor_learning_rate = 0.0005
# critic_learning_rate = 0.001
# actor_l2_reg = 1e-5
# critic_l2_reg = 1e-4

# Define and train the A2C model with different parameters for actor and critic networks
# model = A2C(policy=CustomPolicy, env=env, verbose=1,
#            policy_kwargs=dict(actor_net_arch=actor_net_arch,
#                               critic_net_arch=critic_net_arch,
#                               features_extractor_kwargs=dict(activation_fn=actor_activation_fn),
#                               net_arch_kwargs=dict(activation_fn=critic_activation_fn),
#                               optimizer_kwargs=dict(
#                                   actor_lr=actor_learning_rate,
#                                   critic_lr=critic_learning_rate,
#                                 actor_reg_coef=actor_l2_reg,
#                                   critic_reg_coef=critic_l2_reg))))

# model.learn(total_timesteps=10000)
