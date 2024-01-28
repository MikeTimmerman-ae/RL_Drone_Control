import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3.common.callbacks import BaseCallback, EventCallback
from stable_baselines3.common.logger import Figure


class CustomCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self, verbose=0):
        super(CustomCallback, self).__init__(verbose)
        # Those variables will be accessible in the callback (they are defined in the base class):
        # The RL model
        # self.model = None  # type: BaseRLModel

        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]

        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int

        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]

        # The logger object, used to report things in the terminal
        # self.logger = None  # type: logger.Logger

        # Sometimes, for event callback, it is useful to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]
        self.prev_successes = 0
        self.prev_deviations = 0
        self.prev_time_outs = 0

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        return True

    def _on_rollout_end(self):
        # Plot drone trajectory (xy-position)
        env = self.model.get_env()
        infos = env.reset_infos

        # Log flight trajectories after each simulation run
        success = 0
        deviation = 0
        time_out = 0

        for i, info in enumerate(infos):
            figure = plt.figure(f"Number of time steps: {self.num_timesteps}")
            plt.plot(info['states'][:, 0], info['states'][:, 1], label='State Trajectory')
            plt.plot(info['reference'][:, 0], info['reference'][:, 1], label='Optimal Trajectory')
            plt.legend()
            plt.grid()
            plt.title('Step Reference Tracking')
            self.logger.record(f"trajectory/trajectory_{i+1}_{self.num_timesteps}", Figure(figure, close=True), exclude=("stdout", "log", "json", "csv"))
            plt.close()

            success += info['reach_count']
            deviation += info['deviation_count']
            time_out += info['timeout_count']

        # Log number of terminations, deviations and success
        if (self.num_timesteps // env.num_envs // infos[0]['num_steps']) % infos[0]['log_interval'] == 0:
            self.logger.record("success_rate/success", success - self.prev_successes)
            self.logger.record("success_rate/deviation", deviation - self.prev_deviations)
            self.logger.record("success_rate/time_out", time_out - self.prev_time_outs)

            self.prev_successes = success
            self.prev_deviations = deviation
            self.prev_time_outs = time_out

        return True

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass
