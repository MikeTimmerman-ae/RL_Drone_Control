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
        info = env.reset_infos[0]
        figure = plt.figure(f"Number of time steps: {self.num_timesteps}")
        figure.add_subplot().plot(info['states'][:, 0], info['states'][:, 1])
        # Close the figure after logging it
        self.logger.record(f"trajectory/figure{self.num_timesteps}", Figure(figure, close=True), exclude=("stdout", "log", "json", "csv"))
        plt.close()
        return True

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass
