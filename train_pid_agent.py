import gymnasium as gym
from flying_sim.config import Config

config = Config()
env = gym.make("flying_sim:flying_sim/PIDFlightArena-v0", config=config)
observation, info = env.reset(seed=42)

for _ in range(1000):
    action = env.action_space.sample()  # insert policy
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        print("[INFO] Resetting Environment")
        observation, info = env.reset()

env.close()
