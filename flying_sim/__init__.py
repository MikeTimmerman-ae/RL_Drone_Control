from gymnasium.envs.registration import register

register(
    id="flying_sim/PIDFlightArena-v0",
    entry_point="flying_sim.envs:PIDFlightEnv",
    max_episode_steps=300,
)
