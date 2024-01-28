from gymnasium.envs.registration import register

register(
    id="flying_sim/PIDFlightArena-eval",
    entry_point="flying_sim.envs:PIDFlightEvalEnv",
)

register(
    id="flying_sim/PIDFlightArena-train",
    entry_point="flying_sim.envs:PIDFlightTrainEnv",
)
