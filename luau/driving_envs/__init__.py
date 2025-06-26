from gymnasium.envs.registration import register


register(
    id="driving_envs/StraightRoad-v0",
    entry_point="driving_envs.straight_road:StraightRoad",
)
