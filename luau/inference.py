# %%

from pathlib import Path

import gymnasium as gym
import numpy as np

from luau.iaa_env import IntrospectiveEnv
from luau.ppo import InferencePPO


root_path = Path(__file__).resolve().parent.parent


# %%
rng = np.random.default_rng()
env = IntrospectiveEnv(rng=rng, size=9, locked=False, render_mode="rgb_array")
env = gym.wrappers.RecordEpisodeStatistics(env)

# %%
state_dim = env.observation_space["image"].shape[2]
action_dim = env.action_space.n

agent = InferencePPO(
    state_dim=state_dim,
    action_dim=action_dim,
    env=env,
)

# %%
agent.load(Path(root_path / "models/PPO/IntrospectiveEnvUnlocked/run_0_seed_97/PPO_IntrospectiveEnvUnlocked_run_0_seed_97.pth"))

# %%

state, _ = env.reset()
done = False
while not done:
    state = agent.preprocess(state)
    action = agent.select_action(state)
    state, _, done, _ = env.step(action)
    env.render()
    env.close()
