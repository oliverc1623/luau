# %%

import copy
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from matplotlib import pyplot as plt

from luau.iaa_env import IntrospectiveEnv
from luau.ppo import SingleEnvPPO


root_path = Path(__file__).resolve().parent.parent


# %%
rng = np.random.default_rng()
env = IntrospectiveEnv(rng=rng, size=9, locked=False, render_mode="rgb_array")
env = gym.wrappers.RecordEpisodeStatistics(env)

# %%
state_dim = env.observation_space["image"].shape[2]
action_dim = env.action_space.n

agent = SingleEnvPPO(
    env=env,
    lr_actor=0.0003,
    gamma=0.0003,
    eps_clip=0.0003,
    k_epochs=0.0003,
    minibatch_size=0.0003,
    horizon=6,
    gae_lambda=0.0003,
)
agent.load(Path("/root/../pvcvolume/models/PPO/IntrospectiveEnvUnlocked/run_1_seed_1623/PPO_IntrospectiveEnvUnlocked_run_1_seed_1623.pth"))

teacher_target = SingleEnvPPO(
    env=env,
    lr_actor=0.0003,
    gamma=0.0003,
    eps_clip=0.0003,
    k_epochs=0.0003,
    minibatch_size=0.0003,
    horizon=6,
    gae_lambda=0.0003,
)
teacher_target.policy = copy.deepcopy(agent.policy)
teacher_target.optimizer = torch.optim.Adam(teacher_target.policy.parameters(), lr=0.005, eps=1e-5)
# %%

# %%

state, _ = env.reset()
done = False
while not done:
    state = teacher_target.preprocess(state)
    action, _, _ = teacher_target.select_action(state)
    state, reward, done, _, _ = env.step(action.item())
    img = env.render()
    plt.imshow(img)
    plt.show()
    print(reward)
env.close()