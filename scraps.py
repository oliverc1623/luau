# %%

import matplotlib.pyplot as plt
import numpy as np

from luau.iaa_env import IntrospectiveEnv
from luau.ppo import PPO


# %%

rng = np.random.default_rng()
env = IntrospectiveEnv(rng, size=9, locked=True, render_mode="rgb_array")
obs, _ = env.reset()
img = env.render()
plt.imshow(img)
plt.show()

env.observation_space["image"].shape[-1]

# %%
model = PPO(
    env,
    lr_actor=0.0001,
    gamma=0.99,
    k_epochs=4,
    eps_clip=0.02,
    minibatch_size=128,
    horizon=128,
    gae_lambda=0.8,
)

# %%
