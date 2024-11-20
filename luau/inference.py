# %%

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython.display import clear_output, display
from minigrid.wrappers import FullyObsWrapper, ImgObsWrapper

from luau.iaa_env import SmallIntrospectiveEnv
from luau.no_frills_ppo import ActorCritic


root_path = Path(__file__).resolve().parent.parent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RGB_CHANNEL = 3


# %%
def preprocess(image: np.array) -> dict:
    """Preprocess the input for a grid-based environment, padding it to (12, 12, channels)."""
    image = torch.from_numpy(image).float()
    if image.ndim == RGB_CHANNEL:  # Single image case with shape (height, width, channels)
        image = image.permute(2, 0, 1)
        # Permute back to (batch_size, channels, height, width)
        image = image.unsqueeze(0).to(device)  # Adding batch dimension
    else:  # Batch case with shape (batch_size, height, width, channels)
        image = image.permute(0, 3, 1, 2).to(device)  # Change to (batch, channels, height, width)
    return image


# %%
seed = 1623
door_locked = True

sub_env_rng = np.random.default_rng(seed)
env = SmallIntrospectiveEnv(rng=sub_env_rng, size=7, locked=door_locked, render_mode="rgb_array", max_steps=360)
env = FullyObsWrapper(env)
env = ImgObsWrapper(env)

# %%
observation_shape = np.transpose(env.observation_space.sample(), (2, 0, 1)).shape
state_dim = env.observation_space.shape[-1]
# Initialize teacher model
teacher_model_path = (
    "../../../pvcvolume/models/PPO/SmallIntrospectiveEnv-Locked-False/run-1-seed-50/SmallIntrospectiveEnv-Locked-False-run-1-seed-50.pth"
)
teacher_source_agent = ActorCritic(state_dim, env.action_space.n).to(device)
teacher_source_agent.load_state_dict(torch.load(teacher_model_path))


# %%

for i in range(51, 100):
    state, _ = env.reset(seed=i)
    done = False
    truncated = False
    step = 0
    while not done and not truncated:
        state = preprocess(state)
        action, _, _, _ = teacher_source_agent(state)
        state, reward, done, truncated, _ = env.step(action.item())
        img = env.render()
    print(i, reward)
env.close()

# %%

state, _ = env.reset(seed=60)
done = False
truncated = False
step = 0
while not done and not truncated:
    state = preprocess(state)
    action, _, _, _ = teacher_source_agent(state)
    state, reward, done, truncated, _ = env.step(action.item())
    img = env.render()

    plt.imshow(img)
    plt.axis("off")  # Hide axis
    display(plt.gcf())  # Display the current figure
    clear_output(wait=True)  # Clear the previous output
    plt.clf()  # Clear the current figure to prevent overlap
    time.sleep(0.1)  # Pause to control the update speed

print(reward)
