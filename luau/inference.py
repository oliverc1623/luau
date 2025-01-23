# %%

from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from minigrid.wrappers import ImgObsWrapper

from luau.ppo import Agent


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
door_locked = False
num_envs = 1


# env setup
def make_env(subenv_seed: int) -> gym.Env:
    """Create the environment."""

    def _init() -> gym.Env:
        env = gym.make("FourRoomDoorKeyLocked-v0", render_mode="rgb_array")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = ImgObsWrapper(env)
        env.reset(seed=subenv_seed)
        env.action_space.seed(subenv_seed)
        env.observation_space.seed(subenv_seed)
        return env

    return _init


envs = [make_env(seed + i) for i in range(num_envs)]
envs = gym.vector.SyncVectorEnv(envs)

# %%
observation_shape = np.transpose(envs.single_observation_space.sample(), (2, 0, 1)).shape
state_dim = envs.observation_space.shape[-1]
# Initialize teacher model
teacher_model_path = (
    "../../../pvcvolume/runs/MiniGrid-DoorKey-6x6-v0__PPO_Teacher_Source__1733791056/"
    "model/MiniGrid-DoorKey-6x6-v0__PPO_Teacher_Source__1733791056.pth"
)
teacher_source_agent = Agent(envs).to(device)
teacher_source_agent.load_state_dict(torch.load(teacher_model_path))


# %%

state, _ = envs.reset()
done = False
truncated = False
step = 0
while not done and not truncated:
    state = preprocess(state)
    action, _, _, _ = teacher_source_agent.get_action_and_value(state)
    state, reward, done, truncated, _ = envs.step(action.cpu().numpy())
    img = envs.render()
print(reward)
envs.close()

# %%
