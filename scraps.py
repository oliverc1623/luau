# %%
import logging  # noqa: I001
from pathlib import Path

import minigrid  # noqa: F401
import gymnasium as gym
import torch
from matplotlib import pyplot as plt
from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper
from stable_baselines3.common.vec_env import DummyVecEnv


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Get the root path
root_path = Path(__file__).resolve().parent.parent
gym.register(id="SmallFourRoomDoorKey-v0", entry_point="luau.multi_room_env:SmallFourRoomDoorKey")


# %%
RGB_CHANNEL = 3
KL_THRESHOLD = 0.01

################################## set device ##################################
print("============================================================================================")
# set device to cpu, mps, or cuda
if torch.cuda.is_available():
    device = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print("Device set to : " + str(device))
print("============================================================================================")


# %%
def make_env(subenv_seed: int) -> gym.Env:
    """Create the environment."""

    def _init() -> gym.Env:
        env = gym.make("SmallFourRoomDoorKey-v0", render_mode="rgb_array")
        env.action_space = gym.spaces.Discrete(7)  # make all 7 actions available
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = RGBImgPartialObsWrapper(env)
        env = ImgObsWrapper(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayscaleObservation(env)
        env = gym.wrappers.FrameStackObservation(env, 4)
        env.reset(seed=subenv_seed)
        env.action_space.seed(subenv_seed)
        env.observation_space.seed(subenv_seed)
        return env

    return _init


seed = 232
num_envs = 2
envs = [make_env(seed + i) for i in range(num_envs)]
env = DummyVecEnv(envs)

# %%
ob, _ = env.reset()
print(ob.shape)

# %%
plt.imshow(ob[0], cmap="gray")
# %%
