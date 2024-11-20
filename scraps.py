# %%
import logging
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from minigrid.wrappers import FullyObsWrapper

from luau.iaa_env import SmallIntrospectiveEnv


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Get the root path
root_path = Path(__file__).resolve().parent.parent


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
env = SmallIntrospectiveEnv(rng=np.random.default_rng(25), size=6, locked=False, render_mode="rgb_array")
env = FullyObsWrapper(env)
ob, _ = env.reset()
img = env.render()
plt.imshow(img)
plt.show()
