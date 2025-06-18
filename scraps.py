# %% Sac Protagonist Eval

import math
import os

import gymnasium as gym
import torch
import torch.nn.functional as f
import wandb
from tensordict import TensorDict, from_modules
from torch import nn


os.environ["MUJOCO_GL"] = "egl"  # must precede any mujoco/gym import

# %%
LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    """A neural network that approximates the policy for reinforcement learning."""

    def __init__(self, env: gym.Env, n_obs: int, n_act: int, device: str | None = None):
        super().__init__()
        self.fc1 = nn.Linear(n_obs, 256, device=device)
        self.fc2 = nn.Linear(256, 256, device=device)
        self.fc_mean = nn.Linear(256, n_act, device=device)
        self.fc_logstd = nn.Linear(256, n_act, device=device)
        # action rescaling
        action_space = env.action_space
        self.register_buffer(
            "action_scale",
            torch.tensor((action_space.high - action_space.low) / 2.0, dtype=torch.float32, device=device),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor((action_space.high + action_space.low) / 2.0, dtype=torch.float32, device=device),
        )

    def forward(self, x: torch.tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        x = x.view(x.shape[0], -1)  # flatten the input
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x: torch.tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get action from the policy network."""
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


class SoftQNetwork(nn.Module):
    """A neural network that approximates the soft Q-function for reinforcement learning."""

    def __init__(self, env: gym.Env, n_act: int, n_obs: int, device: str | None = None):  # noqa: ARG002
        super().__init__()
        self.fc1 = nn.Linear(n_act + n_obs, 256, device=device)
        self.fc2 = nn.Linear(256, 256, device=device)
        self.fc3 = nn.Linear(256, 1, device=device)

    def forward(self, x: torch.tensor, a: torch.tensor) -> torch.Tensor:
        """Forward pass."""
        x = torch.cat([x, a], 1)
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def get_q_params() -> tuple[TensorDict, TensorDict, SoftQNetwork]:
    """Initialize the Q-function parameters and target network."""
    qf1 = SoftQNetwork(env, device=device, n_act=n_act, n_obs=n_obs)
    qf2 = SoftQNetwork(env, device=device, n_act=n_act, n_obs=n_obs)
    qnet_params = from_modules(qf1, qf2, as_module=True)
    qnet_target = qnet_params.data.clone()

    # discard params of net
    qnet = SoftQNetwork(env, device="meta", n_act=n_act, n_obs=n_obs)
    qnet_params.to_module(qnet)

    return qnet_params, qnet_target, qnet


# %%
seed = 1
num_envs = 1
run_id = "v25b2n08"
env_id = "BipedalWalker-v3"
env_kwargs = {"hardcore": True, "render_mode": "rgb_array"}

env = gym.make(env_id, **env_kwargs)
env = gym.wrappers.RecordVideo(env, f"luau/videos/inference/{env_id}/{run_id}")

n_act = math.prod(env.action_space.shape)
n_obs = math.prod(env.observation_space.shape)
device = "cuda" if torch.cuda.is_available() else "cpu"

# %%
wandb.login(key="82555a3ad6bd991b8c4019a5a7a86f61388f6df1")
api = wandb.Api()

# %%

pretrained_run = api.run(f"luau/{run_id}")
actor_file = next(f.name for f in pretrained_run.files() if f.name.endswith("_actor.pt"))
qnet_file = next(f.name for f in pretrained_run.files() if f.name.endswith("_qnet.pt"))
actor_model_weights = wandb.restore(actor_file, run_path=f"luau/{run_id}")
qnet_model_weights = wandb.restore(qnet_file, run_path=f"luau/{run_id}")

# %%
protagonist = Actor(env, device=device, n_act=n_act, n_obs=n_obs)
protagonist.load_state_dict(
    torch.load(actor_model_weights.name),
)

# %%

student_qnet_params, student_qnet_target, student_qnet = get_q_params()
student_qnet_target.copy_(student_qnet_params.data)

# teacher new (tn) - to be trained
pretrained_qnet_tensordict = torch.load(qnet_model_weights.name, map_location=device)

tn_qnet_params, tn_qnet_target, tn_qnet = get_q_params()
tn_qnet_params.copy_(pretrained_qnet_tensordict)

# teacher source (ts) - not to be trained
ts_qnet_params, ts_qnet_target, ts_qnet = get_q_params()
ts_qnet_params.copy_(tn_qnet_params)
tn_qnet_params.requires_grad_(False)  # noqa: FBT003

# %%

tn_qnet_params["fc1", "weight"]

# %%

ts_qnet_params["fc1", "weight"]

# %%
