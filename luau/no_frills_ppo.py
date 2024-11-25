# %%
import logging
import random
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from minigrid.wrappers import FullyObsWrapper, ImgObsWrapper
from PIL import Image
from stable_baselines3.common.vec_env import SubprocVecEnv
from torch import nn
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

from luau.iaa_env import IntrospectiveEnv


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Get the root path
root_path = Path(__file__).resolve().parent.parent

# %%
RGB_CHANNEL = 3
KL_THRESHOLD = 0.01

OBJECT_IDX_EMPTY = 1
COLOR_IDX_EMPTY = 5  # Or another value if different in your environment
STATE_EMPTY = 0

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
################################## PPO Policy ##################################
class RolloutBuffer:
    """A buffer to store rollout data for reinforcement learning agents and supports the generation of minibatches for training."""

    def __init__(self, horizon: int, num_envs: int, observation_shape: tuple, action_space: gym.Space):
        # Storage setup
        self.horizon = horizon
        self.num_envs = num_envs
        self.observation_shape = observation_shape
        self.action_space = action_space

        self.images = torch.zeros(self.horizon, self.num_envs, *self.observation_shape).to(device)
        self.actions = torch.zeros((self.horizon, self.num_envs, *self.action_space.shape)).to(device)
        self.logprobs = torch.zeros((self.horizon, self.num_envs)).to(device)
        self.rewards = torch.zeros((self.horizon, self.num_envs)).to(device)
        self.is_terminals = torch.zeros((self.horizon, self.num_envs)).to(device)
        self.state_values = torch.zeros((self.horizon, self.num_envs)).to(device)

    def clear(self) -> None:
        """Clear the buffer."""
        self.images = torch.zeros(self.horizon, self.num_envs, *self.observation_shape).to(device)
        self.actions = torch.zeros((self.horizon, self.num_envs, *self.action_space.shape)).to(device)
        self.logprobs = torch.zeros((self.horizon, self.num_envs)).to(device)
        self.rewards = torch.zeros((self.horizon, self.num_envs)).to(device)
        self.is_terminals = torch.zeros((self.horizon, self.num_envs)).to(device)
        self.state_values = torch.zeros((self.horizon, self.num_envs)).to(device)


class ActorCritic(nn.Module):
    """Actor-Critic class supporting discrete action spaces."""

    def __init__(self, state_dim: torch.tensor, action_dim: int):
        super().__init__()

        # Actor network
        self.actor = nn.Sequential(
            self.layer_init(nn.Conv2d(state_dim, 16, 3, stride=1, padding=1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            self.layer_init(nn.Conv2d(16, 32, 3, stride=1, padding=1)),
            nn.ReLU(),
            self.layer_init(nn.Conv2d(32, 64, 3, stride=1, padding=1)),
            nn.ReLU(),
            nn.Flatten(),
            self.layer_init(nn.Linear(576, 128)),
            nn.ReLU(),
            self.layer_init(nn.Linear(128, action_dim), std=0.01),
        )

        # Critic network
        self.critic = nn.Sequential(
            self.layer_init(nn.Conv2d(state_dim, 16, 3, stride=1, padding=1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            self.layer_init(nn.Conv2d(16, 32, 3, stride=1, padding=1)),
            nn.ReLU(),
            self.layer_init(nn.Conv2d(32, 64, 3, stride=1, padding=1)),
            nn.ReLU(),
            nn.Flatten(),
            self.layer_init(nn.Linear(576, 128)),
            nn.ReLU(),
            self.layer_init(nn.Linear(128, 1), std=1.0),
        )

    def layer_init(self, layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Module:
        """Initialize layer."""
        nn.init.orthogonal_(layer.weight, std)
        nn.init.constant_(layer.bias, bias_const)
        return layer

    def get_value(self, image: torch.tensor) -> torch.tensor:
        """Get the value of the state."""
        return self.critic(image)

    def forward(self, image: torch.tensor, action: torch.tensor = None) -> tuple[torch.tensor, torch.tensor, torch.tensor]:
        """Forward pass. Optionally takes an action for evaluation."""
        logits = self.actor(image)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_values = self.critic(image)
        return action, action_logprob, state_values, dist.entropy()


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


def largest_divisor(n: int) -> int:
    """Find the largest divisor of batch_size that is less than or equal to horizon."""
    for i in range(n // 2, 0, -1):
        if n % i == 0:
            return i
    return 1  # If no divisors found, return 1


def main() -> None:  # noqa: PLR0915
    """Run Main function."""
    # Initialize the PPO agent
    seed = 17
    horizon = 128
    num_envs = 2
    batch_size = num_envs * horizon
    lr_actor = 0.0005
    max_training_timesteps = 500_000
    num_updates = max_training_timesteps // (horizon * num_envs)
    gamma = 0.99
    gae_lambda = 0.8
    eps_clip = 0.2
    k_epochs = 4
    minibatch_size = batch_size // k_epochs
    save_model_freq = largest_divisor(num_updates)
    run_num = 1
    save_frames = False
    env_name = "MiniGrid-LavaGapS6-v0"

    # Initialize TensorBoard writer
    log_dir = Path(f"../../pvcvolume/PPO_logs/PPO/{env_name}/run-{run_num}-seed-{seed}")
    model_dir = Path(f"../../pvcvolume/models/PPO/{env_name}/run-{run_num}-seed-{seed}")
    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))
    checkpoint_path = f"{model_dir}/{env_name}-run-{run_num}-seed-{seed}.pth"
    print(f"Logging to: {log_dir}")
    print(f"Saving to: {checkpoint_path}")

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior in PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    rng = np.random.default_rng(seed)

    def make_env() -> gym.Env:
        """Create the environment."""

        def _init() -> gym.Env:
            env = IntrospectiveEnv(size=9, locked=False, render_mode="rgb_array")
            env = FullyObsWrapper(env)
            env = ImgObsWrapper(env)
            return env

        return _init

    envs = [make_env(seed + i) for i in range(num_envs)]
    env = SubprocVecEnv(envs)
    env.seed(seed=seed)

    observation_shape = np.transpose(env.observation_space.sample(), (2, 0, 1)).shape
    buffer = RolloutBuffer(horizon, num_envs, observation_shape, env.action_space)
    state_dim = env.observation_space.shape[-1]
    policy = ActorCritic(state_dim, env.action_space.n).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr_actor, eps=1e-5)

    next_obs = env.reset()
    next_obs = preprocess(next_obs)
    next_dones = torch.zeros(num_envs).to(device)
    global_step = 0

    # Training loop
    for update in range(1, num_updates + 1):
        for step in range(horizon):
            if save_frames:
                img1, img2, img3, img4, img5 = env.render()
                concatenated_horizontally = np.concatenate((img1, img2, img3, img4, img5), axis=1)  # Along width
                array = concatenated_horizontally.astype(np.uint8)
                image = Image.fromarray(array)
                image.save(f"frames/{global_step}.png")

            # Preprocess the next observation and store relevant data in the PPO agent's buffer
            buffer.images[step] = next_obs
            buffer.is_terminals[step] = next_dones

            with torch.no_grad():
                actions, log_probs, state_values, _ = policy(next_obs)
                buffer.state_values[step] = state_values.flatten()
            buffer.actions[step] = actions
            buffer.logprobs[step] = log_probs

            # Step the environment and store the rewards
            next_obs, rewards, next_dones, info = env.step(actions.tolist())
            next_obs = preprocess(next_obs)
            next_dones = torch.tensor(next_dones).to(device)
            buffer.rewards[step] = torch.tensor(rewards, dtype=torch.float32).to(device).view(-1)

            global_step += 1 * num_envs
            if next_dones.any():
                done_indx = torch.argmax(next_dones.int())
                writer.add_scalar("charts/Episodic Reward", rewards[done_indx], global_step)
                logging.info("i_update: %s, \t Timestep: %s, \t Reward: %s", update, global_step, rewards[done_indx])

        # Calculate rewards and advantages using GAE
        with torch.no_grad():
            next_value = policy.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(buffer.rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(horizon)):
                if t == horizon - 1:
                    next_non_terminal = 1.0 - next_dones.float()
                    nextvalues = next_value  # Bootstrapping for the last value
                else:
                    next_non_terminal = 1.0 - buffer.is_terminals[t + 1]
                    nextvalues = buffer.state_values[t + 1]

                # Temporal difference error
                delta = buffer.rewards[t] + gamma * nextvalues * next_non_terminal - buffer.state_values[t]
                advantages[t] = lastgaelam = delta + gamma * gae_lambda * next_non_terminal * lastgaelam
            returns = advantages + buffer.state_values

        # Flatten the buffer
        b_images = buffer.images.reshape((-1, *observation_shape))
        b_logprobs = buffer.logprobs.reshape(-1)
        b_actions = buffer.actions.reshape((-1, *env.action_space.shape))
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_state_values = buffer.state_values.reshape(-1)

        b_inds = np.arange(batch_size)
        clipfracs = []

        # Optimize policy for K epochs
        for _ in range(k_epochs):
            rng.shuffle(b_inds)

            # Split data into minibatches
            for i in range(0, batch_size, minibatch_size):
                end = i + minibatch_size
                mb_inds = b_inds[i:end]
                mb_images = b_images[mb_inds]
                _, new_logprob, new_value, dist_entropy = policy(mb_images, b_actions.long()[mb_inds])

                # policy gradient
                log_ratio = new_logprob - b_logprobs[mb_inds].detach()
                ratios = torch.exp(log_ratio)  # Finding the ratio (pi_theta / pi_theta__old)

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-log_ratio).mean()
                    approx_kl = ((ratios - 1) - log_ratio).mean()
                    clipfracs += [((ratios - 1.0).abs() > eps_clip).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                surr1 = -mb_advantages * ratios
                surr2 = -mb_advantages * torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip)
                pg_loss_student = torch.max(surr1, surr2).mean()

                # value function loss + clipping
                new_value = new_value.view(-1)
                v_loss_unclipped = (new_value - b_returns[mb_inds]) ** 2
                v_clipped = b_state_values[mb_inds] + torch.clamp(new_value - b_state_values[mb_inds], -10.0, 10.0)
                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss_student = 0.5 * v_loss_max.mean()

                entropy_loss_student = dist_entropy.mean()

                # final loss of clipped objective PPO
                student_loss = pg_loss_student - 0.05 * entropy_loss_student + v_loss_student * 0.5

                optimizer.zero_grad()  # take gradient step
                student_loss.backward()
                optimizer.step()

        # log debug variables
        with torch.no_grad():
            writer.add_scalar("debugging/policy_loss", pg_loss_student.item(), global_step)
            writer.add_scalar("debugging/value_loss", v_loss_student.item(), global_step)
            writer.add_scalar("debugging/entropy_loss", entropy_loss_student.item(), global_step)
            writer.add_scalar("debugging/old_approx_kl", old_approx_kl.item(), global_step)
            writer.add_scalar("debugging/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("debugging/clipfrac", np.mean(clipfracs), global_step)

        if update % save_model_freq == 0:
            logging.info("--------------------------------------------------------------------------------------------")
            logging.info("Saving model to: %s", checkpoint_path)
            logging.info("--------------------------------------------------------------------------------------------")
            torch.save(policy.state_dict(), checkpoint_path)

    env.close()
    writer.close()


if __name__ == "__main__":
    main()
