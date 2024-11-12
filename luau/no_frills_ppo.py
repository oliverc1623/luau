# %%
import logging
import random
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as f
from minigrid.wrappers import FullyObsWrapper
from PIL import Image
from torch import nn
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

from luau.iaa_env import SmallIntrospectiveEnv


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

    def __init__(self, horizon: int, num_envs: int, state: dict, action_space: gym.Space):
        # Storage setup
        self.horizon = horizon
        self.num_envs = num_envs
        self.state = state
        self.action_space = action_space

        sample = state["image"].sample()
        permuted_sample = np.transpose(sample, (2, 0, 1))
        self.img_shape = permuted_sample.shape

        self.images = torch.zeros(self.horizon, self.num_envs, *(3, 6, 6)).to(device)
        self.directions = torch.zeros((self.horizon, self.num_envs, 4)).to(device)
        self.actions = torch.zeros((self.horizon, self.num_envs, *self.action_space.shape)).to(device)
        self.logprobs = torch.zeros((self.horizon, self.num_envs)).to(device)
        self.rewards = torch.zeros((self.horizon, self.num_envs)).to(device)
        self.is_terminals = torch.zeros((self.horizon, self.num_envs)).to(device)
        self.state_values = torch.zeros((self.horizon, self.num_envs)).to(device)

    def clear(self) -> None:
        """Clear the buffer."""
        self.images = torch.zeros(self.horizon, self.num_envs, *(3, 6, 6)).to(device)
        self.directions = torch.zeros((self.horizon, self.num_envs, 4)).to(device)
        self.actions = torch.zeros((self.horizon, self.num_envs, *self.action_space.shape)).to(device)
        self.logprobs = torch.zeros((self.horizon, self.num_envs)).to(device)
        self.rewards = torch.zeros((self.horizon, self.num_envs)).to(device)
        self.is_terminals = torch.zeros((self.horizon, self.num_envs)).to(device)
        self.state_values = torch.zeros((self.horizon, self.num_envs)).to(device)


class ActorCritic(nn.Module):
    """Actor-Critic class. Only discrete action spaces are supported... for now."""

    def __init__(self, state_dim: torch.tensor, action_dim: int):
        super().__init__()
        self.actor_conv1 = self.layer_init(nn.Conv2d(state_dim, 16, 1))
        self.actor_conv2 = self.layer_init(nn.Conv2d(16, 32, 2))
        self.actor_conv3 = self.layer_init(nn.Conv2d(32, 64, 2))
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.actor_fc1 = self.layer_init(nn.Linear(580, action_dim), std=0.01)

        self.critic_conv1 = self.layer_init(nn.Conv2d(state_dim, 16, 1))
        self.critic_conv2 = self.layer_init(nn.Conv2d(16, 32, 2))
        self.critic_conv3 = self.layer_init(nn.Conv2d(32, 64, 2))
        self.critic_fc1 = self.layer_init(nn.Linear(580, 1), std=1.0)

    def layer_init(self, layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Module:
        """Initialize layer."""
        nn.init.orthogonal_(layer.weight, std)
        nn.init.constant_(layer.bias, bias_const)
        return layer

    def _actor_forward(self, image: torch.tensor, direction: torch.tensor) -> torch.tensor:
        """Run common computations for the actor network."""
        x = f.relu(self.actor_conv1(image))
        x = self.pool(x)
        x = f.relu(self.actor_conv2(x))
        x = f.relu(self.actor_conv3(x))
        x = x.reshape(x.size(0), -1)  # Flatten the tensor
        direction = direction.view(-1, 4)
        x = torch.cat((x, direction), 1)
        x = self.actor_fc1(x)
        return x

    def _critic_forward(self, image: torch.tensor, direction: torch.tensor) -> torch.tensor:
        """Run common computations for the critic network."""
        y = f.relu(self.critic_conv1(image))
        y = self.pool(y)
        y = f.relu(self.critic_conv2(y))
        y = f.relu(self.critic_conv3(y))
        y = y.reshape(y.size(0), -1)  # Flatten the tensor
        direction = direction.view(-1, 4)
        y = torch.cat((y, direction), 1)
        y = self.critic_fc1(y).squeeze(-1)
        return y

    def forward(self, state: dict) -> tuple[torch.tensor, torch.tensor, torch.tensor]:
        """Forward pass."""
        image = state["image"]
        direction = state["direction"]
        logits = self._actor_forward(image, direction)
        dist = Categorical(logits=logits)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_values = self._critic_forward(image, direction)
        return action, action_logprob, state_values

    def evaluate(self, states: dict, actions: torch.tensor) -> tuple[torch.tensor, torch.tensor, torch.tensor]:
        """Evaluate the policy."""
        images = states["image"]
        direction = states["direction"]
        logits = self._actor_forward(images, direction)
        dist = Categorical(logits=logits)
        action_logprobs = dist.log_prob(actions)
        dist_entropy = dist.entropy()
        state_values = self._critic_forward(images, direction)
        return action_logprobs, state_values, dist_entropy


def preprocess(x: dict) -> dict:
    """Preprocess the input for a grid-based environment, padding it to (12, 12, channels)."""
    direction = x["direction"]
    image = x["image"]
    image = torch.from_numpy(image).float()

    rgb = 3
    if image.ndim == rgb:  # Single image case with shape (height, width, channels)
        image = image.permute(2, 0, 1)
        # Permute back to (batch_size, channels, height, width)
        image = image.unsqueeze(0).to(device)  # Adding batch dimension
    else:  # Batch case with shape (batch_size, height, width, channels)
        image = image.permute(0, 3, 1, 2).to(device)  # Change to (batch, channels, height, width)
    direction = torch.tensor(direction, dtype=torch.int64).to(device)
    direction = f.one_hot(direction, num_classes=4)
    return {"image": image, "direction": direction}


def main() -> None:  # noqa: PLR0915
    """Run Main function."""
    # Initialize the PPO agent
    seed = 37
    horizon = 128
    num_envs = 10
    lr_actor = 0.0001
    max_training_timesteps = 500_000
    gamma = 0.99
    gae_lambda = 0.8
    eps_clip = 0.2
    minibatch_size = 128
    k_epochs = 4
    save_model_freq = 130
    run_num = 1
    door_locked = False
    save_frames = False

    # Initialize TensorBoard writer
    log_dir = Path(f"../../pvcvolume/PPO_logs/PPO/SmallIntrospectiveEnv-Unlocked/run_{run_num}_seed_{seed}")
    model_dir = Path(f"../../pvcvolume/models/PPO/SmallIntrospectiveEnv-Unlocked/run_{run_num}_seed_{seed}")
    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))
    checkpoint_path = f"{model_dir}/SmallIntrospectiveEnv-Unlocked_run_{run_num}_seed_{seed}.pth"
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

    def make_env(sub_env_seed: int) -> SmallIntrospectiveEnv:
        """Create the environment."""

        def _init() -> SmallIntrospectiveEnv:
            sub_env_rng = np.random.default_rng(sub_env_seed)
            env = SmallIntrospectiveEnv(rng=sub_env_rng, size=6, locked=door_locked, render_mode="rgb_array", max_steps=360)
            env = FullyObsWrapper(env)
            env.reset(seed=sub_env_seed)
            env.action_space.seed(sub_env_seed)
            env.observation_space.seed(sub_env_seed)
            return env

        return _init

    envs = [make_env(seed + i) for i in range(num_envs)]
    env = gym.vector.AsyncVectorEnv(envs, shared_memory=False)

    buffer = RolloutBuffer(horizon, num_envs, env.single_observation_space, env.single_action_space)
    state_dim = env.single_observation_space["image"].shape[-1]
    policy = ActorCritic(state_dim, env.single_action_space.n).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr_actor, eps=1e-5)

    next_obs, _ = env.reset()
    next_obs = preprocess(next_obs)
    next_dones = torch.zeros(num_envs).to(device)
    global_step = 0

    # Training loop
    num_updates = max_training_timesteps // (horizon * num_envs)
    for update in range(1, num_updates + 1):
        for step in range(horizon):
            if save_frames:
                img1, img2, img3, img4, img5 = env.render()
                concatenated_horizontally = np.concatenate((img1, img2, img3, img4, img5), axis=1)  # Along width
                array = concatenated_horizontally.astype(np.uint8)
                image = Image.fromarray(array)
                image.save(f"frames/{global_step}.png")

            # Preprocess the next observation and store relevant data in the PPO agent's buffer
            buffer.images[step] = next_obs["image"]
            buffer.directions[step] = next_obs["direction"]
            buffer.is_terminals[step] = next_dones

            with torch.no_grad():
                actions, log_probs, state_values = policy(next_obs)
                buffer.state_values[step] = state_values
            buffer.actions[step] = actions
            buffer.logprobs[step] = log_probs

            # Step the environment and store the rewards
            next_obs, rewards, next_dones, truncated, info = env.step(actions.tolist())
            next_obs = preprocess(next_obs)
            next_dones = torch.tensor(next_dones).to(device)
            truncated = torch.tensor(truncated, dtype=torch.float32).to(device)
            buffer.rewards[step] = torch.tensor(rewards, dtype=torch.float32).to(device).view(-1)

            global_step += 1 * num_envs
            if next_dones.any() or truncated.any():
                done_indx = torch.argmax(next_dones.int())
                writer.add_scalar("charts/Episodic Reward", rewards[done_indx], global_step)
                logging.info("i_update: %s, \t Timestep: %s, \t Reward: %s", update, global_step, rewards[done_indx])

        # Calculate rewards and advantages using GAE
        with torch.no_grad():
            _, _, next_value = policy(next_obs)
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
        b_returns = returns.reshape(-1).detach()
        b_advantages = advantages.reshape(-1).detach()
        b_actions = torch.flatten(buffer.actions, 0, 1).detach()
        b_logprobs = torch.flatten(buffer.logprobs, 0, 1).detach()
        b_images = torch.flatten(buffer.images, 0, 1).detach()
        b_directions = torch.flatten(buffer.directions, 0, 1).detach()
        b_state_values = torch.flatten(buffer.state_values, 0, 1).detach()

        batch_size = num_envs * horizon
        b_inds = np.arange(batch_size)
        clipfracs = []

        # Optimize policy for K epochs
        for _ in range(k_epochs):
            rng.shuffle(b_inds)

            # Split data into minibatches
            for i in range(0, batch_size, minibatch_size):
                end = i + minibatch_size
                mb_inds = b_inds[i:end]
                mb_states = {"image": b_images[mb_inds], "direction": b_directions[mb_inds]}
                new_logprob, new_value, dist_entropy = policy.evaluate(mb_states, b_actions.long()[mb_inds])

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

                # entropy loss
                entropy_loss_student = dist_entropy.mean()

                # final loss of clipped objective PPO
                student_loss = pg_loss_student - 0.01 * entropy_loss_student + v_loss_student * 0.5

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
