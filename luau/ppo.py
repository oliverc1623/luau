# %%
from pathlib import Path

import gymnasium
import numpy as np
import torch
import torch.nn.functional as f
from torch import nn
from torch.distributions import Bernoulli, Categorical
from torch.utils.tensorboard import SummaryWriter


RGB_CHANNEL = 3

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

    def __init__(self, horizon: int, num_envs: int, state: dict, action_space: gymnasium.Space):
        # Storage setup
        self.horizon = horizon
        self.num_envs = num_envs
        self.state = state
        self.action_space = action_space

        sample = state["image"].sample()
        permuted_sample = np.transpose(sample, (2, 0, 1))
        self.img_shape = permuted_sample.shape

        self.images = torch.zeros(self.horizon, self.num_envs, *self.img_shape).to(device)
        self.directions = torch.zeros((self.horizon, self.num_envs, *self.state["direction"].shape)).to(device)
        self.actions = torch.zeros((self.horizon, self.num_envs, *self.action_space.shape)).to(device)
        self.logprobs = torch.zeros((self.horizon, self.num_envs)).to(device)
        self.rewards = torch.zeros((self.horizon, self.num_envs)).to(device)
        self.is_terminals = torch.zeros((self.horizon, self.num_envs)).to(device)
        self.state_values = torch.zeros((self.horizon, self.num_envs)).to(device)

    def clear(self) -> None:
        """Clear the buffer."""
        self.images = torch.zeros(self.horizon, self.num_envs, *self.img_shape).to(device)
        self.directions = torch.zeros((self.horizon, self.num_envs, *self.state["direction"].shape)).to(device)
        self.actions = torch.zeros((self.horizon, self.num_envs, *self.action_space.shape)).to(device)
        self.logprobs = torch.zeros((self.horizon, self.num_envs)).to(device)
        self.rewards = torch.zeros((self.horizon, self.num_envs)).to(device)
        self.is_terminals = torch.zeros((self.horizon, self.num_envs)).to(device)
        self.state_values = torch.zeros((self.horizon, self.num_envs)).to(device)


class ActorCritic(nn.Module):
    """Actor-Critic class. Only discrete action spaces are supported... for now."""

    def __init__(self, state_dim: torch.tensor, action_dim: int):
        super().__init__()

        # actor conv layers
        # TODO: should probably turn into a Sequential model
        self.actor_conv1 = self.layer_init(nn.Conv2d(state_dim, 16, 2))
        self.actor_conv2 = self.layer_init(nn.Conv2d(16, 32, 2))
        self.actor_conv3 = self.layer_init(nn.Conv2d(32, 64, 2))

        # actor linear layers
        self.actor_fc1 = self.layer_init(nn.Linear(65, 512))
        self.actor_fc2 = self.layer_init(nn.Linear(512, action_dim), std=0.01)

        # critic conv layers
        # TODO: should probably turn into a Sequential model
        self.critic_conv1 = self.layer_init(nn.Conv2d(state_dim, 16, 2))
        self.critic_conv2 = self.layer_init(nn.Conv2d(16, 32, 2))
        self.critic_conv3 = self.layer_init(nn.Conv2d(32, 64, 2))

        # critic linear layers
        self.critic_fc1 = self.layer_init(nn.Linear(65, 512))
        self.critic_fc2 = self.layer_init(nn.Linear(512, 1), std=1.0)

    def layer_init(self, layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Module:
        """Initialize layer."""
        nn.init.orthogonal_(layer.weight, std)
        nn.init.constant_(layer.bias, bias_const)
        return layer

    def _initialize_weights(self) -> None:
        # Orthogonal initialization of conv and linear layers
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d | nn.Linear):
                nn.init.orthogonal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0.0)  # Set bias to 0, or any constant value you prefer

    def _actor_forward(self, image: torch.tensor, direction: torch.tensor) -> torch.tensor:
        """Run common computations for the actor network."""
        x = f.relu(self.actor_conv1(image))
        x = f.max_pool2d(x, 2)
        x = f.relu(self.actor_conv2(x))
        x = f.relu(self.actor_conv3(x))
        x = torch.flatten(x, 1)
        direction = direction.view(-1, 1)
        x = torch.cat((x, direction), 1)
        x = f.relu(self.actor_fc1(x))
        return self.actor_fc2(x)

    def _critic_forward(self, image: torch.tensor, direction: torch.tensor) -> torch.tensor:
        """Run common computations for the critic network."""
        y = f.relu(self.critic_conv1(image))
        y = f.max_pool2d(y, 2)
        y = f.relu(self.critic_conv2(y))
        y = f.relu(self.critic_conv3(y))
        y = torch.flatten(y, 1)
        y = torch.cat((y, direction.view(-1, 1)), 1)
        y = f.relu(self.critic_fc1(y))
        state_values = self.critic_fc2(y).squeeze(-1)
        return state_values

    def forward(self, state: dict) -> tuple[torch.tensor, torch.tensor, torch.tensor]:
        """Forward pass."""
        direction = state["direction"]
        image = state["image"]

        # actor
        logits = self._actor_forward(image, direction)
        dist = Categorical(logits=logits)
        action = dist.sample()
        action_logprob = dist.log_prob(action)

        # critic
        state_values = self._critic_forward(image, direction)

        return action, action_logprob, state_values

    def evaluate(self, states: dict, actions: torch.tensor) -> tuple[torch.tensor, torch.tensor, torch.tensor]:
        """Evaluate the policy."""
        images = states["image"]
        directions = states["direction"]

        # actor
        logits = self._actor_forward(images, directions)
        dist = Categorical(logits=logits)
        action_logprobs = dist.log_prob(actions)
        dist_entropy = dist.entropy()

        # critic
        state_values = self._critic_forward(images, directions)

        return action_logprobs, state_values, dist_entropy


class PPO:
    """Proximal Policy Optimization (PPO) agent."""

    def __init__(
        self,
        state_dim: torch.tensor,
        action_dim: int,
        lr_actor: float,
        gamma: float,
        k_epochs: int,
        eps_clip: float,
        minibatch_size: int,
        env: gymnasium.Env,
        horizon: int,
        num_envs: int,
        gae_lambda: float,
    ):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.minibatch_size = minibatch_size
        self.buffer = RolloutBuffer(horizon, num_envs, env.single_observation_space, env.single_action_space)
        self.policy = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr_actor, eps=1e-5)
        self.MseLoss = nn.MSELoss()
        self.env = env
        self.horizon = horizon
        self.num_envs = num_envs
        self.gae_lambda = gae_lambda

    def _calculate_gae(self, next_obs: torch.tensor, next_done: torch.tensor) -> tuple[torch.tensor, torch.tensor]:
        # Generalized Advantage Estimation (GAE)
        with torch.no_grad():
            advantages = torch.zeros(self.horizon, self.num_envs).to(device)
            rewards = torch.zeros(self.horizon, self.num_envs).to(device)
            advantages = torch.zeros_like(rewards).to(device)
            next_done = torch.tensor(next_done).to(device)
            lastgaelam = 0

            # Get the value of the next observation for bootstrapping
            next_obs = self.preprocess(next_obs)
            _, _, next_value = self.policy(next_obs)

            for step in reversed(range(self.horizon)):
                if step == self.horizon - 1:
                    next_non_terminal = 1.0 - next_done.float()
                    nextvalues = next_value  # Bootstrapping for the last value
                else:
                    next_non_terminal = 1.0 - self.buffer.is_terminals[step + 1]
                    nextvalues = self.buffer.state_values[step + 1]

                # Temporal difference error
                delta = self.buffer.rewards[step] + self.gamma * nextvalues * next_non_terminal - self.buffer.state_values[step]
                advantages[step] = lastgaelam = delta + self.gamma * self.gae_lambda * next_non_terminal * lastgaelam

            rewards = advantages + self.buffer.state_values
            return rewards, advantages

    def update(self, next_obs: torch.tensor, next_done: torch.tensor, writer: SummaryWriter, rollout_step: int) -> None:
        """Update the policy."""
        # Calculate rewards and advantages using GAE
        rewards, advantages = self._calculate_gae(next_obs, next_done)

        # Prepare data for minibatch shuffling
        b_rewards = torch.flatten(rewards, 0, 1)
        b_advantages = torch.flatten(advantages, 0, 1)
        b_actions = torch.flatten(self.buffer.actions, 0, 1)
        b_logprobs = torch.flatten(self.buffer.logprobs, 0, 1)
        b_images = torch.flatten(self.buffer.images, 0, 1)
        b_directions = torch.flatten(self.buffer.directions, 0, 1)
        b_state_values = torch.flatten(self.buffer.state_values, 0, 1)

        batch_size = self.num_envs * self.horizon
        b_inds = np.arange(batch_size)
        clipfracs = []

        # Optimize policy for K epochs
        for _ in range(self.k_epochs):
            # Shuffle the data for each epoch
            np.random.shuffle(b_inds)  # noqa: NPY002

            # Split data into minibatches
            for i in range(0, batch_size, self.minibatch_size):
                end = i + self.minibatch_size
                mb_inds = b_inds[i:end]

                states_mb = {"image": b_images[mb_inds], "direction": b_directions[mb_inds]}
                # Evaluating old actions and values
                logprobs, state_values, dist_entropy = self.policy.evaluate(states_mb, b_actions.long()[mb_inds])

                # policy gradient
                log_ratio = logprobs - b_logprobs[mb_inds]
                ratios = log_ratio.exp()  # Finding the ratio (pi_theta / pi_theta__old)
                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-log_ratio).mean()
                    approx_kl = ((ratios - 1) - log_ratio).mean()
                    clipfracs += [((ratios - 1.0).abs() > self.eps_clip).float().mean().item()]
                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                surr1 = -mb_advantages * ratios
                surr2 = -mb_advantages * torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip)
                pg_loss = torch.max(surr1, surr2).mean()

                # value function loss + clipping
                v_loss_unclipped = (state_values - b_rewards[mb_inds]) ** 2
                v_clipped = b_state_values[mb_inds] + torch.clamp(state_values - b_state_values[mb_inds], -10.0, 10.0)
                v_loss_clipped = (v_clipped - b_rewards[mb_inds]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()

                # entropy loss
                entropy_loss = dist_entropy.mean()

                # final loss of clipped objective PPO
                loss = pg_loss - 0.01 * entropy_loss + v_loss * 0.5  # final loss of clipped objective PPO

                self.optimizer.zero_grad()  # take gradient step
                loss.backward()
                self.optimizer.step()

        # log debug variables
        with torch.no_grad():
            writer.add_scalar("debugging/policy_loss", pg_loss, rollout_step)
            writer.add_scalar("debugging/value_loss", v_loss, rollout_step)
            writer.add_scalar("debugging/entropy_loss", entropy_loss, rollout_step)
            writer.add_scalar("debugging/old_approx_kl", old_approx_kl.item(), rollout_step)
            writer.add_scalar("debugging/approx_kl", approx_kl.item(), rollout_step)
            writer.add_scalar("debugging/clipfrac", np.mean(clipfracs), rollout_step)

        self.buffer.clear()  # clear buffer

    def save(self, checkpoint_path: Path) -> None:
        """Save the model."""
        torch.save(self.policy.state_dict(), checkpoint_path)

    def load(self, checkpoint_path: Path) -> None:
        """Load the model."""
        self.policy.load_state_dict(torch.load(checkpoint_path))

    def preprocess(self, x: dict) -> torch.tensor:
        """Preprocess the input."""
        direction = x["direction"]
        image = x["image"]
        image = torch.from_numpy(image).float()
        if len(image.shape) == RGB_CHANNEL:
            image = image.unsqueeze(0).to(device)
        else:
            image = image.permute(0, 3, 1, 2).to(device)
        direction = torch.tensor(direction, dtype=torch.float).unsqueeze(0).to(device)
        x = {"direction": direction, "image": image}
        return x


# %% V################################## Introspective PPO ##################################


class IAARolloutBuffer(RolloutBuffer):
    """A buffer to store rollout data for Introspective Action Advising (IAA)."""

    def __init__(self):
        super().__init__()
        self.indicators = torch.zeros((self.horizon, self.num_envs)).to(device)

    def clear(self) -> None:
        """Clear the buffer."""
        super().clear()
        self.indicators = torch.zeros((self.horizon, self.num_envs)).to(device)


class IAAPPO(PPO):
    """PPO agent for the IAA."""

    def __init__(self, teacher_ppo_agent: PPO, *args: dict, **kwargs: dict):
        super().__init__(*args, **kwargs)
        self.buffer = IAARolloutBuffer()
        self.introspection_decay = kwargs.get("introspection_decay", 0.99999)
        self.burn_in = kwargs.get("burn_in", 0)
        self.inspection_threshold = kwargs.get("inspection_threshold", 0.9)
        self.teacher_ppo_agent = teacher_ppo_agent
        if self.teacher_ppo_agent is None:
            raise ValueError("Teacher agent is None. Please specify pth model.")

    def introspect(self, t: int, state: dict) -> int:
        """Introspect."""
        probability = self.introspection_decay ** (max(0, t - self.burn_in))
        p = Bernoulli(probability).sample()
        if t > self.burn_in and p == 1:
            _, _, teacher_source_val = self.teacher_ppo_agent.policy_old(state)
            _, _, teacher_target_val = self.teacher_ppo_agent.policy(state)
            return int(abs(teacher_target_val - teacher_source_val) <= self.inspection_threshold)
        return 0

    def select_action(self, state: dict, t: int) -> int:
        """Select an action."""
        state = self.preprocess(state)
        h = self.introspect(t, state)
        if h:
            with torch.no_grad():
                action, action_logprob, state_val = self.teacher_ppo_agent.policy_old(state)
        else:
            with torch.no_grad():
                action, action_logprob, state_val = self.policy_old(state)
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_val)
        self.buffer.indicators.append(h)
        return action.item()

    def correct(self) -> tuple[torch.tensor, torch.tensor]:
        """Apply off-policy correction."""
        teacher_ratios = []
        student_ratios = []
        for state, indicator, logprob in zip(self.buffer.states, self.buffer.indicators, self.buffer.logprobs, strict=False):
            if indicator:
                # compute importance sampling ratio
                _, student_action_logprob, _ = self.policy_old(state)
                ratio = student_action_logprob / logprob
                teacher_ratios.append(1.0)
                student_ratios.append(torch.clamp(ratio, -2, 2).item())

            else:
                # compute importance sampling ratio
                _, teacher_action_logprob, _ = self.teacher_ppo_agent.policy_old(state)
                ratio = teacher_action_logprob / logprob
                teacher_ratios.append(torch.clamp(ratio, -0.2, 0.2).item())
                student_ratios.append(1.0)

        teacher_ratios = torch.tensor(teacher_ratios).float()
        student_ratios = torch.tensor(student_ratios).float()
        return teacher_ratios, student_ratios

    def update(self) -> None:
        """Update the policy with student correction."""
        teacher_correction, student_correction = self.correct()
        self._common_update(teacher_correction, student_correction)
        self._common_update(teacher_correction)
        self.buffer.clear()

    def _common_update(self, teacher_correction: torch.tensor, student_correction: torch.tensor_split = None) -> None:
        """Update logic for both update and update_critic."""
        rewards = self._calculate_rewards()
        old_actions, old_logprobs, old_state_values = self._tensorize_rollout_buffer()

        # Calculate advantages
        advantages = rewards.detach() - old_state_values.detach()

        # Optimize policy for K epochs
        for _ in range(self.k_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(self.buffer.states, old_actions)
            state_values = torch.squeeze(state_values)

            # Determine which correction to apply
            correction = student_correction if student_correction is not None else teacher_correction

            # Calculate ratios and loss
            ratios = torch.exp(logprobs - old_logprobs.detach()) * correction.to(device)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            vf_loss = self.MseLoss(state_values, rewards)

            # Final loss computation
            loss = -torch.min(surr1, surr2) + 0.5 * vf_loss - 0.01 * dist_entropy
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        if student_correction is not None:
            self.policy_old.load_state_dict(self.policy.state_dict())
