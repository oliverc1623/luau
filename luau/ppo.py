# %%
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as f
from torch import nn
from torch.distributions import Categorical


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

    def __init__(self):
        self.actions = []
        self.direction = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []

    def clear(self) -> None:
        """Clear the buffer."""
        del self.actions[:]
        del self.states[:]
        del self.direction[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]

    def generate_minibatch(self, minibatch_size: int = 128) -> tuple:
        """Generate a minibatch of data from the buffer."""
        batch_size = len(self.states)  # Assuming all lists are of the same size
        indices = np.random.default_rng().choice(batch_size, minibatch_size, replace=False)

        minibatch_states = [self.states[i] for i in indices]
        minibatch_direction = [self.direction[i] for i in indices]
        minibatch_actions = [self.actions[i] for i in indices]
        minibatch_logprobs = [self.logprobs[i] for i in indices]
        minibatch_rewards = [self.rewards[i] for i in indices]
        minibatch_state_values = [self.state_values[i] for i in indices]
        minibatch_is_terminals = [self.is_terminals[i] for i in indices]

        return (
            minibatch_states,
            minibatch_direction,
            minibatch_actions,
            minibatch_logprobs,
            minibatch_rewards,
            minibatch_state_values,
            minibatch_is_terminals,
        )


class ActorCritic(nn.Module):
    """Actor-Critic class. Only discrete action spaces are supported... for now."""

    def __init__(self, state_dim: torch.tensor, action_dim: int):
        super().__init__()

        # actor conv layers
        self.actor_conv1 = nn.Conv2d(state_dim, 16, 2)
        self.actor_conv2 = nn.Conv2d(16, 32, 2)
        self.actor_conv3 = nn.Conv2d(32, 64, 2)

        # actor linear layers
        self.actor_fc1 = nn.Linear(65, 512)
        self.actor_fc2 = nn.Linear(512, action_dim)

        # critic conv layers
        self.critic_conv1 = nn.Conv2d(state_dim, 16, 2)
        self.critic_conv2 = nn.Conv2d(16, 32, 2)
        self.critic_conv3 = nn.Conv2d(32, 64, 2)

        # critic linear layers
        self.critic_fc1 = nn.Linear(65, 512)  # Add +1 for the scalar input
        self.critic_fc2 = nn.Linear(512, 1)

    def forward(self, state: dict) -> tuple[torch.tensor, torch.tensor, torch.tensor]:
        """Forward pass."""
        direction = state["direction"]
        image = state["image"]

        # actor
        x = f.relu(self.actor_conv1(image))
        x = f.max_pool2d(x, 2)
        x = f.relu(self.actor_conv2(x))
        x = f.relu(self.actor_conv3(x))
        x = torch.flatten(x, 1)  # Flatten the output for the linear layer
        direction = direction.view(-1, 1)  # Reshape scalar to [batch_size, 1] if it's not already
        x = torch.cat((x, direction), 1)  # Concatenate the scalar with the flattened conv output
        x = f.relu(self.actor_fc1(x))
        action_probs = f.softmax(self.actor_fc2(x), dim=-1)[0]
        dist = Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)

        # critic
        y = f.relu(self.critic_conv1(image))
        y = f.max_pool2d(y, 2)
        y = f.relu(self.critic_conv2(y))
        y = f.relu(self.critic_conv3(y))
        y = torch.flatten(y, 1)  # Flatten the output for the linear layer
        y = torch.cat((y, direction), 1)  # Concatenate the scalar with the flattened conv output
        y = f.relu(self.critic_fc1(y))
        state_values = self.critic_fc2(y)

        return action.detach(), action_logprob.detach(), state_values.detach()

    def evaluate(self, state: torch.tensor, action: torch.tensor) -> tuple[torch.tensor, torch.tensor, torch.tensor]:
        """Evaluate the policy."""
        direction = state["direction"]
        image = state["image"]

        # actor
        x = f.relu(self.actor_conv1(image))
        x = f.max_pool2d(x, 2)
        x = f.relu(self.actor_conv2(x))
        x = f.relu(self.actor_conv3(x))
        x = torch.flatten(x, 1)  # Flatten the output for the linear layer
        scalar = direction.view(-1, 1)  # Reshape scalar to [batch_size, 1] if it's not already
        x = torch.cat((x, scalar), 1)  # Concatenate the scalar with the flattened conv output
        x = f.relu(self.actor_fc1(x))
        action_probs = f.softmax(self.actor_fc2(x), dim=-1)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        # critic
        y = f.relu(self.critic_conv1(image))
        y = f.max_pool2d(y, 2)
        y = f.relu(self.critic_conv2(y))
        y = f.relu(self.critic_conv3(y))
        y = torch.flatten(y, 1)  # Flatten the output for the linear layer
        y = torch.cat((y, scalar), 1)  # Concatenate the scalar with the flattened conv output
        y = f.relu(self.critic_fc1(y))
        state_values = self.critic_fc2(y)

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
        action_std_init: float = 0.6,
    ):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.buffer = RolloutBuffer()
        self.policy = ActorCritic(state_dim, action_dim, action_std_init).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr_actor)
        self.policy_old = ActorCritic(state_dim, action_dim, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()

    def select_action(self, state: dict) -> int:
        """Select an action."""
        direction = state["direction"]
        image = state["image"]
        with torch.no_grad():
            image = self.preprocess(image).to(device)
            direction = torch.tensor(direction, dtype=torch.float).unsqueeze(0).to(device)
            action, action_logprob, state_val = self.policy_old.act(state, direction)
        self.buffer.states.append(state)
        self.buffer.direction.append(direction)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_val)
        return action.item()

    def update(self) -> None:
        """Update the policy."""
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals), strict=False):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_direction = torch.squeeze(torch.stack(self.buffer.direction, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device)
        # calculate advantages
        advantages = rewards.detach() - old_state_values.detach()
        # Optimize policy for K epochs
        for _ in range(self.k_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions, old_direction)
            state_values = torch.squeeze(state_values)  # match state_values tensor dimensions with rewards tensor
            ratios = torch.exp(logprobs - old_logprobs.detach())  # Finding the ratio (pi_theta / pi_theta__old)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            vf_loss = self.MseLoss(state_values, rewards)  # value function loss
            loss = -torch.min(surr1, surr2) + 0.5 * vf_loss - 0.01 * dist_entropy  # final loss of clipped objective PPO
            self.optimizer.zero_grad()  # take gradient step
            loss.mean().backward()
            self.optimizer.step()
        self.policy_old.load_state_dict(self.policy.state_dict())  # Copy new weights into old policy
        self.buffer.clear()  # clear buffer

    def save(self, checkpoint_path: Path) -> None:
        """Save the model."""
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path: Path) -> None:
        """Load the model."""
        self.policy_old.load_state_dict(torch.load(checkpoint_path))
        self.policy.load_state_dict(torch.load(checkpoint_path))

    def preprocess(self, x: np.array, *, image_observation: bool = False, invert: bool = False) -> torch.tensor:
        """Preprocess the input."""
        # if rgb-image, normalize
        if image_observation:
            if invert:
                x = 255 - x
            x = torch.from_numpy(x).float() / 255
        else:
            x = torch.from_numpy(x).float()

        if len(x.shape) == 3:  # noqa: PLR2004
            x = x.permute(2, 0, 1).unsqueeze(0).to(device)
        else:
            x = x.permute(0, 3, 1, 2).to(device)
        return x
