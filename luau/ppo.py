# %%
from pathlib import Path

import torch
import torch.nn.functional as f
from torch import nn
from torch.distributions import Bernoulli, Categorical


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
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []

    def clear(self) -> None:
        """Clear the buffer."""
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    """Actor-Critic class. Only discrete action spaces are supported... for now."""

    def __init__(self, state_dim: torch.tensor, action_dim: int):
        super().__init__()

        # actor conv layers
        # TODO: should probably turn into a Sequential model
        self.actor_conv1 = nn.Conv2d(state_dim, 16, 2)
        self.actor_conv2 = nn.Conv2d(16, 32, 2)
        self.actor_conv3 = nn.Conv2d(32, 64, 2)

        # actor linear layers
        self.actor_fc1 = nn.Linear(65, 512)
        self.actor_fc2 = nn.Linear(512, action_dim)

        # critic conv layers
        # TODO: should probably turn into a Sequential model
        self.critic_conv1 = nn.Conv2d(state_dim, 16, 2)
        self.critic_conv2 = nn.Conv2d(16, 32, 2)
        self.critic_conv3 = nn.Conv2d(32, 64, 2)

        # critic linear layers
        self.critic_fc1 = nn.Linear(65, 512)  # Add +1 for the scalar input
        self.critic_fc2 = nn.Linear(512, 1)

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
        action_probs = f.softmax(self.actor_fc2(x), dim=-1)
        return action_probs

    def _critic_forward(self, image: torch.tensor, direction: torch.tensor) -> torch.tensor:
        """Run common computations for the critic network."""
        y = f.relu(self.critic_conv1(image))
        y = f.max_pool2d(y, 2)
        y = f.relu(self.critic_conv2(y))
        y = f.relu(self.critic_conv3(y))
        y = torch.flatten(y, 1)
        y = torch.cat((y, direction.view(-1, 1)), 1)
        y = f.relu(self.critic_fc1(y))
        state_values = self.critic_fc2(y)
        return state_values

    def forward(self, state: dict) -> tuple[torch.tensor, torch.tensor, torch.tensor]:
        """Forward pass."""
        direction = state["direction"]
        image = state["image"]

        # actor
        action_probs = self._actor_forward(image, direction)
        action_probs = action_probs.squeeze(0)  # Remove batch dimension
        dist = Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)

        # critic
        state_values = self._critic_forward(image, direction)

        return action.detach(), action_logprob.detach(), state_values.detach()

    def evaluate(self, states: list, actions: torch.tensor) -> tuple[torch.tensor, torch.tensor, torch.tensor]:
        """Evaluate the policy."""
        images = torch.squeeze(torch.stack([s["image"] for s in states], dim=0)).detach().to(device)
        directions = torch.squeeze(torch.stack([s["direction"] for s in states], dim=0)).detach().to(device)

        # actor
        action_probs = self._actor_forward(images, directions)
        dist = Categorical(action_probs)
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
    ):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.buffer = RolloutBuffer()
        self.policy = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr_actor)
        self.policy_old = ActorCritic(state_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()

    def select_action(self, state: dict) -> int:
        """Select an action."""
        with torch.no_grad():
            state = self.preprocess(state)
            action, action_logprob, state_val = self.policy_old(state)
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_val)
        return action.item()

    def _calculate_rewards(self) -> torch.tensor:
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
        return (rewards - rewards.mean()) / (rewards.std() + 1e-7)

    def _tensorize_rollout_buffer(self) -> tuple[torch.tensor, torch.tensor, torch.tensor]:
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device)
        return old_actions, old_logprobs, old_state_values

    def update(self) -> None:
        """Update the policy."""
        rewards = self._calculate_rewards()

        # convert list to tensor
        old_actions, old_logprobs, old_state_values = self._tensorize_rollout_buffer()

        # calculate advantages
        advantages = rewards.detach() - old_state_values.detach()

        # Optimize policy for K epochs
        for _ in range(self.k_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(self.buffer.states, old_actions)
            state_values = torch.squeeze(state_values)  # match state_values tensor dimensions with rewards tensor

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())  # Finding the ratio (pi_theta / pi_theta__old)

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            vf_loss = self.MseLoss(state_values, rewards)  # value function loss

            # final loss of clipped objective PPO
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

    def preprocess(self, x: dict) -> torch.tensor:
        """Preprocess the input."""
        direction = x["direction"]
        image = x["image"]
        image = torch.from_numpy(image).float()
        image = image.permute(2, 0, 1).unsqueeze(0).to(device)
        direction = torch.tensor(direction, dtype=torch.float).unsqueeze(0).to(device)
        x = {"direction": direction, "image": image}
        return x


# %% V################################## Introspective PPO ##################################


class IAARolloutBuffer(RolloutBuffer):
    """A buffer to store rollout data for Introspective Action Advising (IAA)."""

    def __init__(self):
        super().__init__()
        self.indicators = []

    def clear(self) -> None:
        """Clear the buffer."""
        super().clear()
        del self.indicators[:]


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
