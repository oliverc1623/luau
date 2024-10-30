# %%
import logging
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as f
from torch import nn
from torch.distributions import Bernoulli, Categorical
from torch.utils.tensorboard import SummaryWriter

from luau.iaa_env import IntrospectiveEnv


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

        self.images = torch.zeros(self.horizon, self.num_envs, *self.img_shape).to(device)
        self.directions = torch.zeros((self.horizon, self.num_envs, *self.state["direction"].shape)).to(device)
        self.actions = torch.zeros((self.horizon, self.num_envs, *self.action_space.shape)).to(device)
        self.logprobs = torch.zeros((self.horizon, self.num_envs)).to(device)
        self.rewards = torch.zeros((self.horizon, self.num_envs)).to(device)
        self.is_terminals = torch.zeros((self.horizon, self.num_envs)).to(device)
        self.state_values = torch.zeros((self.horizon, self.num_envs)).to(device)
        self.indicators = torch.zeros((self.horizon, self.num_envs)).to(device)

    def clear(self) -> None:
        """Clear the buffer."""
        self.images = torch.zeros(self.horizon, self.num_envs, *self.img_shape).to(device)
        self.directions = torch.zeros((self.horizon, self.num_envs, *self.state["direction"].shape)).to(device)
        self.actions = torch.zeros((self.horizon, self.num_envs, *self.action_space.shape)).to(device)
        self.logprobs = torch.zeros((self.horizon, self.num_envs)).to(device)
        self.rewards = torch.zeros((self.horizon, self.num_envs)).to(device)
        self.is_terminals = torch.zeros((self.horizon, self.num_envs)).to(device)
        self.state_values = torch.zeros((self.horizon, self.num_envs)).to(device)
        self.indicators = torch.zeros((self.horizon, self.num_envs)).to(device)


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
        logits = self._actor_forward(image, direction)
        dist = Categorical(logits=logits)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_values = self._critic_forward(image, direction)
        return action, action_logprob, state_values

    def evaluate(self, states: dict, actions: torch.tensor) -> tuple[torch.tensor, torch.tensor, torch.tensor]:
        """Evaluate the policy."""
        images = states["image"]
        directions = states["direction"]
        logits = self._actor_forward(images, directions)
        dist = Categorical(logits=logits)
        action_logprobs = dist.log_prob(actions)
        dist_entropy = dist.entropy()
        state_values = self._critic_forward(images, directions)
        return action_logprobs, state_values, dist_entropy


# %%


def preprocess(x: dict) -> torch.tensor:
    """Preprocess the input."""
    direction = x["direction"]
    image = x["image"]
    image = torch.from_numpy(image).float()
    if len(image.shape) == RGB_CHANNEL:
        image = image.unsqueeze(0).permute(0, 3, 1, 2).to(device)
    else:
        image = image.permute(0, 3, 1, 2).to(device)
    direction = torch.tensor(direction, dtype=torch.float).to(device)
    x = {"direction": direction, "image": image}
    return x


# %%

# Initialize TensorBoard writer
writer = SummaryWriter()

# Initialize the PPO agent
seed = 47
horizon = 128
num_envs = 1
lr_actor = 0.0005
teacher_model_path = "/root/../pvcvolume/models/PPO/IntrospectiveEnvUnlocked/run_1_seed_1623/PPO_IntrospectiveEnvUnlocked_run_1_seed_1623.pth"
max_training_timesteps = 500_000
introspection_decay = 0.99999
burn_in = 0
introspection_threshold = 0.9
gamma = 0.99
gae_lambda = 0.8
eps_clip = 0.2
minibatch_size = 128
k_epochs = 4

rng = np.random.default_rng(seed)
env = IntrospectiveEnv(rng=rng, size=9, locked=True, render_mode="rgb_array")
env = gym.wrappers.RecordEpisodeStatistics(env)
env.reset(seed=seed)
env.action_space.seed(seed)
env.observation_space.seed(seed)

buffer = RolloutBuffer(horizon, num_envs, env.observation_space, env.action_space)
state_dim = env.observation_space["image"].shape[-1]
policy = ActorCritic(state_dim, env.action_space.n).to(device)
optimizer = torch.optim.Adam(policy.parameters(), lr=lr_actor, eps=1e-5)

# Initialize teacher model
teacher_source_agent = ActorCritic(state_dim, env.action_space.n).to(device)
teacher_source_agent.load_state_dict(torch.load(teacher_model_path))

teacher_target_agent = ActorCritic(state_dim, env.action_space.n).to(device)
teacher_target_agent.load_state_dict(torch.load(teacher_model_path))
teacher_optimizer = torch.optim.Adam(teacher_target_agent.parameters(), lr=lr_actor, eps=1e-5)

# %%

# Assert that teacher_source_agent and teacher_target_agent have the same parameters
for param_source, param_target in zip(teacher_source_agent.parameters(), teacher_target_agent.parameters(), strict=False):
    assert torch.equal(param_source, param_target), "Mismatch in parameters between teacher_source_agent and teacher_target_agent"

# %%

next_obs, _ = env.reset()
next_obs = preprocess(next_obs)
next_dones = torch.zeros(num_envs).to(device)
advice_issued = torch.zeros(num_envs).to(device)
global_step = 0

# %%

with torch.no_grad():
    _, _, teacher_source_val = teacher_source_agent(next_obs)
    _, _, teacher_target_val = teacher_target_agent(next_obs)

# %%

# Training loop
num_updates = max_training_timesteps // (horizon * num_envs)
for update in range(1, 2):
    for step in range(5):
        # Preprocess the next observation and store relevant data in the PPO agent's buffer
        buffer.images[step] = next_obs["image"]
        buffer.directions[step] = next_obs["direction"]
        buffer.is_terminals[step] = next_dones

        # Select actions and store them in the PPO agent's buffer
        # Initialize h_t as a tensor of zeros (h_t <- 0)
        h_t = torch.zeros((num_envs,), dtype=torch.bool).to(device)

        # Calculate introspection probability
        probability = introspection_decay ** max(0, global_step - burn_in)
        p = Bernoulli(probability).sample([num_envs]).to(device)  # Bernoulli sampling for all envs

        # Only proceed if t > burn_in
        if global_step > burn_in:
            # Get value estimates from both teacher models in batch mode
            _, _, teacher_source_vals = teacher_source_agent(next_obs)
            _, _, teacher_target_vals = teacher_target_agent(next_obs)
            print(teacher_source_vals, teacher_target_vals)
            differences = torch.abs(teacher_target_vals - teacher_source_vals)
            h_t = (p == 1) & (differences <= introspection_threshold)

        with torch.no_grad():
            buffer.indicators[step] = h_t
            advice_issued += h_t.float()
            teacher_actions, teacher_action_logprobs, teacher_state_vals = teacher_source_agent(next_obs)
            student_actions, student_action_logprobs, student_state_vals = policy(next_obs)
            # Use h_t to select the outputs
            actions = torch.where(h_t, teacher_actions, student_actions)
            log_probs = torch.where(h_t, teacher_action_logprobs, student_action_logprobs)
            state_values = torch.where(h_t, teacher_state_vals, student_state_vals)
            buffer.state_values[step] = state_values

        buffer.actions[step] = actions
        buffer.logprobs[step] = log_probs

        # Step the environment and store the rewards
        next_obs, rewards, next_dones, truncated, info = env.step(actions.item())
        next_obs = preprocess(next_obs)
        next_dones = torch.tensor(np.logical_or(next_dones, truncated)).to(device)
        buffer.rewards[step] = torch.tensor(rewards, dtype=torch.float32).to(device).view(-1)

        global_step += 1 * num_envs
        for k, v in info.items():
            if k == "episode":
                done_indx = torch.argmax(next_dones.int())
                episodic_reward = v["r"][done_indx]
                episodic_length = v["l"][done_indx]

                logging.info(
                    "i_update: %s, \t Timestep: %s, \t Average Reward: %s, \t Episodic length: %s, \t Advice Issued env 0: %s",
                    update,
                    global_step,
                    episodic_reward,
                    episodic_length,
                    advice_issued[0].item(),
                )

                advice_issued = torch.zeros(num_envs).to(device)
            break

    # Calculate rewards and advantages using GAE
    with torch.no_grad():
        _, _, next_value = policy(next_obs)
        # TODO: Calculate advantages for teacher value function
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

    print(returns)

    # %%
    # Flatten the buffer
    b_returns = returns.reshape(-1).detach()
    b_advantages = advantages.reshape(-1).detach()
    b_actions = torch.flatten(buffer.actions, 0, 1).detach()
    b_logprobs = torch.flatten(buffer.logprobs, 0, 1).detach()
    b_images = torch.flatten(buffer.images, 0, 1).detach()
    b_directions = torch.flatten(buffer.directions, 0, 1).detach()
    b_state_values = torch.flatten(buffer.state_values, 0, 1).detach()
    b_indicators = torch.flatten(buffer.indicators, 0, 1).detach()

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
            teacher_new_logprob, teacher_new_value, teacher_dist_entropy = teacher_target_agent.evaluate(mb_states, b_actions.long()[mb_inds])

            with torch.no_grad():
                _, teacher_source_new_logprob, _ = teacher_source_agent(mb_states)
                _, student_new_logprob, _ = policy(mb_states)
                mb_rho_t = torch.ones(minibatch_size).to(device)
                mb_rho_s = torch.ones(minibatch_size).to(device)
                for j, h_i in enumerate(b_indicators[mb_inds]):
                    if h_i.item() == 1:
                        mb_rho_s[j] = torch.clamp(student_new_logprob[j] / b_logprobs[mb_inds][j], -2, 2).item()
                    else:
                        mb_rho_t[j] = torch.clamp(teacher_source_new_logprob[j] / b_logprobs[mb_inds][j], -2, 2).item()

            # policy gradient
            log_ratio = new_logprob - b_logprobs[mb_inds].detach()
            ratios = torch.exp(log_ratio) * mb_rho_s  # Finding the ratio (pi_theta / pi_theta__old)

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
            student_loss = pg_loss_student - 0.01 * entropy_loss_student + v_loss_student * 0.5  # final loss of clipped objective PPO

            # policy gradient
            teacher_log_ratio = teacher_new_logprob - b_logprobs[mb_inds].detach()
            teacher_ratios = teacher_log_ratio.exp() * mb_rho_t  # Finding the ratio (pi_theta / pi_theta__old)
            teacher_surr1 = -mb_advantages * teacher_ratios
            teacher_surr2 = -mb_advantages * torch.clamp(teacher_ratios, 1 - eps_clip, 1 + eps_clip)
            pg_loss_teacher = torch.max(teacher_surr1, teacher_surr2).mean()

            # value function loss + clipping
            teacher_new_value = teacher_new_value.view(-1)
            v_loss_unclipped = (teacher_new_value - b_returns[mb_inds]) ** 2
            v_clipped = b_state_values[mb_inds] + torch.clamp(teacher_new_value - b_state_values[mb_inds], -10.0, 10.0)
            v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
            v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
            v_loss_teacher = 0.5 * v_loss_max.mean()

            # entropy loss
            entropy_loss_teacher = teacher_dist_entropy.mean()
            teacher_loss = pg_loss_teacher - 0.01 * entropy_loss_teacher + v_loss_teacher * 0.5  # final loss of clipped objective PPO

            teacher_optimizer.zero_grad()  # take gradient step
            teacher_loss.backward()
            nn.utils.clip_grad_norm_(teacher_target_agent.parameters(), 0.5)
            teacher_optimizer.step()

            optimizer.zero_grad()  # take gradient step
            student_loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()

        if approx_kl.item() > KL_THRESHOLD:
            break

    # log debug variables
    with torch.no_grad():
        writer.add_scalar("debugging/policy_loss", pg_loss_student.item(), global_step)
        writer.add_scalar("debugging/value_loss", v_loss_student.item(), global_step)
        writer.add_scalar("debugging/entropy_loss", entropy_loss_student.item(), global_step)
        writer.add_scalar("debugging/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("debugging/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("debugging/clipfrac", np.mean(clipfracs), global_step)

    buffer.clear()
