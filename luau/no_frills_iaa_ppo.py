# %%
import argparse
import logging
import random
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as f
import yaml
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
def set_random_seeds(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior in PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# %%
class Trainer:
    """A class to train the agent."""

    def __init__(
        self,
        config_path: str,
        log_dir: str | None = None,
        model_dir: str | None = None,
        random_seed: int | None = None,
        run_id: str | None = None,
    ) -> None:
        with Path.open(config_path, "r") as file:
            config = yaml.safe_load(file)

        ################ Logging and Env hyperparameters ################
        self.algorithm = config["algorithm"]
        self.env_name = config["env_name"]
        self.door_locked = config["door_locked"]
        self.size = config["size"]
        self.save_frames = config["save_frames"]
        self.horizon = config["horizon"]
        self.max_training_timesteps = config["max_training_timesteps"]
        self.save_model_freq = config["save_model_freq"]
        self.image_observation = config["image_observation"]
        ################ IAA-PPO hyperparameters ################
        self.minibatch_size = config["minibatch_size"]
        self.k_epochs = config["k_epochs"]
        self.eps_clip = config["eps_clip"]
        self.gamma = config["gamma"]
        self.lr_actor = config["lr_actor"]
        self.num_envs = config["num_envs"]
        self.gae_lambda = config["gae_lambda"]
        self.teacher_model_path = config["teacher_model_path"]
        if self.teacher_model_path is None:
            raise ValueError("Teacher agent is None. Please specify pth model.")
        self.introspection_decay = config["introspection_decay"]
        self.burn_in = config["burn_in"]
        self.introspection_threshold = config["introspection_threshold"]

        # Store the random seed
        self.random_seed = random_seed
        self.rng = np.random.default_rng(self.random_seed)

        # Store log_dir and model_dir
        self.log_dir = log_dir
        self.model_dir = model_dir
        self.run_id = run_id

    def setup_directories(self) -> tuple[Path, Path]:
        """Make logging and checkpoint directories."""
        if self.log_dir is not None:
            log_dir = Path(f"{self.log_dir}/PPO_logs/{self.algorithm}/{self.env_name}/run_{self.run_id}_seed_{self.random_seed}")
        else:
            log_dir = Path(f"./PPO_logs/{self.algorithm}/{self.env_name}/run_{self.run_id}_seed_{self.random_seed}")
        log_dir.mkdir(parents=True, exist_ok=True)

        if self.model_dir is not None:
            model_dir = Path(f"{self.model_dir}/models/{self.algorithm}/{self.env_name}/run_{self.run_id}_seed_{self.random_seed}")
        else:
            model_dir = Path(f"./models/{self.algorithm}/{self.env_name}/run_{self.run_id}_seed_{self.random_seed}")
        model_dir.mkdir(parents=True, exist_ok=True)

        return log_dir, model_dir

    def save_frame(self, env: IntrospectiveEnv, time_step: int) -> None:
        """Save the current frame if save_frames is enabled."""
        if self.save_frames:
            img = env.render()
            plt.imsave(f"frames/frame_{time_step:06}.png", img)

    def _make_env(self, seed: int) -> IntrospectiveEnv:
        """Create the environment."""

        def _init() -> IntrospectiveEnv:
            rng = np.random.default_rng(seed)
            env = IntrospectiveEnv(rng=rng, size=self.size, locked=self.door_locked, render_mode="rgb_array")
            env = gym.wrappers.RecordEpisodeStatistics(env)
            env.reset(seed=seed)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
            return env

        return _init

    def get_vector_env(self, seed: int) -> gym.vector.AsyncVectorEnv:
        """Create the vectorized environment."""
        envs = [self._make_env(seed + i) for i in range(self.num_envs)]
        return gym.vector.AsyncVectorEnv(envs, shared_memory=False)

    def preprocess(self, x: dict) -> torch.tensor:
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


    def train(self) -> None:  # noqa: PLR0915, PLR0912
        """Train the agent."""
        msg = f"Training the {self.algorithm} agent in the {self.env_name} environment."
        logging.info(msg)

        # Make directories
        log_dir, model_dir = self.setup_directories()
        log_file = f"{log_dir}/{self.algorithm}_{self.env_name}_run_{self.run_id}_seed_{self.random_seed}_log.csv"
        checkpoint_path = f"{model_dir}/{self.algorithm}_{self.env_name}_run_{self.run_id}_seed_{self.random_seed}.pth"
        logging.info("Save checkpoint path: %s", checkpoint_path)

        # Initialize TensorBoard writer
        writer = SummaryWriter(log_dir=str(log_dir))

        # Initialize the PPO agent
        env = self.get_vector_env(self.random_seed)
        logging.info("Gridworld size: %s", env.env_fns[0]().unwrapped.max_steps)
        buffer = RolloutBuffer(self.horizon, self.num_envs, env.single_observation_space, env.single_action_space)
        state_dim = env.single_observation_space["image"].shape[-1]
        policy = ActorCritic(state_dim, env.single_action_space.n).to(device)
        optimizer = torch.optim.Adam(policy.parameters(), lr=self.lr_actor, eps=1e-5)

        # Initialize teacher model
        teacher_source_agent = ActorCritic(state_dim, env.single_action_space.n).to(device)
        teacher_source_agent.load_state_dict(torch.load(self.teacher_model_path))

        teacher_target_agent = ActorCritic(state_dim, env.single_action_space.n).to(device)
        teacher_target_agent.load_state_dict(torch.load(self.teacher_model_path))
        teacher_optimizer = torch.optim.Adam(teacher_target_agent.parameters(), lr=self.lr_actor, eps=1e-5)

        self.print_hyperparameters()

        # Track total training time
        start_time = datetime.now().astimezone().replace(microsecond=0)
        logging.info("Started training at (GMT): %s", start_time)
        logging.info("============================================================================================")

        # Logging file
        log_f = Path.open(log_file, "w+")
        log_f.write("episode,timestep,reward,episode_len\n")

        next_obs, _ = env.reset()
        next_obs = self.preprocess(next_obs)
        next_dones = torch.zeros(self.num_envs)
        global_step = 0

        # Training loop
        num_updates = self.max_training_timesteps // (self.horizon * self.num_envs)
        for update in range(1, num_updates + 1):
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * self.lr_actor
            optimizer.param_groups[0]["lr"] = lrnow
            teacher_optimizer.param_groups[0]["lr"] = lrnow
            for step in range(self.horizon):
                # Preprocess the next observation and store relevant data in the PPO agent's buffer
                buffer.images[step] = next_obs["image"]
                buffer.directions[step] = next_obs["direction"]
                buffer.is_terminals[step] = next_dones

                # Select actions and store them in the PPO agent's buffer
                with torch.no_grad():
                    # Initialize h_t as a tensor of zeros (h_t <- 0)
                    h_t = torch.zeros((self.num_envs,), dtype=torch.bool).to(device)

                    # Calculate introspection probability
                    probability = self.introspection_decay ** max(0, global_step - self.burn_in)
                    p = Bernoulli(probability).sample([self.num_envs]).to(device)  # Bernoulli sampling for all envs

                    # Only proceed if t > burn_in
                    if global_step > self.burn_in:
                        # Get value estimates from both teacher models in batch mode
                        _, _, teacher_source_vals = teacher_source_agent(next_obs)
                        _, _, teacher_target_vals = teacher_target_agent(next_obs)

                        # Compute the absolute difference between the source and target values
                        differences = torch.abs(teacher_target_vals - teacher_source_vals)

                        # Apply introspection condition in a vectorized manner
                        h_t = (p == 1) & (differences <= self.introspection_threshold)

                    buffer.indicators[step] = h_t
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
                next_obs, rewards, next_dones, truncated, info = env.step(actions.tolist())
                next_obs = self.preprocess(next_obs)
                next_dones = torch.tensor(np.logical_or(next_dones, truncated)).to(device)
                buffer.rewards[step] = torch.tensor(rewards, dtype=torch.float32).to(device).view(-1)

                global_step += 1 * self.num_envs
                for k, v in info.items():
                    if k == "episode":
                        done_indx = torch.argmax(next_dones.int())
                        episodic_reward = v["r"][done_indx]
                        episodic_length = v["l"][done_indx]
                        writer.add_scalar("charts/Episodic Reward", episodic_reward, global_step)
                        writer.add_scalar("charts/Episodic Length", episodic_length, global_step)
                        writer.add_scalar("charts/Rollout Reward", episodic_reward, update)
                        # Print average reward
                        logging.info(
                            "i_update: %s, Timestep: %s, Average Reward: %s, Episodic length: %s",
                            update,
                            global_step,
                            episodic_reward,
                            episodic_length,
                        )
                        log_f.write(f"{update},{global_step},{episodic_reward},{episodic_length}\n")
                        log_f.flush()
                    break

            # Calculate rewards and advantages using GAE
            with torch.no_grad():
                _, _, next_value = policy(next_obs)
                advantages = torch.zeros_like(buffer.rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(self.horizon)):
                    if t == self.horizon - 1:
                        next_non_terminal = 1.0 - next_dones.float()
                        nextvalues = next_value  # Bootstrapping for the last value
                    else:
                        next_non_terminal = 1.0 - buffer.is_terminals[t + 1]
                        nextvalues = buffer.state_values[t + 1]

                    # Temporal difference error
                    delta = buffer.rewards[t] + self.gamma * nextvalues * next_non_terminal - buffer.state_values[t]
                    advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * next_non_terminal * lastgaelam
                returns = advantages + buffer.state_values

            # Flatten the buffer
            b_returns = returns.reshape(-1).detach()
            b_advantages = advantages.reshape(-1).detach()
            b_actions = torch.flatten(buffer.actions, 0, 1).detach()
            b_logprobs = torch.flatten(buffer.logprobs, 0, 1).detach()
            b_images = torch.flatten(buffer.images, 0, 1).detach()
            b_directions = torch.flatten(buffer.directions, 0, 1).detach()
            b_state_values = torch.flatten(buffer.state_values, 0, 1).detach()
            b_indicators = torch.flatten(buffer.indicators, 0, 1).detach()

            batch_size = self.num_envs * self.horizon
            b_inds = np.arange(batch_size)
            clipfracs = []

            # Optimize policy for K epochs
            for _ in range(self.k_epochs):
                self.rng.shuffle(b_inds)

                # Split data into minibatches
                for i in range(0, batch_size, self.minibatch_size):
                    end = i + self.minibatch_size
                    mb_inds = b_inds[i:end]
                    mb_states = {"image": b_images[mb_inds], "direction": b_directions[mb_inds]}
                    new_logprob, new_value, dist_entropy = policy.evaluate(mb_states, b_actions.long()[mb_inds])
                    teacher_new_logprob, teacher_new_value, teacher_dist_entropy = teacher_target_agent.evaluate(mb_states,
                                                                                                                b_actions.long()[mb_inds])

                    with torch.no_grad():
                        _, teacher_source_new_logprob, _ = teacher_source_agent(mb_states)
                        _, student_new_logprob, _ = policy(mb_states)
                        mb_rho_t = torch.ones(self.minibatch_size).to(device)
                        mb_rho_s = torch.ones(self.minibatch_size).to(device)
                        for j, h_i in enumerate(b_indicators[mb_inds]):
                            if h_i.item() == 1:
                                mb_rho_s[j] = torch.clamp(student_new_logprob[j] / b_logprobs[mb_inds][j], -0.2, 0.2).item()
                            else:
                                mb_rho_t[j] = torch.clamp(teacher_source_new_logprob[j] / b_logprobs[mb_inds][j], -0.2, 0.2).item()

                    # policy gradient
                    log_ratio = new_logprob - b_logprobs[mb_inds].detach()
                    ratios = torch.exp(log_ratio) * mb_rho_s  # Finding the ratio (pi_theta / pi_theta__old)

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-log_ratio).mean()
                        approx_kl = ((ratios - 1) - log_ratio).mean()
                        clipfracs += [((ratios - 1.0).abs() > self.eps_clip).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                    surr1 = -mb_advantages * ratios
                    surr2 = -mb_advantages * torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip)
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
                    teacher_surr2 = -mb_advantages * torch.clamp(teacher_ratios, 1 - self.eps_clip, 1 + self.eps_clip)
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

            if update % self.save_model_freq == 0:
                logging.info("--------------------------------------------------------------------------------------------")
                logging.info("Saving model to: %s", checkpoint_path)
                logging.info("--------------------------------------------------------------------------------------------")
                torch.save(policy.state_dict(), checkpoint_path)

        log_f.close()
        env.close()
        writer.close()

        # Print total training time
        logging.info("============================================================================================")
        end_time = datetime.now().astimezone().replace(microsecond=0)
        logging.info("Started training at (GMT): %s", start_time)
        logging.info("Finished training at (GMT): %s", end_time)
        logging.info("Total training time: %s", end_time - start_time)
        logging.info("============================================================================================")

    def print_hyperparameters(self) -> None:
        """Print the hyperparameters."""
        logging.info("--------------------------------------------------------------------------------------------")
        logging.info("Training the agent in the %s environment. Door: %s", self.env_name, self.door_locked)
        logging.info("max training timesteps: %s", self.max_training_timesteps)
        logging.info("max timesteps per episode (M/horizon/rollout): %s", self.horizon)
        logging.info("run num: %s, seed: %s", self.run_id, self.random_seed)
        logging.info("model saving frequency: %s timesteps", self.save_model_freq)
        logging.info("--------------------------------------------------------------------------------------------")
        logging.info("Initializing a discrete action space policy")
        logging.info("--------------------------------------------------------------------------------------------")
        logging.info("PPO update frequency: %s timesteps", self.horizon)
        logging.info("PPO K epochs: %s", self.k_epochs)
        logging.info("PPO epsilon clip: %s", self.eps_clip)
        logging.info("discount factor (gamma): %s", self.gamma)
        logging.info("--------------------------------------------------------------------------------------------")
        logging.info("optimizer learning rate actor: %s", self.lr_actor)
        logging.info("============================================================================================")


# %%

if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description="Train the agent.")
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to the config file.",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default=None,
        help="Directory to save logs.",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="Directory to save models.",
    )
    parser.add_argument(
        "--num_experiments",
        type=int,
        default=1,
        help="Number of experiments to run.",
    )
    args = parser.parse_args()

    with Path(args.config_path).open("r") as file:
        base_config = yaml.safe_load(file)
    base_random_seed = base_config.get("random_seed", 0)

    for i in range(args.num_experiments):
        # Generate a unique random seed for each experiment
        random_seed = base_random_seed + i * base_config.get("num_envs", 0)
        run_id = base_config.get("run_num", 0)
        logging.info("============================================================================================")
        logging.info("Running experiment %s with random seed %s.", i + 1, random_seed)
        logging.info("--------------------------------------------------------------------------------------------")

        # # Update the random seeds for each experiment
        set_random_seeds(random_seed)

        trainer = Trainer(
            config_path=args.config_path,
            log_dir=args.log_dir,
            model_dir=args.model_dir,
            random_seed=random_seed,
            run_id=run_id,
        )
        trainer.train()
