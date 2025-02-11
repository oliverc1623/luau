# Actor-Critic DQN

import argparse  # noqa: I001
import random
import time
from distutils.util import strtobool
from pathlib import Path

import minigrid  # import minigrid before gym to register envs  # noqa: F401
import gymnasium as gym
import numpy as np
import torch
from torch.distributions.categorical import Categorical
from minigrid.wrappers import ImgObsWrapper
from torch.nn import functional as f
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter

from typing import NamedTuple

RGB_CHANNEL = 3


# Define a transition tuple for DQN
class Transition(NamedTuple):
    """A named tuple representing a single transition in the environment."""

    state: torch.tensor
    action: torch.tensor
    reward: torch.tensor
    next_state: torch.tensor
    done: torch.tensor


class DQNReplayBuffer:
    """Replay buffer for storing transitions experienced by the agent."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args: tuple) -> None:
        """Save a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> Transition:
        """Sample a batch of transitions from the replay buffer."""
        batch = random.sample(self.memory, batch_size)
        batch = Transition(*zip(*batch, strict=False))
        return Transition(
            state=torch.cat(batch.state, dim=0),
            action=torch.stack(batch.action),
            reward=torch.stack(batch.reward),
            next_state=torch.cat(batch.next_state, dim=0),
            done=torch.stack(batch.done),
        )

    def __len__(self):
        return len(self.memory)


def parse_args() -> argparse.Namespace:
    """Parse the arguments for the script."""
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=str(Path(__file__).stem),
        help="the name of this experiment")
    parser.add_argument("--gym-id", type=str, default="MiniGrid-Empty-5x5-v0",
        help="the id of the gym environment")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--total-timesteps", type=int, default=25000,
        help="total timesteps of the experiments")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="weather to capture videos of the agent performances (check out `videos` folder)")
    parser.add_argument("--save-model-freq", type=int, default=1000,
        help="the frequency of saving the model")

    # Algorithm specific arguments
    parser.add_argument("--buffer-size", type=int, default=10000,
        help="the size of the replay buffer")
    parser.add_argument("--tau", type=float, default=1.0,
        help="the soft update rate")
    parser.add_argument("--target-network-frequency", type=int, default=500,
        help="the frequency of updating the target network")
    parser.add_argument("--batch-size", type=int, default=128,
        help="the batch size of the training")
    parser.add_argument("--start-e", type=float, default=1,
        help="the starting exploration rate")
    parser.add_argument("--end-e", type=float, default=0.05,
        help="the final exploration rate")
    parser.add_argument("--exploration-fraction", type=float, default=0.5,
        help="the fraction of the total timesteps for exploration")
    parser.add_argument("--learning-starts", type=int, default=1000,
        help="the number of steps to take before learning starts")
    parser.add_argument("--train-frequency", type=int, default=10,
        help="the frequency of training the agent")
    parser.add_argument("--actor-update-frequency", type=int, default=20,
        help="the frequency of training the agent")
    parser.add_argument("--num-envs", type=int, default=4,
        help="the number of parallel game environments")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gradient-steps", type=int, default=1,
        help="the number of gradient steps to take per iteration")

    args = parser.parse_args()
    # fmt: on
    return args


def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Module:
    """Initialize the layers of the network."""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def preprocess(image: np.array) -> dict:
    """Preprocess the input for a grid-based environment."""
    image = torch.from_numpy(image).float()
    if image.ndim == RGB_CHANNEL:  # Single image case with shape (height, width, channels)
        image = image.permute(2, 0, 1)
        # Permute back to (batch_size, channels, height, width)
        image = image.unsqueeze(0).to(device)  # Adding batch dimension
    else:  # Batch case with shape (batch_size, height, width, channels)
        image = image.permute(0, 3, 1, 2).to(device)  # Change to (batch, channels, height, width)
    return image


def linear_schedule(start_e: float, end_e: float, duration: int, t: int) -> float:
    """Calculate the linear schedule for exploration rate."""
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def write_to_tensorboard(writer: SummaryWriter, global_step: int, info: dict) -> None:
    """Write data to TensorBoard."""
    if "episode" in info:
        # Extract the mask for completed episodes
        completed_mask = info["_episode"]

        # Filter the rewards and lengths for completed episodes
        episodic_returns = info["episode"]["r"][completed_mask]
        episodic_lengths = info["episode"]["l"][completed_mask]

        # Log each completed episode
        for ep_return, ep_length in zip(episodic_returns, episodic_lengths, strict=False):
            print(f"global_step={global_step}, episodic_return={ep_return}")
            writer.add_scalar("charts/episodic_return", ep_return, global_step)
            writer.add_scalar("charts/episodic_length", ep_length, global_step)


class Actor(nn.Module):
    """The agent class for the AC DQN algorithm."""

    def __init__(self, envs: gym.vector.SyncVectorEnv):
        super().__init__()
        # Actor network
        c = envs.single_observation_space.shape[-1]
        # Define image embedding
        self.image_conv = nn.Sequential(
            nn.Conv2d(c, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU(),
        )
        n = envs.single_observation_space.shape[0]
        m = envs.single_observation_space.shape[1]
        self.image_embedding_size = ((n - 1) // 2 - 2) * ((m - 1) // 2 - 2) * 64

        # Define actor's model
        self.actor = nn.Sequential(
            nn.Linear(self.image_embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, envs.single_action_space.n),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the network."""
        x = self.image_conv(x)
        x = x.reshape(x.shape[0], -1)
        embedding = x
        return self.actor(embedding)

    def get_action(self, x: torch.Tensor) -> torch.Tensor:
        """Get the action, log probs, and probs for all actions from the actor network."""
        logits = self(x)
        policy_dist = Categorical(logits=logits)
        action = policy_dist.sample()
        action_probs = policy_dist.probs
        log_prob = f.log_softmax(logits, dim=1)
        return action, log_prob, action_probs


class QNetwork(nn.Module):
    """The critic (QNetwork) class for the DQN algorithm."""

    def __init__(self, envs: gym.vector.SyncVectorEnv):
        super().__init__()
        c = envs.single_observation_space.shape[-1]
        # Define image embedding
        self.image_conv = nn.Sequential(
            nn.Conv2d(c, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU(),
        )
        n = envs.single_observation_space.shape[0]
        m = envs.single_observation_space.shape[1]
        self.image_embedding_size = ((n - 1) // 2 - 2) * ((m - 1) // 2 - 2) * 64

        self.critic = nn.Sequential(
            nn.Linear(self.image_embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, envs.single_action_space.n),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the critic network."""
        x = self.image_conv(x)
        x = x.reshape(x.shape[0], -1)
        return self.critic(x)


if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.gym_id}__{args.exp_name}__{int(time.time())}"
    writer = SummaryWriter(f"../../pvcvolume/runs/{run_name}")
    model_dir = Path(f"../../pvcvolume/model/{run_name}")
    model_dir.mkdir(parents=True, exist_ok=True)
    actor_checkpoint_path = f"{model_dir}/{run_name}_actor.pth"
    critic_checkpoint_path = f"{model_dir}/{run_name}_critic.pth"
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n{}".format("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    def make_env(subenv_seed: int, idx: int, capture_video: int, run_name: str, save_model_freq: int) -> gym.Env:
        """Create the environment."""

        def _init() -> gym.Env:
            env = gym.make(args.gym_id, render_mode="rgb_array")
            env.action_space = gym.spaces.Discrete(7)  # make all 7 actions available
            if capture_video and idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}", episode_trigger=lambda x: x % save_model_freq == 0)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            env = ImgObsWrapper(env)
            env.reset(seed=subenv_seed)
            env.action_space.seed(subenv_seed)
            env.observation_space.seed(subenv_seed)
            return env

        return _init

    envs = [make_env(args.seed + i, i, args.capture_video, run_name, args.save_model_freq) for i in range(args.num_envs)]
    envs = gym.vector.SyncVectorEnv(envs)
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    # agent setup
    agent = Actor(envs).to(device)
    critic = QNetwork(envs).to(device)
    target_network = QNetwork(envs).to(device)
    target_network.load_state_dict(critic.state_dict())
    actor_optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    critic_optimizer = optim.Adam(critic.parameters(), lr=args.learning_rate, eps=1e-5)

    # replay buffer
    rb = DQNReplayBuffer(args.buffer_size)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    obs, infos = envs.reset()
    obs = preprocess(obs)
    next_done = torch.zeros(args.num_envs).to(device)

    for global_step in range(args.total_timesteps):
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        if rng.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions, _, _ = agent.get_action(obs).cpu().numpy()

        # step the envs
        next_obs, rewards, dones, truncations, infos = envs.step(actions)

        # process rewards and dones
        rewards = torch.tensor(rewards).to(device).view(-1)
        dones = torch.Tensor(dones).to(device)

        # record metrics for plotting
        for _, info in enumerate(infos):
            if "episode" in info:
                ep_return = info["episode"]["r"]
                ep_length = info["episode"]["l"]

                print(f"global_step={global_step}, episodic_return={ep_return}")
                writer.add_scalar("charts/episodic_return", ep_return, global_step)
                writer.add_scalar("charts/episodic_length", ep_length, global_step)
                break

        # get real terminal observation
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                print(infos)
                terminal_obs = infos[idx]["terminal_observation"]
                real_next_obs[idx] = terminal_obs
        real_next_obs = preprocess(real_next_obs)

        rb.push(obs, torch.tensor(actions), rewards, real_next_obs, dones)
        obs = preprocess(next_obs)

        # ALGO LOGIC: training.
        if len(rb) > args.batch_size:
            if global_step % args.train_frequency == 0:
                for _ in range(args.gradient_steps):
                    # sample a batch of data
                    data = rb.sample(args.batch_size)

                    # compute advantages
                    with torch.no_grad():
                        next_q_values, _ = target_network(data.next_state).max(dim=1)
                        td_target = data.reward.flatten().float() + args.gamma * next_q_values.float() * (1 - data.done.flatten().float())
                    old_val = critic(data.state.to(device).float()).gather(1, data.action.to(device).view(-1, 1).long()).squeeze(-1)
                    critic_loss = f.mse_loss(td_target, old_val)

                    # optimize the critic QNetwork
                    critic_optimizer.zero_grad()
                    critic_loss.backward()
                    critic_optimizer.step()

                    if global_step % args.actor_update_frequency == 0:
                        # Policy gradient update for the actor
                        action_probs = agent(data.state)
                        chosen_action_probs = action_probs.gather(1, data.action.to(device).view(-1, 1)).squeeze(1)
                        advantages = td_target - old_val
                        actor_loss = -torch.mean(torch.log(chosen_action_probs) * advantages.detach())

                        # optimize the actor
                        actor_optimizer.zero_grad()
                        actor_loss.backward()
                        actor_optimizer.step()
                        writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)

            # update target network
            if global_step % args.target_network_frequency == 0:
                for target_network_param, q_network_param in zip(target_network.parameters(), critic.parameters(), strict=False):
                    target_network_param.data.copy_(
                        args.tau * q_network_param.data + (1.0 - args.tau) * target_network_param.data,
                    )

                writer.add_scalar("losses/critic_loss", critic_loss.item(), global_step)
                writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                writer.add_scalar("losses/epsilon", epsilon, global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

            if global_step % args.save_model_freq == 0:
                print(f"Saving model checkpoint at step {global_step} to {actor_checkpoint_path}")
                torch.save(agent.state_dict(), actor_checkpoint_path)
                torch.save(critic.state_dict(), critic_checkpoint_path)

    envs.close()
    writer.close()
