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
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.vec_env import SubprocVecEnv


RGB_CHANNEL = 3

gym.register(id="SmallFourRoomDoorKey-v0", entry_point="luau.multi_room_env:SmallFourRoomDoorKey")
gym.register(id="SmallFourRoomDoorKeyLocked-v0", entry_point="luau.multi_room_env:SmallFourRoomDoorKeyLocked")


class DQNReplayBuffer:
    """Replay buffer for storing transitions experienced by the agent."""

    def __init__(self, capacity: int, n_envs: 1):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.n_envs = n_envs

    def push(self, state: np.array, action: np.array, reward: np.array, next_state: np.array, done: np.array) -> None:
        """Save a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        action = action.reshape((self.n_envs, 1))
        self.memory[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> tuple:
        """Sample a batch of transitions from the replay buffer."""
        batch = random.sample(self.memory, batch_size)
        s, a, r, ns, d = zip(*batch, strict=False)
        w, h, c = s[0].shape[1], s[0].shape[2], s[0].shape[3]
        s_t = torch.as_tensor(np.stack(s), device=device).float().view(args.batch_size * args.num_envs, w, h, c)
        a_t = torch.as_tensor(np.array(a), device=device).long().view(-1)
        r_t = torch.as_tensor(np.array(r), device=device).float().view(-1)
        ns_t = torch.as_tensor(np.stack(ns), device=device).float().view(args.batch_size * args.num_envs, w, h, c)
        d_t = torch.as_tensor(np.array(d), device=device).float().view(-1)
        return s_t, a_t, r_t, ns_t, d_t


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
    parser.add_argument("--save-video-freq", type=int, default=100_000,
            help="the frequency of saving the video")

    # Algorithm specific arguments
    parser.add_argument("--ql-lr", type=float, default=0.0003,
        help="the learning rate of the ql optimizer")
    parser.add_argument("--buffer-size", type=int, default=10000,
        help="the size of the replay buffer")
    parser.add_argument("--tau", type=float, default=1.0,
        help="the soft update rate")
    parser.add_argument("--target-network-frequency", type=int, default=8000,
        help="the frequency of updating the target network")
    parser.add_argument("--batch-size", type=int, default=128,
        help="the batch size of the training")
    parser.add_argument("--start-e", type=float, default=1,
        help="the starting exploration rate")
    parser.add_argument("--end-e", type=float, default=0.05,
        help="the final exploration rate")
    parser.add_argument("--exploration-fraction", type=float, default=0.5,
        help="the fraction of the total timesteps for exploration")
    parser.add_argument("--learning-starts", type=int, default=10000,
        help="the number of steps to take before learning starts")
    parser.add_argument("--update-frequency", type=int, default=4,
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


def linear_schedule(start_e: float, end_e: float, duration: int, t: int) -> float:
    """Calculate the linear schedule for exploration rate."""
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


class Actor(nn.Module):
    """The agent class for the AC DQN algorithm."""

    def __init__(self, envs: gym.vector.SyncVectorEnv):
        super().__init__()
        # Actor network
        c = envs.observation_space.shape[-1]
        # Define image embedding
        self.image_conv = nn.Sequential(
            layer_init(nn.Conv2d(c, 16, (2, 2))),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            layer_init(nn.Conv2d(16, 32, (2, 2))),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, (2, 2))),
            nn.ReLU(),
        )
        n = envs.observation_space.shape[0]
        m = envs.observation_space.shape[1]
        self.image_embedding_size = ((n - 1) // 2 - 2) * ((m - 1) // 2 - 2) * 64

        # Define actor's model
        self.actor = nn.Sequential(
            nn.Linear(self.image_embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, envs.action_space.n),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the network."""
        x = self.image_conv(x)
        x = x.reshape(x.shape[0], -1)
        return self.actor(x)

    def get_action(self, x: torch.Tensor) -> torch.Tensor:
        """Get the action, log probs, and probs for all actions from the actor network."""
        logits = self(x)
        policy_dist = Categorical(logits=logits)
        action = policy_dist.sample()
        action_probs = policy_dist.probs
        log_prob = f.log_softmax(logits, dim=1)
        return action, log_prob, action_probs, policy_dist


class QNetwork(nn.Module):
    """The critic (QNetwork) class for the DQN algorithm."""

    def __init__(self, envs: gym.vector.SyncVectorEnv):
        super().__init__()
        c = envs.observation_space.shape[-1]
        # Define image embedding
        self.image_conv = nn.Sequential(
            layer_init(nn.Conv2d(c, 16, (2, 2))),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            layer_init(nn.Conv2d(16, 32, (2, 2))),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, (2, 2))),
            nn.ReLU(),
        )
        n = envs.observation_space.shape[0]
        m = envs.observation_space.shape[1]
        self.image_embedding_size = ((n - 1) // 2 - 2) * ((m - 1) // 2 - 2) * 64

        self.critic = nn.Sequential(
            nn.Linear(self.image_embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, envs.action_space.n),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the critic network."""
        x = self.image_conv(x)
        x = x.reshape(x.shape[0], -1)
        return self.critic(x)


if __name__ == "__main__":
    args = parse_args()

    # tensorboard
    run_name = f"{args.gym_id}__{args.exp_name}"
    log_dir = f"../../pvcvolume/runs/{run_name}"
    writer = SummaryWriter(log_dir, flush_secs=5)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n{}".format("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # model_dir
    model_dir = Path(f"../../pvcvolume/models/{run_name}")
    model_dir.mkdir(parents=True, exist_ok=True)
    actor_checkpoint_path = f"{model_dir}/{run_name}_actor.pth"
    critic_checkpoint_path = f"{model_dir}/{run_name}_critic.pth"

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    def make_env(subenv_seed: int, idx: int, capture_video: int, run_name: str) -> gym.Env:
        """Create the environment."""

        def _init() -> gym.Env:
            env = gym.make(args.gym_id, render_mode="rgb_array")
            env.action_space = gym.spaces.Discrete(7)  # make all 7 actions available
            if capture_video and idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}", episode_trigger=lambda x: x % args.save_video_freq == 0)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            env = ImgObsWrapper(env)
            env.reset(seed=subenv_seed)
            env.action_space.seed(subenv_seed)
            env.observation_space.seed(subenv_seed)
            return env

        return _init

    envs = [make_env(args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    envs = SubprocVecEnv(envs)
    assert isinstance(envs.action_space, gym.spaces.Discrete), "only discrete action space is supported"
    envs = VecMonitor(envs, log_dir)

    # agent setup
    agent = Actor(envs).to(device)
    critic = QNetwork(envs).to(device)
    target_network = QNetwork(envs).to(device)
    target_network.load_state_dict(critic.state_dict())
    actor_optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-4)
    critic_optimizer = optim.Adam(critic.parameters(), lr=args.ql_lr, eps=1e-4)

    # replay buffer
    rb = DQNReplayBuffer(args.buffer_size, n_envs=args.num_envs)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    obs = envs.reset()

    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: select action
        if global_step < args.learning_starts:
            actions = np.array([envs.action_space.sample() for _ in range(envs.num_envs)])
        else:
            # Convert obs to torch only for the policy's forward pass
            obs_torch = torch.as_tensor(obs, device=device).float().permute(0, 3, 1, 2)
            with torch.no_grad():
                actions, _, _, _ = agent.get_action(obs_torch)
            actions = actions.cpu().numpy()

        # step the envs
        next_obs, rewards, dones, infos = envs.step(actions)

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
        for idx, done in enumerate(dones):
            if done:
                terminal_obs = infos[idx]["terminal_observation"]
                real_next_obs[idx] = terminal_obs
        rb.push(obs, actions, rewards, real_next_obs, dones)  # add transition to replay buffer
        obs = real_next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            if global_step % args.update_frequency == 0:
                # sample a batch of data
                states, actions_t, rewards_t, next_states, dones_t = rb.sample(args.batch_size)
                states = states.permute(0, 3, 1, 2)
                next_states = next_states.permute(0, 3, 1, 2)

                # compute advantages
                with torch.no_grad():
                    _, next_state_log_pi, next_state_action_probs, _ = agent.get_action(next_states)
                    next_q_values1 = target_network(next_states)
                    qf_next_target = next_state_action_probs * (next_q_values1)
                    qf_next_target = qf_next_target.sum(dim=1)
                    next_q_vals = rewards_t.flatten() + args.gamma * qf_next_target * (1 - dones_t.flatten())

                q_vals = critic(states)
                q_vals = q_vals.gather(dim=1, index=actions_t.unsqueeze(1)).squeeze(1)
                critic_loss = f.mse_loss(next_q_vals, q_vals)

                # optimize the critic QNetwork
                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()

                # Policy gradient update for the actor
                _, _, action_probs, _ = agent.get_action(states)
                with torch.no_grad():
                    q_vals = critic(states)
                actor_loss = -(action_probs * q_vals).mean()

                # optimize the actor
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

            # update target network
            if global_step % args.target_network_frequency == 0:
                for target_network_param, q_network_param in zip(target_network.parameters(), critic.parameters(), strict=False):
                    target_network_param.data.copy_(
                        args.tau * q_network_param.data + (1.0 - args.tau) * target_network_param.data,
                    )

                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/critic_loss", critic_loss.item(), global_step)
                writer.add_scalar("losses/q_values", q_vals.mean().item(), global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    # save model
    print(f"Saving model checkpoint at step {global_step} to {actor_checkpoint_path}")
    torch.save(agent.state_dict(), actor_checkpoint_path)
    torch.save(critic.state_dict(), critic_checkpoint_path)

    envs.close()
    writer.close()
