# SAC

import argparse  # noqa: I001
import random
import time
from distutils.util import strtobool
from pathlib import Path

import minigrid  # import minigrid before gym to register envs
import gymnasium as gym
import numpy as np
import torch
from torch.distributions.categorical import Categorical
from minigrid.wrappers import ImgObsWrapper, RGBImgObsWrapper
from torch.nn import functional as f
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.buffers import ReplayBuffer

import ale_py
import luau

gym.register_envs(ale_py)
gym.register_envs(minigrid)
gym.register_envs(luau)

RGB_CHANNEL = 3


# TODO: Try sb3 replay buffer
class DQNReplayBuffer:
    """Replay buffer for storing transitions experienced by the agent."""

    def __init__(self, capacity: int, n_envs: int = 1):
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
    parser.add_argument("--autotune", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, the entropy coefficient will be automatically tuned")
    parser.add_argument("--target-entropy-scale", type=float, default=0.89,
        help="the target entropy scale")
    parser.add_argument("--alpha", type=float, default=0.2,
        help="the entropy coefficient")

    args = parser.parse_args()
    # fmt: on
    return args


def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Module:
    """Initialize the layers of the network."""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Actor(nn.Module):
    """The agent class for the AC DQN algorithm."""

    def __init__(self, envs: gym.vector.SyncVectorEnv):
        super().__init__()
        # Actor network
        obs_shape = envs.observation_space.shape
        # Define image embedding
        self.image_conv = nn.Sequential(
            layer_init(nn.Conv2d(obs_shape[0], 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.Flatten(),
            nn.ReLU(),
        )
        with torch.inference_mode():
            self.image_embedding_size = self.image_conv(torch.zeros(1, *obs_shape)).shape[1]

        # Define actor's model
        self.actor = nn.Sequential(
            nn.Linear(self.image_embedding_size, 512),
            nn.ReLU(),
            nn.Linear(512, envs.action_space.n),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the network."""
        x = self.image_conv(x)
        return self.actor(x)

    def get_action(self, x: torch.Tensor) -> torch.Tensor:
        """Get the action, log probs, and probs for all actions from the actor network."""
        logits = self(x / 255.0)
        policy_dist = Categorical(logits=logits)
        action = policy_dist.sample()
        action_probs = policy_dist.probs
        log_prob = f.log_softmax(logits, dim=1)
        return action, log_prob, action_probs, policy_dist


class QNetwork(nn.Module):
    """The critic (QNetwork) class for the DQN algorithm."""

    def __init__(self, envs: gym.vector.SyncVectorEnv):
        super().__init__()
        obs_shape = envs.observation_space.shape
        # Define image embedding
        self.image_conv = nn.Sequential(
            layer_init(nn.Conv2d(obs_shape[0], 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.Flatten(),
            nn.ReLU(),
        )
        with torch.inference_mode():
            self.image_embedding_size = self.image_conv(torch.zeros(1, *obs_shape)).shape[1]

        # Define actor's model
        self.critic = nn.Sequential(
            nn.Linear(self.image_embedding_size, 512),
            nn.ReLU(),
            nn.Linear(512, envs.action_space.n),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the critic network."""
        x = self.image_conv(x / 255.0)
        return self.critic(x)


if __name__ == "__main__":
    args = parse_args()

    # tensorboard
    run_name = f"{args.gym_id}__{args.exp_name}"
    log_dir = f"runs/{run_name}"
    writer = SummaryWriter(log_dir, flush_secs=5)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n{}".format("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # model_dir
    model_dir = Path(f"models/{run_name}")
    model_dir.mkdir(parents=True, exist_ok=True)
    actor_checkpoint_path = f"{model_dir}/{run_name}_actor.pth"
    qf1_checkpoint_path = f"{model_dir}/{run_name}_qf1.pth"
    qf2_checkpoint_path = f"{model_dir}/{run_name}_qf2.pth"

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
            if capture_video and idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}", episode_trigger=lambda x: x % args.save_video_freq == 0)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            env = RGBImgObsWrapper(env)
            env = ImgObsWrapper(env)
            env = gym.wrappers.ResizeObservation(env, (84, 84))
            env = gym.wrappers.GrayscaleObservation(env)
            env = gym.wrappers.FrameStackObservation(env, 4)
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
    qf1 = QNetwork(envs).to(device)
    qf2 = QNetwork(envs).to(device)
    qf1_target = QNetwork(envs).to(device)
    qf2_target = QNetwork(envs).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())

    actor_optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-4)
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.ql_lr, eps=1e-4)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -args.target_entropy_scale * torch.log(1 / torch.tensor(envs.action_space.n))
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.ql_lr, eps=1e-4)
    else:
        alpha = args.alpha

    # replay buffer
    print(f"action space: {envs.action_space}")
    rb = ReplayBuffer(
        buffer_size=args.buffer_size,
        observation_space=envs.observation_space,
        action_space=envs.action_space,
        device=device,
        n_envs=args.num_envs,
        handle_timeout_termination=False,
    )

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
            obs_torch = torch.as_tensor(obs, device=device).float()
            with torch.no_grad():
                actions, _, _, _ = agent.get_action(obs_torch)
            actions = actions.detach().cpu().numpy()

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
        rb.add(obs, real_next_obs, actions, rewards, dones, infos)  # add transition to replay buffer
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            if global_step % args.update_frequency == 0:
                for _ in range(args.gradient_steps):
                    # sample a batch of data
                    data = rb.sample(args.batch_size)
                    states = data.observations
                    actions_t = data.actions
                    rewards_t = data.rewards
                    next_states = data.next_observations
                    dones_t = data.dones

                    # compute advantages
                    with torch.no_grad():
                        _, next_state_log_pi, next_state_action_probs, _ = agent.get_action(next_states)
                        qf1_next_target = qf1_target(next_states)
                        qf2_next_target = qf2_target(next_states)
                        min_qf_next_target = next_state_action_probs * (torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi)
                        min_qf_next_target = min_qf_next_target.sum(dim=1)
                        next_q_vals = rewards_t.flatten() + (1 - dones_t.flatten()) * args.gamma * (min_qf_next_target)

                    qf1_values = qf1(states)
                    qf2_values = qf2(states)

                    qf1_a_values = qf1_values.gather(dim=1, index=actions_t.long()).squeeze(1)
                    qf2_a_values = qf2_values.gather(dim=1, index=actions_t.long()).squeeze(1)
                    qf1_loss = f.mse_loss(qf1_a_values, next_q_vals)
                    qf2_loss = f.mse_loss(qf2_a_values, next_q_vals)
                    qf_loss = qf1_loss + qf2_loss

                    # optimize the critic QNetwork
                    q_optimizer.zero_grad()
                    qf_loss.backward()
                    q_optimizer.step()

                    # Policy gradient update for the actor
                    _, log_pi, action_probs, dist = agent.get_action(states)
                    with torch.no_grad():
                        qf1_values = qf1(states)
                        qf2_values = qf2(states)
                        min_qf_values = torch.min(qf1_values, qf2_values)

                    actor_loss = (action_probs * ((alpha * log_pi) - min_qf_values)).mean()

                    # optimize the actor
                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    if args.autotune:
                        # re-use action probabilities for temperature loss
                        alpha_loss = (action_probs.detach() * (-log_alpha.exp() * (log_pi + target_entropy).detach())).mean()
                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

            # update target network
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters(), strict=False):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters(), strict=False):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                print("SPS:", int(global_step / (time.time() - start_time)))  # don't need to see as often
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    # save model
    print(f"Saving model checkpoint at step {global_step} to {actor_checkpoint_path}")
    torch.save(agent.state_dict(), actor_checkpoint_path)
    torch.save(qf1.state_dict(), qf1_checkpoint_path)
    torch.save(qf2.state_dict(), qf2_checkpoint_path)

    envs.close()
    writer.close()
