# Dynamically Introspective Action Advising (DIAA)

import argparse  # noqa: I001
import collections
import random
import time
from distutils.util import strtobool
from pathlib import Path

import minigrid  # import minigrid before gym to register envs  # noqa: F401
import gymnasium as gym
import numpy as np
import torch
from minigrid.wrappers import ImgObsWrapper
from torch.nn import functional as f
from torch import nn, optim
from torch.distributions.bernoulli import Bernoulli
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.vec_env import SubprocVecEnv


from typing import NamedTuple

RGB_CHANNEL = 3

gym.register(id="FourRoomDoorKey-v0", entry_point="luau.multi_room_env:FourRoomDoorKey")
gym.register(id="FourRoomDoorKeyLocked-v0", entry_point="luau.multi_room_env:FourRoomDoorKeyLocked")
gym.register(id="TrafficLight5x5-v0", entry_point="luau.traffic_light_env:TrafficLightEnv")
gym.register(id="SmallFourRoomDoorKey-v0", entry_point="luau.multi_room_env:SmallFourRoomDoorKey")
gym.register(id="SmallFourRoomDoorKeyLocked-v0", entry_point="luau.multi_room_env:SmallFourRoomDoorKeyLocked")
gym.register(id="MediumFourRoomDoorKeyLocked-v0", entry_point="luau.multi_room_env:MediumFourRoomDoorKeyLocked")


# Define a transition tuple for DQN
class Transition(NamedTuple):
    """A named tuple representing a single transition in the environment."""

    state: torch.tensor
    action: torch.tensor
    reward: torch.tensor
    next_state: torch.tensor
    done: torch.tensor
    indicator: torch.tensor
    timestep: torch.tensor


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
            indicator=torch.stack(batch.indicator),
            timestep=torch.stack(batch.timestep),
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
    parser.add_argument("--teacher-model", type=str, default="",
        help="the path to the teacher model")

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
    parser.add_argument("--num-envs", type=int, default=4,
        help="the number of parallel game environments")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gradient-steps", type=int, default=1,
        help="the number of gradient steps to take per iteration")

    # Introspection specific arguments
    parser.add_argument("--lagrange-lambda", type=float, default=9.0,
        help="the lagrange multiplier for estimating performance difference")
    parser.add_argument("--alpha", type=float, default=3.0,
        help="the coefficient for balancing the two policies")
    parser.add_argument("--introspection-decay", type=float, default=0.99999,
        help="the decay rate for introspection")
    parser.add_argument("--burn-in", type=int, default=0,
        help="the burn-in period for introspection")
    parser.add_argument("--history-length", type=int, default=20,
        help="number of previous rewards to take into account for TGRL update")
    parser.add_argument("--performance-coefficient-frequency", type=int, default=1000,
        help="the frequency of updates for performance coefficient")

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


class QNetwork(nn.Module):
    """The critic (QNetwork) class for the DQN algorithm."""

    def __init__(self, envs: gym.vector.SyncVectorEnv):
        super().__init__()
        c = envs.observation_space.shape[-1]
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
    log_dir = f"../../pvcvolume/runs2/{run_name}"
    writer = SummaryWriter(log_dir, flush_secs=5)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n{}".format("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # model_dir
    model_dir = Path(f"../../pvcvolume/models2/{run_name}")
    model_dir.mkdir(parents=True, exist_ok=True)
    actor_checkpoint_path = f"{model_dir}/{run_name}_actor.pth"

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    def make_env(subenv_seed: int, idx: int, capture_video: int, run_name: str, save_model_freq: int) -> callable:
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
    envs = SubprocVecEnv(envs)
    assert isinstance(envs.action_space, gym.spaces.Discrete), "only discrete action space is supported... for now"
    envs = VecMonitor(envs, log_dir)

    # teacher agent setup Q^I
    teacher_source_agent = QNetwork(envs).to(device)
    teacher_source_agent.load_state_dict(torch.load(args.teacher_model, weights_only=True))
    for param in teacher_source_agent.parameters():
        param.requires_grad = False

    # Q_teacher^new
    teacher_new_agent = QNetwork(envs).to(device)  # new task
    teacher_new_agent.load_state_dict(torch.load(args.teacher_model, weights_only=True))
    teacher_new_agent_optimizer = optim.Adam(teacher_new_agent.parameters(), lr=args.learning_rate, eps=1e-5)
    teacher_target_network = QNetwork(envs).to(device)
    teacher_target_network.load_state_dict(teacher_source_agent.state_dict())

    # student agent setup Q^R
    student_agent = QNetwork(envs).to(device)
    optimizer = optim.Adam(student_agent.parameters(), lr=args.learning_rate, eps=1e-5)
    target_network = QNetwork(envs).to(device)
    target_network.load_state_dict(student_agent.state_dict())

    collect_from_pi = False
    actor_performance = collections.deque(args.history_length * [0], args.history_length)
    actor_aux_performance = collections.deque(args.history_length * [0], args.history_length)
    performance_difference = 0

    # replay buffer
    rb = DQNReplayBuffer(args.buffer_size)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    obs = envs.reset()
    obs = preprocess(obs)
    next_done = torch.zeros(args.num_envs).to(device)
    advice_counter = torch.zeros(args.num_envs).to(device)
    effective_coeff = args.alpha / (1 + args.lagrange_lambda)

    for global_step in range(0, args.total_timesteps, args.num_envs):
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        h_t = torch.zeros(args.num_envs).to(device)
        if rng.random() < epsilon:
            actions = np.array([envs.action_space.sample() for _ in range(envs.num_envs)])
        elif collect_from_pi:
            # Introspection
            probability = args.introspection_decay ** max(0, global_step - args.burn_in)
            p = Bernoulli(probability).sample([args.num_envs]).to(device)
            if global_step > args.burn_in:
                teacher_source_q, _ = teacher_source_agent(obs).max(dim=1)
                teacher_target_q, _ = teacher_new_agent(obs).max(dim=1)
                # Calculate absolute differences for introspection across the batch
                abs_diff = torch.abs(teacher_source_q - teacher_target_q)
                # Update h_t based on the introspection threshold
                h_t = (abs_diff <= effective_coeff).int() * (p == 1).int()
            advice_counter += h_t

            if h_t.sum() > 0:
                # Get advice from the teacher for the environments where h_t is 1
                teacher_actions = torch.argmax(teacher_source_agent(obs), dim=1)  # Q^I
                student_actions = torch.argmax(student_agent(obs) + effective_coeff * teacher_new_agent(obs), dim=1)  # Q^R + Q^I
                actions = torch.where(h_t.bool(), teacher_actions, student_actions).cpu().numpy()
            else:
                # Get actions from Q^R + Q^I
                actions = torch.argmax(student_agent(obs) + effective_coeff * teacher_new_agent(obs), dim=1).cpu().numpy()
        else:
            # Get actions from Q^R
            actions = torch.argmax(student_agent(obs), dim=1).cpu().numpy()

        # step the envs
        next_obs, rewards, dones, infos = envs.step(actions)
        rewards = torch.tensor(rewards).to(device).view(-1)
        dones = torch.Tensor(dones).to(device)

        for i, info in enumerate(infos):
            if "episode" in info:
                ep_return = info["episode"]["r"]
                ep_length = info["episode"]["l"]
                ep_advice = advice_counter[i]
                print(f"global_step={global_step}, episodic_return={ep_return}")
                if collect_from_pi:
                    writer.add_scalar("charts/episodic_return", ep_return, global_step)
                    writer.add_scalar("charts/episodic_length", ep_length, global_step)
                    writer.add_scalar("charts/advice_issued", ep_advice, global_step)
                    actor_performance.appendleft(ep_return)
                    collect_from_pi = False
                else:
                    writer.add_scalar("charts/episodic_return_aux", ep_return, global_step)
                    writer.add_scalar("charts/episodic_length_aux", ep_length, global_step)
                    actor_aux_performance.appendleft(ep_return)
                    collect_from_pi = True
                break

        # get terminal observation
        real_next_obs = next_obs.copy()
        for idx, done in enumerate(dones):
            if done:
                terminal_obs = infos[idx]["terminal_observation"]
                real_next_obs[idx] = terminal_obs
                advice_counter[idx] = 0
        real_next_obs = preprocess(real_next_obs)

        # add to replay buffer
        next_obs = preprocess(next_obs)
        rb.push(obs, torch.tensor(actions), rewards, real_next_obs, dones, h_t, torch.tensor(global_step + args.num_envs))
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts * args.num_envs:
            if global_step % args.train_frequency == 0:
                for _ in range(args.gradient_steps):
                    # sample a batch of data
                    data = rb.sample(args.batch_size)
                    # compute advantages
                    with torch.no_grad():
                        next_q_values, _ = target_network(data.next_state).max(dim=1)
                        td_target = data.reward.flatten().float() + args.gamma * next_q_values.float() * (1 - data.done.flatten().float())
                    old_val = student_agent(data.state.to(device).float()).gather(1, data.action.to(device).view(-1, 1).long()).squeeze(-1)
                    loss = f.mse_loss(td_target, old_val)

                    # optimize the actor
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # update teacher new agent
                    with torch.no_grad():
                        teacher_next_q_values, _ = teacher_target_network(data.next_state).max(dim=1)
                        teacher_td_target = data.reward.flatten().float() + args.gamma * teacher_next_q_values.float() * (
                            1 - data.done.flatten().float()
                        )
                    teacher_old_val = (
                        teacher_new_agent(data.state.to(device).float())
                        .gather(
                            1,
                            data.action.to(device).view(-1, 1).long(),
                        )
                        .squeeze(-1)
                    )
                    teacher_loss = f.mse_loss(teacher_next_q_values, teacher_old_val)

                    # optimize the teacher new agent
                    teacher_new_agent_optimizer.zero_grad()
                    teacher_loss.backward()
                    teacher_new_agent_optimizer.step()

            # update target network
            if global_step % args.target_network_frequency == 0:
                for target_network_param, q_network_param in zip(target_network.parameters(), student_agent.parameters(), strict=False):
                    target_network_param.data.copy_(
                        args.tau * q_network_param.data + (1.0 - args.tau) * target_network_param.data,
                    )
                # update teacher target network
                for target_network_param, q_network_param in zip(teacher_target_network.parameters(), teacher_new_agent.parameters(), strict=False):
                    target_network_param.data.copy_(
                        args.tau * q_network_param.data + (1.0 - args.tau) * target_network_param.data,
                    )

                writer.add_scalar("losses/actor_loss", loss.item(), global_step)
                writer.add_scalar("losses/teacher_loss", teacher_loss.item(), global_step)
                writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                writer.add_scalar("losses/epsilon", epsilon, global_step)
                writer.add_scalar("losses/lagrange_lambda", args.lagrange_lambda, global_step)
                writer.add_scalar("losses/balance_coeff_aka_ithreshold", effective_coeff, global_step)
                writer.add_scalar("losses/performance_difference", performance_difference, global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

            if global_step % (args.save_model_freq - args.num_envs) == 0:
                print(f"Saving model checkpoint at step {global_step} to {actor_checkpoint_path}")
                torch.save(student_agent.state_dict(), actor_checkpoint_path)

        # compute performance difference
        if global_step % args.performance_coefficient_frequency == 0:
            performance_difference = np.mean(actor_aux_performance) - np.mean(actor_performance)
            args.lagrange_lambda = args.lagrange_lambda + performance_difference
            effective_coeff = args.alpha / (1 + args.lagrange_lambda)

    envs.close()
    writer.close()
