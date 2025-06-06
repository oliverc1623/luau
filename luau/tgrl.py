# Actor-Critic DQN

import argparse  # noqa: I001
import random
import time
from distutils.util import strtobool
from pathlib import Path
import collections

import minigrid  # import minigrid before gym to register envs  # noqa: F401
import gymnasium as gym
import numpy as np
import torch
from minigrid.wrappers import ImgObsWrapper
from torch.nn import functional as f
from torch.distributions.categorical import Categorical
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from torch.distributions import kl_divergence


RGB_CHANNEL = 3
EPS = 1e-10

gym.register(id="FourRoomDoorKey-v0", entry_point="luau.multi_room_env:FourRoomDoorKey")
gym.register(id="FourRoomDoorKeyLocked-v0", entry_point="luau.multi_room_env:FourRoomDoorKeyLocked")
gym.register(id="TrafficLight5x5-v0", entry_point="luau.traffic_light_env:TrafficLightEnv")
gym.register(id="SmallFourRoomDoorKey-v0", entry_point="luau.multi_room_env:SmallFourRoomDoorKey")
gym.register(id="SmallFourRoomDoorKeyLocked-v0", entry_point="luau.multi_room_env:SmallFourRoomDoorKeyLocked")
gym.register(id="MediumFourRoomDoorKeyLocked-v0", entry_point="luau.multi_room_env:MediumFourRoomDoorKeyLocked")


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
    parser.add_argument("--save-model-freq", type=int, default=1000,
        help="the frequency of saving the model")
    parser.add_argument("--teacher-model", type=str, default="",
        help="the path to the teacher model")
    parser.add_argument("--teacher-qnetwork", type=str, default="",
        help="the path to the teacher qnetwork")

    # Algorithm specific arguments
    parser.add_argument("--buffer-size", type=int, default=10000,
        help="the size of the replay buffer")
    parser.add_argument("--tau", type=float, default=1.0,
        help="the soft update rate")
    parser.add_argument("--target-network-frequency", type=int, default=500,
        help="the frequency of updating the target network")
    parser.add_argument("--batch-size", type=int, default=128,
        help="the batch size of the training")
    parser.add_argument("--learning-starts", type=int, default=10000,
        help="the number of steps to take before learning starts")
    parser.add_argument("--update-frequency", type=int, default=10,
        help="the frequency of training the agent")
    parser.add_argument("--num-envs", type=int, default=4,
        help="the number of parallel game environments")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gradient-steps", type=int, default=1,
        help="the number of gradient steps to take per iteration")
    parser.add_argument("--ql-lr", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")

    # Introspection specific arguments
    parser.add_argument("--history-length", type=int, default=20,
        help="number of previous rewards to take into account for TGRL update")
    parser.add_argument("--coefficient-frequency", type=int, default=1000,
        help="the frequency of updates for performance coefficient")
    parser.add_argument("--lagrange-lambda", type=float, default=9.0,
        help="the lagrange multiplier for estimating performance difference")
    parser.add_argument("--alpha", type=float, default=3.0,
        help="the coefficient for balancing the two policies")
    parser.add_argument("--mu", type=float, default=0.0003,
        help="learning rate for lagrange lambda")
    parser.add_argument("--teacher-coef", type=float, default=1.0,
        help="coefficient of the entropy")
    parser.add_argument("--teacher-coef-update", type=float, default=0.01,
        help="coefficient of the entropy")

    args = parser.parse_args()
    # fmt: on
    return args


def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Module:
    """Initialize the layers of the network."""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


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
    critic_checkpoint_path = f"{model_dir}/{run_name}_critic.pth"

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

    # teacher agent setup
    teacher_source_agent = Actor(envs).to(device)
    teacher_source_qnetwork = QNetwork(envs).to(device)
    teacher_source_agent.load_state_dict(torch.load(args.teacher_model, weights_only=True))
    teacher_source_qnetwork.load_state_dict(torch.load(args.teacher_qnetwork, weights_only=True))
    for param in teacher_source_agent.parameters():
        param.requires_grad = False
    for param in teacher_source_qnetwork.parameters():
        param.requires_grad = False

    # student agent setup
    student_agent = Actor(envs).to(device)
    student_aux = Actor(envs).to(device)
    student_agent.load_state_dict(torch.load(args.teacher_model, weights_only=True))

    # Q_R
    student_qnetwork = QNetwork(envs).to(device)
    student_target_qnetwork = QNetwork(envs).to(device)
    student_target_qnetwork.load_state_dict(student_qnetwork.state_dict())

    # Q_I
    kl_qnetwork = QNetwork(envs).to(device)
    kl_target_qnetwork = QNetwork(envs).to(device)
    kl_target_qnetwork.load_state_dict(kl_qnetwork.state_dict())

    optimizer = optim.Adam(list(student_agent.parameters()) + list(student_aux.parameters()), lr=args.learning_rate, eps=1e-4)
    q_optimizer = optim.Adam(list(student_qnetwork.parameters()) + list(kl_qnetwork.parameters()), lr=args.ql_lr, eps=1e-4)

    # aux performance tracking
    teacher_coef = args.teacher_coef
    collect_from_pi = False
    actor_performance = collections.deque(args.history_length * [0], args.history_length)
    actor_aux_performance = collections.deque(args.history_length * [0], args.history_length)
    performance_difference = 0

    current_actor = student_agent

    # replay buffer
    rb = DQNReplayBuffer(args.buffer_size, n_envs=args.num_envs)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    obs = envs.reset()
    effective_coeff = args.alpha / (1 + args.lagrange_lambda)

    for global_step in range(args.total_timesteps):
        h_t = torch.zeros(args.num_envs).to(device)
        obs_torch = torch.as_tensor(obs, device=device).float().permute(0, 3, 1, 2)
        if global_step < args.learning_starts:
            actions = np.array([envs.action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions, _, _, _ = current_actor.get_action(obs_torch)
            actions = actions.cpu().numpy()

        # step the envs
        next_obs, rewards, dones, infos = envs.step(actions)

        # record metrics for plotting
        for _, info in enumerate(infos):
            if "episode" in info:
                ep_return = info["episode"]["r"]
                ep_length = info["episode"]["l"]

                print(f"global_step={global_step}, episodic_return={ep_return}")
                if current_actor is student_agent:
                    writer.add_scalar("charts/episodic_return", ep_return, global_step)
                    writer.add_scalar("charts/episodic_length", ep_length, global_step)
                    actor_performance.appendleft(ep_return)
                    current_actor = student_aux
                else:
                    writer.add_scalar("charts/episodic_return_aux", ep_return, global_step)
                    writer.add_scalar("charts/episodic_length_aux", ep_length, global_step)
                    actor_aux_performance.appendleft(ep_return)
                    current_actor = student_agent
                break

        # get real terminal observation
        real_next_obs = next_obs.copy()
        for idx, done in enumerate(dones):
            if done:
                terminal_obs = infos[idx]["terminal_observation"]
                real_next_obs[idx] = terminal_obs

        # add to replay buffer
        rb.push(obs, actions, rewards, real_next_obs, dones)
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            if global_step % args.update_frequency == 0:
                for _ in range(args.gradient_steps):
                    # sample a batch of data
                    states, actions_t, rewards_t, next_states, dones_t = rb.sample(args.batch_size)
                    states = states.permute(0, 3, 1, 2)
                    next_states = next_states.permute(0, 3, 1, 2)

                    # compute advantages
                    with torch.no_grad():
                        # Q_R mext states
                        _, next_state_log_pi, next_state_action_probs, next_student_dist = student_agent.get_action(next_states)
                        next_q_values1 = student_target_qnetwork(next_states)
                        qf_next_target = next_state_action_probs * (next_q_values1)
                        qf_next_target = qf_next_target.sum(dim=1)
                        td_target = rewards_t.flatten() + args.gamma * qf_next_target * (1 - dones_t.flatten())

                        # compute Q_I next states
                        _, _, _, next_teacher_dist = teacher_source_agent.get_action(next_states)
                        kl_next_obs = kl_divergence(next_student_dist, next_teacher_dist)
                        next_q_kl_value = (1 - dones_t.flatten()) * args.gamma * (qf_next_target + kl_next_obs)

                    old_val = student_qnetwork(states).gather(dim=1, index=actions_t.unsqueeze(1)).squeeze(1)
                    q_kl_a_values = kl_qnetwork(states).gather(dim=1, index=actions_t.unsqueeze(1)).squeeze(1)

                    # compute the total Q loss
                    qf1loss = f.mse_loss(old_val, td_target)  # update Q_R
                    q_kl_loss = f.mse_loss(q_kl_a_values, next_q_kl_value)  # update Q_I
                    q_loss = qf1loss + q_kl_loss

                    # optimize the q networks TODO: might need to update teacher's qnetwork individually
                    q_optimizer.zero_grad()
                    q_loss.backward()
                    q_optimizer.step()

                    # Policy gradient update for the actor
                    _, _, action_probs, student_dist = student_agent.get_action(states)
                    _, _, aux_action_probs, _ = student_aux.get_action(states)
                    _, _, _, teacher_dist = teacher_source_agent.get_action(states)
                    cross_entropy = kl_divergence(student_dist, teacher_dist)
                    with torch.no_grad():
                        q_values = student_qnetwork(states)

                    pi_actor_loss = -(action_probs * q_values).sum(dim=1).mean()
                    aux_actor_loss = -(aux_action_probs * q_values).sum(dim=1).mean()
                    cross_entropy_loss = (cross_entropy).mean()

                    teacher_cross_entropy_loss = teacher_coef * cross_entropy_loss
                    overall_loss = pi_actor_loss + teacher_cross_entropy_loss + aux_actor_loss
                    overall_loss = aux_actor_loss

                    # optimize the actor
                    optimizer.zero_grad()
                    overall_loss.backward()
                    optimizer.step()

            if global_step % args.coefficient_frequency == 0:
                performance_difference = np.mean(actor_performance) - np.mean(actor_aux_performance)
                if performance_difference > 0:
                    teacher_coef = teacher_coef + args.teacher_coef_update
                else:
                    teacher_coef = teacher_coef - args.teacher_coef_update
                teacher_coef = max(teacher_coef, 0)

            # update target network
            if global_step % args.target_network_frequency == 0:
                for target_network_param, q_network_param in zip(student_target_qnetwork.parameters(), student_qnetwork.parameters(), strict=False):
                    target_network_param.data.copy_(
                        args.tau * q_network_param.data + (1.0 - args.tau) * target_network_param.data,
                    )

                for target_network_param, q_network_param in zip(kl_target_qnetwork.parameters(), kl_qnetwork.parameters(), strict=False):
                    target_network_param.data.copy_(
                        args.tau * q_network_param.data + (1.0 - args.tau) * target_network_param.data,
                    )

                writer.add_scalar("losses/actor_loss", pi_actor_loss.item(), global_step)
                writer.add_scalar("losses/teacher_cross_entropy_loss", teacher_cross_entropy_loss.item(), global_step)
                writer.add_scalar("losses/aux_actor_loss", aux_actor_loss.item(), global_step)
                writer.add_scalar("losses/QR_loss", qf1loss.item(), global_step)
                writer.add_scalar("losses/QI_loss", q_kl_loss.item(), global_step)
                writer.add_scalar("losses/q_kl_a_values", q_kl_a_values.mean().item(), global_step)
                writer.add_scalar("losses/kl_next_obs", kl_next_obs.mean().item(), global_step)
                writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                writer.add_scalar("losses/lagrange_lambda", args.lagrange_lambda, global_step)
                writer.add_scalar("losses/balance_coeff_aka_ithreshold", teacher_coef, global_step)
                writer.add_scalar("losses/performance_difference", performance_difference, global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    print(f"Saving model checkpoint at step {global_step} to {actor_checkpoint_path}")
    torch.save(student_agent.state_dict(), actor_checkpoint_path)
    torch.save(student_qnetwork.state_dict(), actor_checkpoint_path)

    envs.close()
    writer.close()
