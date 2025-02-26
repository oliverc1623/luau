# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import math
import os
import random
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import highway_env  # noqa: F401
import numpy as np
import torch
import torch.nn.functional as f
import tyro
from tensordict import TensorDict, from_module, from_modules
from tensordict.nn import CudaGraphModule, TensorDictModule
from torch import nn, optim
from torch.distributions import Bernoulli
from torchrl.data import LazyTensorStorage, ReplayBuffer

import wandb


os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"


@dataclass
class Args:
    """Arguments for the SAC algorithm."""

    exp_name: str = Path(__file__).name[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "highway-fast-v0"
    """the environment id of the task"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    learning_starts: int = 5e3
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 1e-3
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""

    compile: bool = False
    """whether to use torch.compile."""
    cudagraphs: bool = False
    """whether to use cudagraphs on top of compile."""
    measure_burnin: int = 3
    """Number of burn-in iterations for speed measure."""

    teacher_actor: str = ""
    """Path to the teacher actor model for tf/distillation."""
    teacher_qnet: str = ""
    """Path to the teacher Q-network model for tf/distillation."""
    introspection_threshold: float = 0.5
    """Threshold for introspection. If the difference between the source and target Q-values is below this threshold, the teacher's action is used."""
    introspection_decay: float = 0.99999
    """Decay rate for the introspection probability. The probability of using the teacher's action decreases over time."""
    burn_in: int = 1000
    """Number of steps to burn in before starting introspection. During this period, the agent learns without using the teacher's actions."""


config = {
    "observation": {
        "type": "Kinematics",
    },
    "action": {
        "type": "ContinuousAction",
    },
    "lanes_count": 2,
    "vehicles_count": 10,
    "duration": 50,
    "initial_spacing": 2,
    "collision_reward": -1,
    "reward_speed_range": [20, 30],
    "simulation_frequency": 5,
    "policy_frequency": 1,
    "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
    "screen_width": 600,
    "screen_height": 150,
    "centering_position": [0.3, 0.5],
    "scaling": 5.5,
    "show_trajectories": False,
    "render_agent": False,
    "offscreen_rendering": False,
}


def make_env(env_id: str, seed: int, idx: int, capture_video: int, run_name: str) -> callable:
    """Create and configure a gym environment based on the provided parameters."""

    def thunk() -> gym.Env:
        if capture_video and idx == 0:
            env = gym.make(env_id, config=config, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id, config=config)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    """A neural network that approximates the soft Q-function for reinforcement learning."""

    def __init__(self, env: gym.Env, n_act: int, n_obs: int, device: str | None = None):  # noqa: ARG002
        super().__init__()
        self.fc1 = nn.Linear(n_act + n_obs, 256, device=device)
        self.fc2 = nn.Linear(256, 256, device=device)
        self.fc3 = nn.Linear(256, 1, device=device)

    def forward(self, x: torch.tensor, a: torch.tensor) -> torch.Tensor:
        """Forward pass."""
        x = x.view(x.shape[0], -1)  # flatten the input
        x = torch.cat([x, a], 1)
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = self.fc3(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    """A neural network that approximates the policy for reinforcement learning."""

    def __init__(self, env: gym.Env, n_obs: int, n_act: int, device: str | None = None):
        super().__init__()
        self.fc1 = nn.Linear(n_obs, 256, device=device)
        self.fc2 = nn.Linear(256, 256, device=device)
        self.fc_mean = nn.Linear(256, n_act, device=device)
        self.fc_logstd = nn.Linear(256, n_act, device=device)
        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor((env.single_action_space.high - env.single_action_space.low) / 2.0, dtype=torch.float32, device=device),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor((env.single_action_space.high + env.single_action_space.low) / 2.0, dtype=torch.float32, device=device),
        )

    def forward(self, x: torch.tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        x = x.view(x.shape[0], -1)  # flatten the input
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x: torch.tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get action from the policy network."""
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


if __name__ == "__main__":
    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{args.compile}__{args.cudagraphs}"

    wandb.init(
        project="sac_continuous_action",
        name=f"{Path(__file__).stem}-{run_name}",
        config=vars(args),
        save_code=True,
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np_rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, run_name)])
    n_act = math.prod(envs.single_action_space.shape)
    n_obs = math.prod(envs.single_observation_space.shape)
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    max_action = float(envs.single_action_space.high[0])

    actor = Actor(envs, device=device, n_act=n_act, n_obs=n_obs)
    actor_detach = Actor(envs, device=device, n_act=n_act, n_obs=n_obs)
    # Copy params to actor_detach without grad
    from_module(actor).data.to_module(actor_detach)
    policy = TensorDictModule(actor_detach.get_action, in_keys=["observation"], out_keys=["action"])

    def get_q_params() -> tuple[TensorDict, TensorDict, SoftQNetwork]:
        """Initialize the Q-function parameters and target network."""
        qf1 = SoftQNetwork(envs, device=device, n_act=n_act, n_obs=n_obs)
        qf2 = SoftQNetwork(envs, device=device, n_act=n_act, n_obs=n_obs)
        qnet_params = from_modules(qf1, qf2, as_module=True)
        qnet_target = qnet_params.data.clone()

        # discard params of net
        qnet = SoftQNetwork(envs, device="meta", n_act=n_act, n_obs=n_obs)
        qnet_params.to_module(qnet)

        return qnet_params, qnet_target, qnet

    def load_q_params(filepath: str) -> tuple[TensorDict, TensorDict, SoftQNetwork]:
        """Load the Q-function parameters and target network from a file."""
        qnet_params, qnet_target, qnet = get_q_params()
        qnet_params.load_state_dict(torch.load(filepath))
        qnet_target.load_state_dict(qnet_params.data.state_dict())
        return qnet_params, qnet_target, qnet

    qnet_params, qnet_target, qnet = get_q_params()
    teacher_qnet_params, teacher_qnet_target, teacher_qnet = get_q_params()

    # teacher actor and qnet
    assert args.teacher_actor != "" or args.teacher_qnet != "", "teacher actor and qnet must be provided for distillation"
    teacher_actor = Actor(envs, device=device, n_act=n_act, n_obs=n_obs)
    teacher_actor.load_state_dict(torch.load(args.teacher_actor, map_location=device))
    for param in teacher_actor.parameters():
        param.requires_grad = False
    teacher_qnet.load_state_dict(torch.load(args.teacher_qnet, map_location=device))

    q_optimizer = optim.Adam(list(qnet.parameters()) + list(teacher_qnet.parameters()), lr=args.q_lr, capturable=args.cudagraphs and not args.compile)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr, capturable=args.cudagraphs and not args.compile)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.detach().exp()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr, capturable=args.cudagraphs and not args.compile)
    else:
        alpha = torch.as_tensor(args.alpha, device=device)

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(storage=LazyTensorStorage(args.buffer_size, device=device))

    def batched_qf(params: any, obs: any, action: any, next_q_value=None) -> torch.Tensor:  # noqa: ANN001
        """Compute the Q-value for a batch of observations and actions using the provided parameters."""
        with params.to_module(qnet):
            vals = qnet(obs, action)
            if next_q_value is not None:
                loss_val = f.mse_loss(vals.view(-1), next_q_value)
                return loss_val
            return vals

    def update_main(data: any) -> TensorDict:
        """Update the main Q-function and policy networks."""
        # optimize the model
        q_optimizer.zero_grad()
        with torch.no_grad():
            next_state_actions, next_state_log_pi, _ = actor.get_action(data["next_observations"])
            qf_next_target = torch.vmap(batched_qf, (0, None, None))(
                qnet_target,
                data["next_observations"],
                next_state_actions,
            )
            min_qf_next_target = qf_next_target.min(dim=0).values - alpha * next_state_log_pi  # noqa: PD011
            next_q_value = data["rewards"].flatten() + (~data["dones"].flatten()).float() * args.gamma * min_qf_next_target.view(-1)

        qf_a_values = torch.vmap(batched_qf, (0, None, None, None))(
            qnet_params,
            data["observations"],
            data["actions"],
            next_q_value,
        )
        qf_loss = qf_a_values.sum(0)

        qf_loss.backward()
        q_optimizer.step()
        return TensorDict(qf_loss=qf_loss.detach())

    def update_pol(data: any) -> TensorDict:
        """Update the policy network."""
        actor_optimizer.zero_grad()
        pi, log_pi, _ = actor.get_action(data["observations"])
        qf_pi = torch.vmap(batched_qf, (0, None, None))(qnet_params.data, data["observations"], pi)
        min_qf_pi = qf_pi.min(0).values  # noqa: PD011
        actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

        actor_loss.backward()
        actor_optimizer.step()

        if args.autotune:
            a_optimizer.zero_grad()
            with torch.no_grad():
                _, log_pi, _ = actor.get_action(data["observations"])
            alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

            alpha_loss.backward()
            a_optimizer.step()
        return TensorDict(alpha=alpha.detach(), actor_loss=actor_loss.detach(), alpha_loss=alpha_loss.detach())

    def extend_and_sample(transition: any) -> TensorDict:
        """Extend the replay buffer and sample a batch of transitions."""
        rb.extend(transition)
        return rb.sample(args.batch_size)

    is_extend_compiled = False
    if args.compile:
        mode = None  # "reduce-overhead" if not args.cudagraphs else None
        update_main = torch.compile(update_main, mode=mode)
        update_pol = torch.compile(update_pol, mode=mode)
        policy = torch.compile(policy, mode=mode)

    if args.cudagraphs:
        update_main = CudaGraphModule(update_main, in_keys=[], out_keys=[])
        update_pol = CudaGraphModule(update_pol, in_keys=[], out_keys=[])

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset()
    obs = torch.as_tensor(obs, device=device, dtype=torch.float)
    start_time = None
    max_ep_ret = -float("inf")
    avg_returns = deque(maxlen=20)
    aux_performance = deque(maxlen=20)
    iaa_performance = deque(maxlen=20)
    desc = ""
    avg_advice = deque(maxlen=20)
    h_t = 0
    for global_step in range(args.total_timesteps):
        if global_step == args.measure_burnin + args.learning_starts:
            start_time = time.time()
            measure_burnin = global_step

        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            # Introspection
            obs_torch = torch.Tensor(obs).to(device)
            probability = args.introspection_decay ** max(0, global_step - args.burn_in)
            teacher_actions, _, _ = teacher_actor.get_action(obs_torch)
            student_actions = policy(obs)
            p = Bernoulli(probability).sample([envs.num_envs]).to(device)
            if global_step > args.burn_in:
                teacher_source_q = torch.vmap(batched_qf, (0, None, None))(teacher_qnet_params, obs_torch, teacher_actions).min(dim=0).values  # noqa: PD011
                teacher_target_q = torch.vmap(batched_qf, (0, None, None))(teacher_qnet_target, obs_torch, teacher_actions).min(dim=0).values  # noqa: PD011
                abs_diff = torch.abs(teacher_source_q - teacher_target_q)
                h_t = (abs_diff <= args.introspection_threshold).int() * (p == 1).int()
                avg_advice.append(h_t)
            if h_t:
                actions = teacher_actions.cpu().numpy()
            else:
                actions = student_actions.cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        if h_t:
            iaa_performance.append(rewards)
        else:
            aux_performance.append(rewards)
        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        next_obs = torch.as_tensor(next_obs, device=device, dtype=torch.float)
        real_next_obs = next_obs.clone()

        if "episode" in infos:
            # Extract the mask for completed episodes
            completed_mask = infos["_episode"]
            episodic_returns = infos["episode"]["r"][completed_mask]
            episodic_lengths = infos["episode"]["l"][completed_mask]

            # Log each completed episode
            for ep_return, _ in zip(episodic_returns, episodic_lengths, strict=False):
                max_ep_ret = max(max_ep_ret, ep_return)
                avg_returns.append(ep_return)
                desc = f"global_step={global_step}, episodic_return={torch.tensor(avg_returns).mean(): 4.2f}, \
                    normalized_reward={rewards[0]: 4.2f}, advice={torch.tensor(avg_advice, dtype=torch.float32).mean(): 4.2f}"
                print(desc)
                break

        transition = TensorDict(
            observations=obs,
            next_observations=real_next_obs,
            actions=torch.as_tensor(actions, device=device, dtype=torch.float),
            rewards=torch.as_tensor(rewards, device=device, dtype=torch.float),
            terminations=terminations,
            dones=terminations,
            batch_size=obs.shape[0],
            device=device,
        )

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs
        data = extend_and_sample(transition)

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            out_main = update_main(data)
            if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                for _ in range(args.policy_frequency):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    out_main.update(update_pol(data))

                    alpha.copy_(log_alpha.detach().exp())

            # update the target networks
            if global_step % args.target_network_frequency == 0:
                # lerp is defined as x' = x + w (y-x), which is equivalent to x' = (1-w) x + w y
                qnet_target.lerp_(qnet_params.data, args.tau)
                teacher_qnet_target.lerp_(teacher_qnet_params.data, args.tau)

            if global_step % 100 == 0 and start_time is not None:
                speed = (global_step - measure_burnin) / (time.time() - start_time)
                with torch.no_grad():
                    logs = {
                        "episode_return": torch.tensor(avg_returns).mean(),
                        "actor_loss": out_main["actor_loss"].mean(),
                        "alpha_loss": out_main.get("alpha_loss", 0),
                        "qf_loss": out_main["qf_loss"].mean(),
                        "normalized_reward": rewards.mean(),
                        "advice": torch.tensor(avg_advice, dtype=torch.float32).mean(),
                        "introspection_threshold": args.introspection_threshold,
                    }
                wandb.log(
                    {
                        "speed": speed,
                        **logs,
                    },
                    step=global_step,
                )
            if global_step % 1000 == 0:
                performance_difference = np.mean(iaa_performance) - np.mean(aux_performance)
                if performance_difference > 0:
                    args.introspection_threshold += 0.1
                else:
                    args.introspection_threshold -= 0.1

    # save the model
    torch.save(actor.state_dict(), f"../../pvcvolume/models/{run_name}_actor.pt")
    torch.save(qnet.state_dict(), f"../../pvcvolume/models/{run_name}_qnet.pt")
    envs.close()
