# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import random
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as f
import tqdm
import tyro
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from tensordict import TensorDict, from_module, from_modules
from tensordict.nn import CudaGraphModule, TensorDictModule
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from torchrl.data import LazyTensorStorage, ReplayBuffer


@dataclass
class Args:
    """Arguments for the experiment."""

    exp_name: str = Path(__file__).name[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    num_envs: int = 1
    """the number of parallel environments"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "Hopper-v4"
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
    noise_clip: float = 0.5
    """noise clip parameter of the Target Policy Smoothing Regularization"""
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


def make_env(env_id: str, seed: int, idx: int, run_name: str, capture_video: int = 0) -> callable:
    """Create a gym environment with optional video recording."""

    def thunk() -> gym.Env:
        env = gym.make(env_id, render_mode="rgb_array")
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayscaleObservation(env)
        env = gym.wrappers.FrameStackObservation(env, 4)
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    """A neural network model for the Soft Q-learning algorithm with convolution on image observations."""

    def __init__(self, env: gym.Env, device=None):  # noqa: ANN001
        super().__init__()
        obs_shape = env.observation_space.shape
        action_dim = int(np.prod(env.action_space.shape))
        # image convolution layers
        self.img_conv = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, 8, stride=4, device=device),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, device=device),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, device=device),
            nn.Flatten(),
            nn.ReLU(),
        )
        # compute the output size of the convolutional layers
        with torch.inference_mode():
            dummy_input = torch.zeros(1, *obs_shape, device=device)
            conv_out = self.img_conv(dummy_input)
        conv_out_size = conv_out.shape[1]

        self.fc1 = nn.Linear(conv_out_size + action_dim, 256, device=device)
        self.fc3 = nn.Linear(256, 1, device=device)

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Forward pass for the Soft Q-learning network."""
        x = self.img_conv(x / 255.0)
        # concatenate the image embedding with the action
        x = torch.cat([x, a], 1)
        x = f.relu(self.fc1(x))
        x = self.fc3(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    """Actor network for a continuous action space in a reinforcement learning environment."""

    def __init__(self, env: gym.Env, device=None):  # noqa: ANN001
        super().__init__()
        obs_shape = env.observation_space.shape
        self.image_conv = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, 8, stride=4, device=device),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, device=device),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, device=device),
            nn.Flatten(),
            nn.ReLU(),
        )
        with torch.inference_mode():
            dummy_input = torch.zeros(1, *obs_shape, device=device)
            image_embedding = self.image_conv(dummy_input)
        self.image_embedding_size = image_embedding.shape[1]

        self.fc1 = nn.Linear(self.image_embedding_size, 256, device=device)
        self.fc_mean = nn.Linear(256, int(np.prod(env.action_space.shape)), device=device)
        self.fc_logstd = nn.Linear(256, int(np.prod(env.action_space.shape)), device=device)
        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32, device=device),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32, device=device),
        )

    def forward(self, x: torch.Tensor) -> tuple:
        """Forward pass for the Actor network."""
        # Process input images through the convolutional layers.
        x = self.image_conv(x / 255.0)
        x = f.relu(self.fc1(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mean, log_std

    def get_action(self, x: torch.Tensor) -> tuple:
        """Get action, log probability, and mean for the given input tensor."""
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # reparameterization trick
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:
            poetry run pip install "stable_baselines3==2.0.0a1"
            """,
        )

    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}"
    log_dir = f"../../pvcvolume/runs/{run_name}"
    writer = SummaryWriter(log_dir)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n{}".format("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    model_dir = Path(f"{log_dir}/models/")
    model_dir.mkdir(parents=True, exist_ok=True)
    actor_checkpoint_path = f"{model_dir}/actor.pth"
    qf1_checkpoint_path = f"{model_dir}/qf1.pth"
    qf2_checkpoint_path = f"{model_dir}/qf2.pth"

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = SubprocVecEnv([make_env(args.env_id, args.seed + i, 0, run_name, args.capture_video) for i in range(args.num_envs)])
    assert isinstance(envs.action_space, gym.spaces.Box), "only continuous action space is supported"
    envs = VecMonitor(envs, log_dir)

    max_action = float(envs.action_space.high[0])

    actor = Actor(envs).to(device)
    actor_detach = Actor(envs, device=device)
    from_module(actor).data.to_module(actor_detach)
    policy = TensorDictModule(actor_detach.get_action, in_keys=["observation"], out_keys=["action"])

    qf1 = SoftQNetwork(envs, device=device).to(device)
    qf2 = SoftQNetwork(envs, device=device).to(device)
    qnet_params = from_modules(qf1, qf2, as_module=True)
    qnet_target = qnet_params.data.clone()

    qnet = SoftQNetwork(envs, device="meta")
    qnet_params.to_module(qnet)

    q_optimizer = optim.Adam(qnet.parameters(), lr=args.q_lr, capturable=args.cudagraphs and not args.compile)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr, capturable=args.cudagraphs and not args.compile)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.detach().exp()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr, capturable=args.cudagraphs and not args.compile)
    else:
        alpha = torch.as_tensor(args.alpha, device=device)

    envs.observation_space.dtype = np.float32
    rb = ReplayBuffer(storage=LazyTensorStorage(args.buffer_size, device=device))
    start_time = time.time()

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
    obs = envs.reset()
    obs = torch.as_tensor(obs, device=device, dtype=torch.float32)
    max_ep_ret = -float("inf")
    avg_returns = deque(maxlen=20)
    for global_step in tqdm.tqdm(range(args.total_timesteps)):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions = policy(obs)
            actions = actions.detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
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
        next_obs = torch.as_tensor(next_obs, device=device, dtype=torch.float)
        real_next_obs = next_obs.clone()
        for idx, done in enumerate(dones):
            if done:
                terminal_obs = infos[idx]["terminal_observation"]
                real_next_obs[idx] = torch.as_tensor(terminal_obs, device=device, dtype=torch.float32)

        transition = TensorDict(
            observations=obs,
            next_observations=real_next_obs,
            actions=torch.as_tensor(actions, device=device, dtype=torch.float),
            rewards=torch.as_tensor(rewards, device=device, dtype=torch.float),
            terminations=dones,
            dones=dones,
            batch_size=obs.shape[0],
            device=device,
        )
        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs
        data = extend_and_sample(transition)

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            out_main = update_main(data)
            if global_step % args.policy_frequency == 0:
                for _ in range(args.policy_frequency):
                    out_main.update(update_pol(data))
                    alpha.copy_(log_alpha.detach().exp())

            # update the target networks
            if global_step % args.target_network_frequency == 0:
                # lerp is defined as x' = x + w (y-x), which is equivalent to x' = (1-w) x + w y
                qnet_target.lerp_(qnet_params.data, args.tau)

            if global_step % 1000 == 0:
                writer.add_scalar("losses/qf_loss", out_main["qf_loss"].mean(), global_step)
                writer.add_scalar("losses/actor_loss", out_main["actor_loss"].mean(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()

    print(f"Saving model checkpoint at step {global_step} to {actor_checkpoint_path}")
    torch.save(actor.state_dict(), actor_checkpoint_path)
    torch.save(qf1.state_dict(), qf1_checkpoint_path)
    torch.save(qf2.state_dict(), qf2_checkpoint_path)
