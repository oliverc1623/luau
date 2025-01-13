import argparse
import random
import time
from distutils.util import strtobool
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from minigrid.wrappers import ImgObsWrapper
from torch import nn, optim
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.categorical import Categorical
from torch.nn import functional as f
from torch.utils.tensorboard import SummaryWriter


RGB_CHANNEL = 3


def parse_args() -> argparse.Namespace:
    """Parse the arguments for the script."""
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=str(Path(__file__).stem),
        help="the name of this experiment")
    parser.add_argument("--gym-id", type=str, default="CartPole-v1",
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
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="ppo-implementation-details",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="weather to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--num-envs", type=int, default=4,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=128,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gae", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Use GAE for advantage computation")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=4,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=4,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    parser.add_argument("--locked", type=bool, default=False,
        help="Toggle whether the environment is locked after the first observation")
    parser.add_argument("--grid-size", type=int, default=6,
        help="the size of the grid")
    parser.add_argument("--vf-clip-coef", type=float, default=10.0,
        help="the coefficient for the value clipping")
    parser.add_argument("--teacher-model", type=str, required=True,
        help="the path to the teacher model")
    parser.add_argument("--introspection-threshold", type=float, default=0.9,
        help="the threshold for introspection")
    parser.add_argument("--introspection-decay", type=float, default=0.99999,
        help="the decay rate for introspection")
    parser.add_argument("--burn-in", type=int, default=0,
        help="the burn-in period for introspection")
    parser.add_argument("--kl-loss", type=bool, default=False,
        help="Toggle kl loss")
    parser.add_argument("--kl-coef", type=float, default=0.5,
        help="kl coefficient for the loss")
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
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


def largest_divisor(n: int) -> int:
    """Find the largest divisor of batch_size that is less than or equal to horizon."""
    for i in range(n // 2, 0, -1):
        if n % i == 0:
            return i
    return 1  # If no divisors found, return 1


def adjust_beta(beta: float, kl: float, target_kl: float) -> float:
    """Adjust the beta parameter based on the KL divergence."""
    if kl > 1.5 * target_kl:
        beta *= 2  # Increase beta (reduce updates)
    elif kl < 0.5 * target_kl:
        beta /= 2  # Decrease beta (allow larger updates)
    return beta


class Agent(nn.Module):
    """The agent class for the PPO algorithm."""

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

        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(self.image_embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def get_value(self, x: torch.tensor) -> torch.tensor:
        """Get the value of the state."""
        x = self.image_conv(x)
        x = x.reshape(x.shape[0], -1)
        return self.critic(x)

    def get_action_and_value(self, x: torch.tensor, action: int | None = None) -> tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
        """Get the action and value of the state."""
        x = self.image_conv(x)
        x = x.reshape(x.shape[0], -1)
        embedding = x
        logits = self.actor(embedding)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        critic_val = self.critic(embedding)
        return action, probs.log_prob(action), probs.entropy(), critic_val


if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.gym_id}__{args.exp_name}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    model_dir = Path(f"model/{run_name}")
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = f"{model_dir}/{run_name}.pth"
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

    num_updates = args.total_timesteps // args.batch_size
    save_model_freq = largest_divisor(num_updates)

    envs = [make_env(args.seed + i, i, args.capture_video, run_name, save_model_freq) for i in range(args.num_envs)]
    envs = gym.vector.SyncVectorEnv(envs)
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Initialize teacher model
    teacher_source_agent = Agent(envs).to(device)
    teacher_source_agent.load_state_dict(torch.load(args.teacher_model))
    for param in teacher_source_agent.parameters():
        param.requires_grad = False

    teacher_target_agent = Agent(envs).to(device)
    teacher_target_agent.load_state_dict(torch.load(args.teacher_model))
    for param in list(teacher_target_agent.actor.parameters()) + list(teacher_target_agent.image_conv.parameters()):
        param.requires_grad = False
    for param in teacher_target_agent.critic.parameters():
        param.requires_grad = True
    teacher_optimizer = torch.optim.Adam(teacher_target_agent.critic.parameters(), lr=0.00001, eps=1e-5)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, info = envs.reset()
    next_obs = preprocess(next_obs)
    next_done = torch.zeros(args.num_envs).to(device)
    advice_counter = torch.zeros(args.num_envs).to(device)

    # ALGO Logic: Storage setup
    observation_shape = next_obs.shape[1:]
    obs = torch.zeros((args.num_steps, args.num_envs, *observation_shape)).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs, *envs.single_action_space.shape), dtype=torch.int64).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    indicators = torch.zeros((args.num_steps, args.num_envs)).to(device)

    for update in range(1, num_updates + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # Introspection
            h_t = torch.zeros(args.num_envs).to(device)
            probability = args.introspection_decay ** max(0, global_step - args.burn_in)
            p = Bernoulli(probability).sample([args.num_envs]).to(device)
            if global_step > args.burn_in:
                teacher_source_vals = teacher_source_agent.get_value(next_obs).flatten()
                teacher_target_vals = teacher_target_agent.get_value(next_obs).flatten()
                # Calculate absolute differences for introspection across the batch
                abs_diff = torch.abs(teacher_target_vals - teacher_source_vals)
                # Update h_t based on the introspection threshold
                h_t = (abs_diff <= args.introspection_threshold).int() * (p == 1).int()
            advice_counter += h_t
            indicators[step] = h_t

            with torch.no_grad():
                teacher_mask = h_t == 1
                student_mask = h_t == 0
                num_envs = next_obs.shape[0]
                action = torch.zeros((num_envs, *envs.single_action_space.shape), dtype=torch.int64, device=device)
                logprob = torch.zeros(num_envs, device=device)
                value = torch.zeros(num_envs, 1, device=device)
                # If any environment needs teacher actions, run teacher forward pass for that subset
                teacher_indices = teacher_mask.nonzero(as_tuple=True)[0]
                if teacher_indices.numel() > 0:
                    teacher_obs = next_obs[teacher_indices]
                    t_action, t_logprob, _, t_value = teacher_source_agent.get_action_and_value(teacher_obs)
                    action[teacher_indices] = t_action
                    logprob[teacher_indices] = t_logprob
                    value[teacher_indices] = t_value
                # If any environment needs student actions, run student forward pass for that subset
                student_indices = student_mask.nonzero(as_tuple=True)[0]
                if student_indices.numel() > 0:
                    student_obs = next_obs[student_indices]
                    s_action, s_logprob, _, s_value = agent.get_action_and_value(student_obs)
                    action[student_indices] = s_action
                    logprob[student_indices] = s_logprob
                    value[student_indices] = s_value
                # Store the final values
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, done, truncated, info = envs.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = preprocess(next_obs), torch.Tensor(done).to(device)

            if "episode" in info:
                # Extract the mask for completed episodes
                completed_mask = info["_episode"]

                # Filter the rewards and lengths for completed episodes
                episodic_returns = info["episode"]["r"][completed_mask]
                episodic_lengths = info["episode"]["l"][completed_mask]

                # Log each completed episode
                for ep_return, ep_length, ep_advice in zip(episodic_returns, episodic_lengths, advice_counter, strict=False):
                    print(f"global_step={global_step}, episodic_return={ep_return}, advice_counter={ep_advice}")
                    writer.add_scalar("charts/episodic_return", ep_return, global_step)
                    writer.add_scalar("charts/episodic_length", ep_length, global_step)
                    writer.add_scalar("charts/advice_issued", ep_advice, global_step)
                advice_counter[completed_mask] = 0

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            if args.gae:
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values
            else:
                returns = torch.zeros_like(rewards).to(device)
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        next_return = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        next_return = returns[t + 1]
                    returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return
                advantages = returns - values

        # flatten the batch
        b_obs = obs.reshape((-1), *observation_shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1, *envs.single_action_space.shape))
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Initialize corrections
        student_correction = torch.ones((args.num_steps, args.num_envs)).to(device)
        teacher_correction = torch.ones((args.num_steps, args.num_envs)).to(device)

        with torch.no_grad():
            for s in range(args.num_steps):
                # Extract batch of indicators, actions, and states for the current step across all environments
                h = indicators[s]  # Shape: (num_envs,)
                a = actions[s]  # Shape: (num_envs,)
                states = obs[s]  # Shape: (num_envs, ...)
                # Evaluate student and teacher log probabilities as a batch
                _, student_logprobs, _, _ = agent.get_action_and_value(states, a.long())  # Assuming batched input is supported
                _, teacher_logprobs, _, _ = teacher_source_agent.get_action_and_value(states, a.long())  # Assuming batched input is supported)

                # Calculate corrections using vectorized operations
                teacher_ratio = torch.clamp(torch.exp(teacher_logprobs - student_logprobs), -0.2, 0.2)
                student_ratio = torch.clamp(torch.exp(student_logprobs - teacher_logprobs), -0.2, 0.2)
                teacher_correction[s] = torch.where(h == 1, torch.ones_like(teacher_correction[s]), teacher_ratio)
                student_correction[s] = torch.where(h == 1, student_ratio, torch.ones_like(student_correction[s]))

        b_student_correction = student_correction.reshape(-1)
        b_teacher_correction = teacher_correction.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for _epoch in range(args.update_epochs):
            rng.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]
                mb_rho_s = b_student_correction[mb_inds]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = mb_advantages * mb_rho_s
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1 * mb_rho_s, pg_loss2 * mb_rho_s).mean()

                # KL divergence regularization
                kl_div = f.kl_div(b_logprobs[mb_inds], newlogprob, reduction="batchmean", log_target=True)

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2 * mb_rho_s
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.vf_clip_coef,
                        args.vf_clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2 * mb_rho_s
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = (entropy * mb_rho_s).mean()

                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
                if args.kl_loss:
                    loss = loss + args.kl_coef * kl_div
                    args.kl_coeff = adjust_beta(args.kl_coef, kl_div.item(), args.target_kl)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

                # Teacher loss
                mb_rho_t = b_teacher_correction[mb_inds]
                teacher_newvalue = teacher_target_agent.get_value(b_obs[mb_inds]).view(-1)
                teacher_v_loss = ((teacher_newvalue - b_returns[mb_inds]) ** 2 * mb_rho_t).mean()

                teacher_optimizer.zero_grad()
                teacher_v_loss.backward()
                teacher_optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/teacher_value_loss", teacher_v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        if update % save_model_freq == 0:
            print(f"Saving model checkpoint at step {global_step} to {checkpoint_path}")
            torch.save(agent.state_dict(), checkpoint_path)

    envs.close()
    writer.close()
