import math
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as f
import tyro
import wandb
from custom_metadrive_env import CustomMetaDriveEnv
from torch import nn
from tqdm import tqdm


# --- Model and Environment Definitions (Copied from training script) ---

LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    """A neural network that approximates the policy for reinforcement learning."""

    def __init__(self, n_obs: int, n_act: int, device: str | None = None, action_space: gym.spaces = None):
        super().__init__()
        self.fc1 = nn.Linear(n_obs, 256, device=device)
        self.fc2 = nn.Linear(256, 256, device=device)
        self.fc_mean = nn.Linear(256, n_act, device=device)
        self.fc_logstd = nn.Linear(256, n_act, device=device)
        # action rescaling
        if action_space is not None:
            self.register_buffer(
                "action_scale",
                torch.tensor((action_space.high - action_space.low) / 2.0, dtype=torch.float32, device=device),
            )
            self.register_buffer(
                "action_bias",
                torch.tensor((action_space.high + action_space.low) / 2.0, dtype=torch.float32, device=device),
            )

    def forward(self, x: torch.tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mean, log_std

    def get_action(self, x: torch.tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get action from the policy network."""
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


# --- Argument Definitions ---


@dataclass
class Args:
    """Arguments for the evaluation script."""

    teacher_run_id: str
    """Run ID of the pretrained teacher actor model"""
    student_run_id: str
    """Run ID of the pretrained student (DIAA) actor model"""
    env_id: str = "HalfCheetah-v4"
    """The environment id of the task"""
    eval_steps: int = 5000
    """Number of steps to run the evaluation"""
    output_file: Path = Path("action_distribution_comparison.png")
    """Path to save the output plot"""
    seed: int = 42
    """Seed for reproducibility"""
    cuda: bool = True
    """Whether to use CUDA if available"""
    traffic_density: float = 0.2
    """Traffic density for the driving environment"""
    accident_prob: float = 1.0
    """Probability of accidents in the driving environment"""
    map: str = "Cy"
    """Map type for the driving environment"""


wandb.login(key="82555a3ad6bd991b8c4019a5a7a86f61388f6df1")
api = wandb.Api()


def get_model_weights(run_id: str, file_suffix: str = "_actor.pt") -> Path:
    """Retrieve the model weights from a WandB run."""
    run = api.run(f"luau/{run_id}")
    actor_file = next(f.name for f in run.files() if f.name.endswith(file_suffix))
    model_weights = wandb.restore(actor_file, run_path=f"luau/{run_id}")
    return model_weights


def run_evaluation(args: Args) -> None:
    """Load models, collects actions, and generates the comparison plot."""
    print("Starting evaluation...")
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Using device: {device}")

    # 1. Setup Environment
    env = CustomMetaDriveEnv(
        dict(
            map=args.map,
            random_lane_num=True,
            num_scenarios=1,
            start_seed=100,
            traffic_density=args.traffic_density,
            accident_prob=args.accident_prob,
        ),
    )
    video_dest_dir = Path(f"videos/inference/{args.env_id}/")
    video_dest_dir.mkdir(exist_ok=True)
    video_dest = f"{video_dest_dir}/{args.student_run_id}.gif"

    n_obs = math.prod(env.observation_space.shape)
    n_act = math.prod(env.action_space.shape)

    # 2. Load Models
    teacher_weights = get_model_weights(args.teacher_run_id)
    student_weights = get_model_weights(args.student_run_id)
    teacher_actor = Actor(n_obs, n_act, device, env.action_space).to(device)
    student_actor = Actor(n_obs, n_act, device, env.action_space).to(device)

    teacher_actor.load_state_dict(torch.load(teacher_weights.name, map_location=device))
    student_actor.load_state_dict(torch.load(student_weights.name, map_location=device))

    teacher_actor.eval()
    student_actor.eval()

    # 3. Collect Action Data
    teacher_actions_log = []
    student_actions_log = []
    obs, _ = env.reset()
    total_reward = 0
    print(f"Collecting actions for {args.eval_steps} steps...")
    for _ in tqdm(range(args.eval_steps)):
        with torch.no_grad():
            obs_tensor = torch.Tensor(obs).to(device).unsqueeze(0)
            teacher_action, _, _ = teacher_actor.get_action(obs_tensor)
            student_action, _, _ = student_actor.get_action(obs_tensor)

            teacher_actions_log.append(teacher_action.cpu().numpy().flatten())
            student_actions_log.append(student_action.cpu().numpy().flatten())
        obs, r, terminated, truncated, _ = env.step(student_actions_log[-1])
        total_reward += r
        env.render(total_reward, env.episode_step)
        if terminated or truncated:
            print(f"Episode finished with reward: {total_reward}")
            assert env
            env.top_down_renderer.generate_gif(video_dest)
            obs, _ = env.reset()
            total_reward = 0
    env.close()

    # 4. Process Data and Plot
    print("Processing data and generating plot...")
    teacher_actions = np.array(teacher_actions_log)
    student_actions = np.array(student_actions_log)

    action_dim_names = ["steering", "throttle"]
    teacher_df = pd.DataFrame(teacher_actions, columns=action_dim_names)
    teacher_df["Source"] = "Teacher"
    student_df = pd.DataFrame(student_actions, columns=action_dim_names)
    student_df["Source"] = "DIAA Student"

    output_dest = f"{args.env_id}_{args.student_run_id}"
    combined_df = pd.concat([teacher_df, student_df])
    melted_df = combined_df.melt(id_vars=["Source"], var_name="Action Dimension", value_name="Value")
    melted_df.to_csv(f"{output_dest}_actions.csv", index=False)

    g = sns.displot(
        data=melted_df,
        x="Value",
        hue="Source",
        col="Action Dimension",
        kind="kde",
        fill=True,
        alpha=0.4,
        palette={"Teacher": "blue", "DIAA Student": "red"},
        height=4,
        aspect=1.2,
        col_wrap=min(n_act, 3),
    )
    g.set_titles("{col_name}")
    g.set_xlabels("Action Value")

    # 5. Save the Output
    g.savefig(f"{output_dest}.png", bbox_inches="tight")
    print(f"✅ Plot successfully saved to {output_dest}.png")


if __name__ == "__main__":
    args = tyro.cli(Args)
    run_evaluation(args)
