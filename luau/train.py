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
import yaml
from torch.utils.tensorboard import SummaryWriter

from luau.iaa_env import IntrospectiveEnv
from luau.ppo import IAAPPO, PPO


ALGORITHM_CLASSES = {
    "PPO": PPO,
    "IAAPPO": IAAPPO,
}

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Get the root path
root_path = Path(__file__).resolve().parent.parent


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

        ################ Logging hyperparameters ################
        self.algorithm = config["algorithm"]
        self.env_name = config["env_name"]
        self.door_locked = config["door_locked"]
        self.size = config["size"]
        self.save_frames = config["save_frames"]
        self.horizon = config["horizon"]
        self.max_training_timesteps = config["max_training_timesteps"]
        self.save_model_freq = config["save_model_freq"]
        self.image_observation = config["image_observation"]
        if self.algorithm == "IAAPPO":
            self.teacher_model_path = config["teacher_model_path"]
            self.introspection_decay = config["introspection_decay"]
            self.burn_in = config["burn_in"]
            self.introspection_threshold = config["introspection_threshold"]
        ################ PPO hyperparameters ################
        self.minibatch_size = config["minibatch_size"]
        self.k_epochs = config["k_epochs"]
        self.eps_clip = config["eps_clip"]
        self.gamma = config["gamma"]
        self.lr_actor = config["lr_actor"]
        self.num_envs = config["num_envs"]
        self.gae_lambda = config["gae_lambda"]

        # Store the random seed
        self.random_seed = random_seed

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

    def select_action(self, ppo_agent: PPO, state: dict, time_step: int) -> int:
        """Select an action based on the algorithm."""
        if self.algorithm == "IAAPPO":
            return ppo_agent.select_action(state, time_step)
        return ppo_agent.select_action(state)

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

    def get_ppo_agent(self, env: gym.vector.AsyncVectorEnv) -> PPO | IAAPPO:
        """Get the PPO agent."""
        # State space dimension
        state_dim = env.env_fns[0]().observation_space["image"].shape[2]
        action_dim = env.env_fns[0]().action_space.n
        logging.info("state_dim: %s \t action_dim: %s", state_dim, action_dim)

        ppo_agent = PPO(
            state_dim,
            action_dim,
            self.lr_actor,
            self.gamma,
            self.k_epochs,
            self.eps_clip,
            self.minibatch_size,
            env=env,
            horizon=self.horizon,
            num_envs=self.num_envs,
            gae_lambda=self.gae_lambda,
        )
        # Check if the specified algorithm exists in the mapping
        if self.algorithm not in ALGORITHM_CLASSES:
            msg = f"Unknown algorithm: {self.algorithm}"
            raise ValueError(msg)

        if self.algorithm == "IAAPPO":
            teacher_ppo_agent = ppo_agent
            teacher_ppo_agent.load(self.teacher_model_path)
            ppo_agent = ALGORITHM_CLASSES[self.algorithm](
                state_dim,
                action_dim,
                self.lr_actor,
                self.gamma,
                self.k_epochs,
                self.eps_clip,
                self.minibatch_size,
                env=env,
                horizon=self.horizon,
                num_envs=self.num_envs,
                gae_lambda=self.gae_lambda,
                teacher_ppo_agent=teacher_ppo_agent,
                introspection_decay=self.introspection_decay,
                burn_in=self.burn_in,
                introspection_threshold=self.introspection_threshold,
            )
        return ppo_agent

    def train(self) -> None:
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
        ppo_agent = self.get_ppo_agent(env)
        self.print_hyperparameters()

        # Track total training time
        start_time = datetime.now().astimezone().replace(microsecond=0)
        logging.info("Started training at (GMT): %s", start_time)
        logging.info("============================================================================================")

        # Logging file
        log_f = Path.open(log_file, "w+")
        log_f.write("episode,timestep,reward,episode_len\n")
        # Logging variables
        time_step = 0

        # Reset all environments
        next_obs, _ = env.reset()
        next_dones = np.zeros(self.num_envs, dtype=bool)
        time_step = 0

        # Training loop
        num_updates = self.max_training_timesteps // (self.horizon * self.num_envs)
        for update in range(1, num_updates + 1):
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * self.lr_actor
            ppo_agent.optimizer.param_groups[0]["lr"] = lrnow
            for step in range(self.horizon):
                # Preprocess the next observation and store relevant data in the PPO agent's buffer
                obs = ppo_agent.preprocess(next_obs)
                done = next_dones
                ppo_agent.buffer.images[step] = obs["image"]
                ppo_agent.buffer.directions[step] = obs["direction"]
                ppo_agent.buffer.is_terminals[step] = torch.from_numpy(done)

                # Select actions and store them in the PPO agent's buffer
                with torch.no_grad():
                    actions, action_logprobs, state_vals = ppo_agent.policy(obs)
                    ppo_agent.buffer.state_values[step] = state_vals
                ppo_agent.buffer.actions[step] = actions
                ppo_agent.buffer.logprobs[step] = action_logprobs

                # Step the environment and store the rewards
                next_obs, rewards, next_dones, truncated, info = env.step(actions.tolist())
                next_dones = np.logical_or(next_dones, truncated)
                ppo_agent.buffer.rewards[step] = torch.from_numpy(rewards)

                time_step += self.num_envs
                for k, v in info.items():
                    if k == "episode":
                        done_indx = np.argmax(next_dones)
                        episodic_reward = v["r"][done_indx]
                        episodic_length = v["l"][done_indx]
                        writer.add_scalar("charts/Episodic Reward", episodic_reward, time_step)
                        writer.add_scalar("charts/Episodic Length", episodic_length, time_step)
                        writer.add_scalar("charts/Rollout Reward", episodic_reward, update)
                        # Print average reward
                        logging.info(
                            "i_update: %s, Timestep: %s, Average Reward: %s, Episodic length: %s",
                            update,
                            time_step,
                            episodic_reward,
                            episodic_length,
                        )
                        log_f.write(f"{update},{time_step},{episodic_reward},{episodic_length}\n")
                        log_f.flush()
                    break

            # PPO update at the end of the horizon
            ppo_agent.update(next_obs, next_dones, writer, time_step)
            if update % self.save_model_freq == 0:
                logging.info("--------------------------------------------------------------------------------------------")
                logging.info("Saving model to: %s", checkpoint_path)
                logging.info("--------------------------------------------------------------------------------------------")
                ppo_agent.save(checkpoint_path)

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
