# %%
import argparse
import logging
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter  # Added for TensorBoard

from luau.iaa_env import IntrospectiveEnv
from luau.ppo import IAAPPO, PPO


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Get the root path
root_path = Path(__file__).resolve().parent.parent

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
        self.has_continuous_action_space = config["has_continuous_action_space"]
        self.save_frames = config["save_frames"]
        self.horizon = config["horizon"]
        self.max_training_timesteps = config["max_training_timesteps"]
        self.print_freq = config["print_freq"]
        self.log_freq = config["log_freq"]
        self.save_model_freq = config["save_model_freq"]
        self.action_std = config["action_std"]
        self.action_std_decay_rate = config["action_std_decay_rate"]
        self.min_action_std = config["min_action_std"]
        self.action_std_decay_freq = config["action_std_decay_freq"]
        self.image_observation = config["image_observation"]
        self.run_num = config["run_num"]
        if self.algorithm == "IAAPPO":
            self.teacher_model_path = config["teacher_model_path"]
        #####################################################
        ## Note : print and log frequencies should be > than max_ep_len
        ################ PPO hyperparameters ################
        self.minibatch_size = config["minibatch_size"]
        self.k_epochs = config["k_epochs"]
        self.eps_clip = config["eps_clip"]
        self.gamma = config["gamma"]
        self.lr_actor = config["lr_actor"]
        self.lr_critic = config["lr_critic"]
        self.num_envs = config["num_envs"]
        # Set the random seed
        if random_seed is not None:
            self.random_seed = random_seed
        else:
            self.random_seed = config["random_seed"]
        torch.manual_seed(self.random_seed)
        self.rng = np.random.default_rng(self.random_seed)

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

    def _make_env(self) -> IntrospectiveEnv:
        """Create the environment."""

        def _init() -> IntrospectiveEnv:
            env = IntrospectiveEnv(size=self.size, locked=self.door_locked, max_steps=self.horizon)
            return env

        return _init

    def train(self) -> None:
        """Train the agent."""
        msg = f"Training the {self.algorithm} agent in the {self.env_name} environment."
        logging.info(msg)
        envs = [self._make_env() for _ in range(self.num_envs)]
        env = gym.vector.AsyncVectorEnv(envs, shared_memory=False)
        logging.info("Gridworld size: %s", envs[0]().max_steps)

        # make directory
        log_dir, model_dir = self.setup_directories()
        log_file = f"{log_dir}/{self.algorithm}_{self.env_name}_run_{self.run_num}_seed_{self.random_seed}_log.csv"
        checkpoint_path = f"{model_dir}/{self.algorithm}_{self.env_name}_run_{self.run_num}_seed_{self.random_seed}.pth"
        logging.info("Save checkpoint path: %s", checkpoint_path)

        # Initialize TensorBoard writer
        writer = SummaryWriter(log_dir=str(log_dir))

        # state space dimension
        state_dim = envs[0]().observation_space["image"].shape[2]
        action_dim = envs[0]().action_space.n
        logging.info("state_dim: %s \t action_dim: %s", state_dim, action_dim)
        if self.algorithm == "PPO":
            ppo_agent = PPO(state_dim, action_dim, self.lr_actor, self.gamma, self.k_epochs, self.eps_clip, self.minibatch_size)
        elif self.algorithm == "IAAPPO":
            # TODO: test we're overwriting args
            teacher_ppo_agent = PPO(state_dim, action_dim, self.lr_actor, self.gamma, self.k_epochs, self.eps_clip)
            teacher_ppo_agent.load(self.teacher_model_path)
            ppo_agent = IAAPPO(
                state_dim=state_dim,
                action_dim=action_dim,
                lr_actor=self.lr_actor,
                gamma=self.gamma,
                k_epochs=self.k_epochs,
                eps_clip=self.eps_clip,
                teacher_ppo_agent=teacher_ppo_agent,
            )
        else:
            raise ValueError("Unknown algorithm: %s", self.algorithm)
        self.print_hyperparameters()

        # track total training time
        start_time = datetime.now().astimezone().replace(microsecond=0)
        logging.info("Started training at (GMT): %s", start_time)
        logging.info("============================================================================================")

        # logging file
        log_f = Path.open(log_file, "w+")
        log_f.write("episode,timestep,reward\n")
        # printing and logging variables
        print_running_reward = 0
        print_running_episodes = 0
        log_running_reward = 0
        log_running_episodes = 0
        time_step = 0
        i_episode = 0

        # Reset all environments
        states, _ = env.reset()
        dones = np.zeros(self.num_envs, dtype=bool)
        time_step = 0

        # training loop
        for _ in range(1, self.max_training_timesteps // (self.horizon * self.num_envs) + 1):
            for _ in range(1, self.horizon + 1):
                # Select actions for all environments
                actions = []
                for env_idx in range(self.num_envs):
                    single_state = {"image": states["image"][env_idx], "direction": states["direction"][env_idx]}
                    action = ppo_agent.select_action(single_state)
                    actions.append(action)
                    time_step += 1

                # Step in all environments
                actions = np.array(actions)
                states, rewards, dones, truncated, info = env.step(actions)
                ppo_agent.buffer.rewards.extend(rewards)
                ppo_agent.buffer.is_terminals.extend(dones)

                # Update reward tracking
                log_running_reward += sum(rewards)  # sum rewards over all environments
                print_running_reward += sum(rewards)

                # Track the number of episodes for logging
                log_running_episodes += np.sum(dones)  # count completed episodes
                print_running_episodes += np.sum(dones)

                # zip truncated and dones for logging TODO: if current run bad
                dones = [a or b for a, b in zip(dones, truncated, strict=False)]

                # Log rewards and break if any environment is done
                for env_idx in range(self.num_envs):
                    if dones[env_idx]:
                        i_episode += 1

                # Log and print average reward at the specified interval
                if time_step % self.log_freq == 0:
                    log_avg_reward = log_running_reward / log_running_episodes
                    log_f.write(f"{i_episode},{time_step},{log_avg_reward}\n")
                    log_f.flush()

                    # Log to TensorBoard
                    writer.add_scalar("Average Reward", log_avg_reward, time_step)
                    writer.add_scalar("Time Step", time_step, time_step)

                    # Print average reward
                    logging.info("i_episode: %s \t\t Timestep: %s \t\t Average Reward: %s", i_episode, time_step, log_avg_reward)

                    # Reset logging variables
                    log_running_reward = 0
                    log_running_episodes = 0

            # PPO update at the end of the horizon
            ppo_agent.update()

        log_f.close()
        env.close()
        writer.close()

        # print total training time
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
        logging.info("model saving frequency: %s timesteps", self.save_model_freq)
        logging.info("log frequency: %s timesteps", self.log_freq)
        logging.info("printing average reward over episodes in last: %s timesteps", self.print_freq)
        logging.info("--------------------------------------------------------------------------------------------")
        logging.info("Initializing a discrete action space policy")
        logging.info("--------------------------------------------------------------------------------------------")
        logging.info("PPO update frequency: %s timesteps", self.horizon)
        logging.info("PPO K epochs: %s", self.k_epochs)
        logging.info("PPO epsilon clip: %s", self.eps_clip)
        logging.info("discount factor (gamma): %s", self.gamma)
        logging.info("--------------------------------------------------------------------------------------------")
        logging.info("optimizer learning rate actor: %s", self.lr_actor)
        logging.info("optimizer learning rate critic: %s", self.lr_critic)
        logging.info("============================================================================================")


# %%

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the agent.")
    parser.add_argument(
        "--config_path",
        type=str,
        default="./hyperparams/ppo-iaa-env-unlocked-config.yaml",
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

    # Load the base random seed from the config file
    with Path(args.config_path).open("r") as file:
        base_config = yaml.safe_load(file)
    base_random_seed = base_config.get("random_seed", 0)

    for i in range(args.num_experiments):
        # Generate a unique random seed for each experiment
        random_seed = base_random_seed + i
        run_id = base_config.get("run_num", 0)
        logging.info("Running experiment %s with random seed %s.", i + 1, random_seed)

        trainer = Trainer(
            config_path=args.config_path,
            log_dir=args.log_dir,
            model_dir=args.model_dir,
            random_seed=random_seed,
            run_id=run_id,
        )
        trainer.train()
