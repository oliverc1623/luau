# %%
import argparse
import logging
import os
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from minigrid.wrappers import RGBImgObsWrapper

from luau.iaa_env import IntrospectiveEnv
from luau.ppo import PPO


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
        self.env_name = config["env_name"]
        self.door_locked = config["door_locked"]
        self.size = config["size"]
        self.has_continuous_action_space = config["has_continuous_action_space"]
        self.save_frames = config["save_frames"]
        self.max_ep_len = config["max_ep_len"]
        self.max_training_timesteps = config["max_training_timesteps"]
        self.print_freq = config["print_freq"]
        self.log_freq = config["log_freq"]
        self.save_model_freq = config["save_model_freq"]
        self.action_std = config["action_std"]
        self.action_std_decay_rate = config["action_std_decay_rate"]
        self.min_action_std = config["min_action_std"]
        self.action_std_decay_freq = config["action_std_decay_freq"]
        self.image_observation = config["image_observation"]
        self.run_num_pretrained = config["run_num"]
        #####################################################
        ## Note : print and log frequencies should be > than max_ep_len
        ################ PPO hyperparameters ################
        self.update_timestep = self.max_ep_len * 4
        self.k_epochs = config["k_epochs"]
        self.eps_clip = config["eps_clip"]
        self.gamma = config["gamma"]
        self.lr_actor = config["lr_actor"]
        self.lr_critic = config["lr_critic"]
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
            log_dir = Path(f"{self.log_dir}/PPO_logs/{self.env_name}/run_{self.run_id}")
        else:
            log_dir = Path(f"./PPO_logs/{self.env_name}/")
        log_dir.mkdir(parents=True, exist_ok=True)

        if self.model_dir is not None:
            model_dir = Path(f"{self.model_dir}/models/{self.env_name}/run_{self.run_id}")
        else:
            model_dir = Path(f"./models/{self.env_name}/")
        model_dir.mkdir(parents=True, exist_ok=True)

        return log_dir, model_dir

    def train(self) -> None:
        """Train the agent."""
        msg = f"Training the agent in the {self.env_name} environment."
        logging.info(msg)
        env = IntrospectiveEnv(size=self.size, locked=self.door_locked)
        if self.image_observation:
            env = RGBImgObsWrapper(env)
        logging.info("Gridworld size: %s", env.max_steps)

        # make directory
        log_dir, model_dir = self.setup_directories()
        log_file = f"{log_dir}/PPO_{self.env_name}_log_{len(next(os.walk(log_dir))[2])}.csv"
        checkpoint_path = f"{model_dir}/PPO_{self.env_name}_{self.random_seed}_{self.run_num_pretrained}.pth"
        print("save checkpoint path : " + checkpoint_path)

        # state space dimension
        state_dim = env.observation_space["image"].shape[2]
        action_dim = env.action_space.n
        print(f"state_dim : {state_dim} \t action_dim : {action_dim}")
        ppo_agent = PPO(state_dim, action_dim, self.lr_actor, self.gamma, self.k_epochs, self.eps_clip)
        self.print_hyperparameters()

        # track total training time
        start_time = datetime.now().astimezone().replace(microsecond=0)
        print("Started training at (GMT) : ", start_time)
        print("============================================================================================")
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

        # training loop
        while time_step <= self.max_training_timesteps:
            state, _ = env.reset()
            current_ep_reward = 0
            for _ in range(1, self.max_ep_len + 1):
                # select action with policy
                action = ppo_agent.select_action(state)
                state, reward, done, truncated, info = env.step(action)
                # saving reward and is_terminals
                ppo_agent.buffer.rewards.append(reward)
                ppo_agent.buffer.is_terminals.append(done)

                if self.save_frames:
                    img = env.render()
                    plt.imsave(f"frames/frame_{time_step:06}.png", img)

                time_step += 1
                current_ep_reward += reward

                # update PPO agent
                if time_step % self.update_timestep == 0:
                    ppo_agent.update()

                # log in logging file
                if time_step % self.log_freq == 0:
                    # log average reward till last episode
                    log_avg_reward = log_running_reward / log_running_episodes
                    log_avg_reward = round(log_avg_reward, 4)
                    log_f.write(f"{i_episode},{time_step},{log_avg_reward}\n")
                    log_f.flush()
                    log_running_reward = 0
                    log_running_episodes = 0

                # printing average reward
                if time_step % self.print_freq == 0:
                    # print average reward till last episode
                    print_avg_reward = print_running_reward / print_running_episodes
                    print_avg_reward = round(print_avg_reward, 2)
                    print(f"Episode : {i_episode} \t\t Timestep : {time_step} \t\t Average Reward : {print_avg_reward}")
                    print_running_reward = 0
                    print_running_episodes = 0

                # save model weights
                if time_step % self.save_model_freq == 0:
                    print("--------------------------------------------------------------------------------------------")
                    print("saving model at : " + checkpoint_path)
                    ppo_agent.save(checkpoint_path)
                    print("model saved")
                    pacific_time = datetime.now().astimezone().replace(microsecond=0)
                    print("Elapsed Time  : ", pacific_time - start_time)
                    print("--------------------------------------------------------------------------------------------")

                # break; if the episode is over
                if done:
                    break

            print_running_reward += current_ep_reward
            print_running_episodes += 1
            log_running_reward += current_ep_reward
            log_running_episodes += 1
            i_episode += 1

        log_f.close()
        env.close()

        # print total training time
        print("============================================================================================")
        end_time = datetime.now().astimezone().replace(microsecond=0)
        print("Started training at (GMT) : ", start_time)
        print("Finished training at (GMT) : ", end_time)
        print("Total training time  : ", end_time - start_time)
        print("============================================================================================")

    def print_hyperparameters(self) -> None:
        """Print the hyperparameters."""
        print("--------------------------------------------------------------------------------------------")
        print(f"Training the agent in the {self.env_name} environment. Door: {self.door_locked}")
        print(f"max training timesteps : {self.max_training_timesteps}")
        print(f"max timesteps per episode : {self.max_ep_len}")
        print(f"model saving frequency : {self.save_model_freq} timesteps")
        print(f"log frequency : {self.log_freq} timesteps")
        print(f"printing average reward over episodes in last : {self.print_freq} timesteps")
        print("--------------------------------------------------------------------------------------------")
        print("Initializing a discrete action space policy")
        print("--------------------------------------------------------------------------------------------")
        print(f"PPO update frequency : {self.update_timestep} timesteps")
        print(f"PPO K epochs : {self.k_epochs}")
        print(f"PPO epsilon clip : {self.eps_clip}")
        print(f"discount factor (gamma) : {self.gamma}")
        print("--------------------------------------------------------------------------------------------")
        print(f"optimizer learning rate actor : {self.lr_actor}")
        print(f"optimizer learning rate critic : {self.lr_critic}")
        print("============================================================================================")


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
        run_id = f"seed_{random_seed}"
        print(f"Running experiment {i + 1} with random seed {random_seed}.")

        trainer = Trainer(
            config_path=args.config_path,
            log_dir=args.log_dir,
            model_dir=args.model_dir,
            random_seed=random_seed,
            run_id=run_id,
        )
        trainer.train()
