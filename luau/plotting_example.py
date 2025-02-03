# %%
import minigrid  # noqa: I001, F401
import gymnasium as gym
import matplotlib.pyplot as plt
import pandas as pd
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3.common import results_plotter
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

# %%
env = gym.make("MiniGrid-Empty-5x5-v0", render_mode="rgb_array")
env = gym.wrappers.RecordEpisodeStatistics(env)
env = ImgObsWrapper(env)

# %%
check_env(env)

# %%
Monitor(env, "test")

# %%

log_dir = "../../pvcvolume/runs2/MiniGrid-Empty-5x5-v0__DQN_Teacher_Source_Run"
# Helper from the library
results_plotter.plot_results(
    [log_dir],
    1e5,
    results_plotter.X_EPISODES,
    "DQN EmptyEnv-5x5-v0",
)

# %%

df = pd.read_csv(log_dir + "/monitor.csv").reset_index()
df.columns = df.iloc[0]
df = df[1:].reset_index(drop=True)

# %%
x, ys = results_plotter.ts2xy(df, "episodes")
x.dtype = int
ys = ys.astype(float)

# %%

plt.scatter(x, ys)

# %%
