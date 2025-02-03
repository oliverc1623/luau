# %%
import matplotlib.pyplot as plt
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import load_results


# %%

log_dir = "../../../pvcvolume/runs2/MiniGrid-Empty-5x5-v0__DQN_Teacher_Source_Run"
# Helper from the library
results_plotter.plot_results(
    [log_dir],
    1e5,
    results_plotter.X_TIMESTEPS,
    "DQN EmptyEnv-5x5-v0",
)

# %%

df = load_results(log_dir)
x, ys = results_plotter.ts2xy(df, "timesteps")

# %%

plt.scatter(x, ys)

# %%
