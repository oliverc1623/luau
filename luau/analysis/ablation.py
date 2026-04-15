# %%

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


sns.set_theme(style="darkgrid")  # you can also pick 'darkgrid', 'white', etc.
# Set font family to Times New Roman
plt.rcParams["font.size"] = 34
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["font.family"] = "Times New Roman"
plt.figure(figsize=(2, 1))
sns.set_context("talk")


# %%
df_bipedal_25 = pd.read_csv("data/bwhm-lc-0.25.csv")
df_bipedal_25 = df_bipedal_25.rename(
    columns={
        "Group: iaa - episode_return": "IAA",
        "Group: iaa - episode_return__MIN": "IAA_Min",
        "Group: iaa - episode_return__MAX": "IAA_Max",
        "Group: diaa - episode_return": "DIAA",
        "Group: diaa - episode_return__MIN": "DIAA_Min",
        "Group: diaa - episode_return__MAX": "DIAA_Max",
    },
)
df_bipedal_25["Threshold"] = "0.25"

# %%
df_bipedal_75 = pd.read_csv("data/bwhm-lc-0.75.csv")
df_bipedal_75 = df_bipedal_75.rename(
    columns={
        "Group: iaa - episode_return": "IAA",
        "Group: iaa - episode_return__MIN": "IAA_Min",
        "Group: iaa - episode_return__MAX": "IAA_Max",
        "Group: diaa - episode_return": "DIAA",
        "Group: diaa - episode_return__MIN": "DIAA_Min",
        "Group: diaa - episode_return__MAX": "DIAA_Max",
    },
)
df_bipedal_75["Threshold"] = "0.75"

# %%

df_threshold_25 = pd.read_csv("data/bwhm-thr-0.25.csv")
df_threshold_25 = df_threshold_25.rename(
    columns={
        "Group: iaa - introspection_threshold": "IAA",
        "Group: diaa - introspection_threshold": "DIAA",
        "Group: iaa - introspection_threshold__MIN": "IAA_Min",
        "Group: iaa - introspection_threshold__MAX": "IAA_Max",
        "Group: diaa - introspection_threshold__MIN": "DIAA_Min",
        "Group: diaa - introspection_threshold__MAX": "DIAA_Max",
    },
)
df_threshold_25["Threshold"] = "0.25"

# %%
df_threshold_75 = pd.read_csv("data/bwhm-thr-0.75.csv")
df_threshold_75 = df_threshold_75.rename(
    columns={
        "Group: iaa - introspection_threshold": "IAA",
        "Group: diaa - introspection_threshold": "DIAA",
        "Group: iaa - introspection_threshold__MIN": "IAA_Min",
        "Group: iaa - introspection_threshold__MAX": "IAA_Max",
        "Group: diaa - introspection_threshold__MIN": "DIAA_Min",
        "Group: diaa - introspection_threshold__MAX": "DIAA_Max",
    },
)
df_threshold_75["Threshold"] = "0.75"

# %%

# Combine the two DataFrames
combined_df = pd.concat([df_bipedal_25, df_bipedal_75], ignore_index=True)

# %%

# Apply a rolling mean to smooth the curves, calculated per environment
window_size = 10
algorithms = ["IAA", "DIAA"]

# Columns to apply smoothing on
cols_to_smooth = []
for algo in algorithms:
    cols_to_smooth.extend([algo, f"{algo}_Min", f"{algo}_Max"])

# Group by environment and apply rolling mean
for col in cols_to_smooth:
    if col in combined_df.columns:
        # Use transform to apply rolling mean within each group and align results
        combined_df[col] = combined_df.groupby("Threshold")[col].transform(
            lambda x: x.rolling(window=window_size, min_periods=1).mean(),
        )

algorithms = ["IAA", "DIAA"]
for algo in algorithms:
    # Estimate standard deviation from the range
    std_dev_est = combined_df[f"{algo}_Max"] - combined_df[f"{algo}_Min"]
    # Calculate standard error and add it as a new column
    combined_df[f"{algo}_SE"] = std_dev_est / 4

# %% Sanity check plot

# Specify the algorithms you want to plot
algorithms = ["IAA", "DIAA"]  # Extend as needed
colors = sns.color_palette("Set2", 8)  # 8 discrete colors from "Set2"

# Iterate over each algorithm
for algo, color in zip(algorithms, colors, strict=False):
    # Plot the main line
    plt.plot(df_bipedal_25["Step"], df_bipedal_25[algo], color=color, label=algo)

    # Shade the min and max
    plt.fill_between(df_bipedal_25["Step"], df_bipedal_25[f"{algo}_Min"], df_bipedal_25[f"{algo}_Max"], color=color, alpha=0.2)

plt.xlabel("Step")
plt.ylabel("Episodic Returns")
plt.tight_layout()
plt.show()


# %%
df_threshold = pd.concat([df_threshold_25, df_threshold_75], ignore_index=True)

sns.set_theme(context="paper", font_scale=2.5, font="Times New Roman", style="darkgrid")

algorithms = ["DIAA", "IAA"]
palette = sns.color_palette("Set2", len(algorithms))
color_map = dict(zip(algorithms, palette, strict=False))
thresholds = ["0.25", "0.75"]

fig, axes = plt.subplots(1, 4, figsize=(24, 5))

# Learning curve panels (cols 0, 1)
for i, thr in enumerate(thresholds):
    ax = axes[i]
    df_env = combined_df[combined_df["Threshold"] == thr]
    for algo in algorithms:
        ax.plot(df_env["Step"], df_env[algo], color=color_map[algo], label=algo, linewidth=1.5)
        ax.fill_between(
            df_env["Step"],
            df_env[algo] - df_env[f"{algo}_SE"],
            df_env[algo] + df_env[f"{algo}_SE"],
            color=color_map[algo],
            alpha=0.2,
        )
    ax.set_title(f"Initial Threshold: {thr}")
    ax.set_xlabel("Step")
    if i == 0:
        ax.set_ylabel("Episodic Returns")

# Threshold panels (cols 2, 3)
for i, thr in enumerate(thresholds):
    ax = axes[2 + i]
    df_env = df_threshold[df_threshold["Threshold"] == thr]
    for algo in algorithms:
        ax.plot(df_env["Step"], df_env[algo], color=color_map[algo], label=algo, linewidth=1.5)
    ax.set_title(f"Initial Threshold: {thr}")
    ax.set_xlabel("Step")
    if i == 0:
        ax.set_ylabel("Threshold Value")

handles = [plt.Line2D([0], [0], color=color_map[a], linewidth=2, label=a) for a in algorithms]
fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.1), ncol=len(algorithms), frameon=False)
fig.tight_layout()
fig.savefig("ablation-combined.pdf", format="pdf", bbox_inches="tight")
plt.show()

# %%
