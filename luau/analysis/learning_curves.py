# %%

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


sns.set_theme(style="darkgrid")  # you can also pick 'darkgrid', 'white', etc.
# Set font family to Times New Roman
plt.rcParams["font.size"] = 34
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["font.family"] = "Times New Roman"
plt.figure(figsize=(5, 4))
sns.set_context("talk")

# %%
df_bipedal = pd.read_csv("data/bipedal-learning-curves.csv")
df_bipedal = df_bipedal.rename(
    columns={
        "Group: sac-baseline - episode_return": "Baseline",
        "Group: sac-baseline - episode_return__MIN": "Baseline_Min",
        "Group: sac-baseline - episode_return__MAX": "Baseline_Max",
        "Group: sac-finetune - episode_return": "Finetune",
        "Group: sac-finetune - episode_return__MIN": "Finetune_Min",
        "Group: sac-finetune - episode_return__MAX": "Finetune_Max",
        "Group: sac-iaa - episode_return": "IAA",
        "Group: sac-iaa - episode_return__MIN": "IAA_Min",
        "Group: sac-iaa - episode_return__MAX": "IAA_Max",
        "Group: sac-diaa - episode_return": "DIAA",
        "Group: sac-diaa - episode_return__MIN": "DIAA_Min",
        "Group: sac-diaa - episode_return__MAX": "DIAA_Max",
    },
)
df_bipedal["Environment"] = "BWHM"

# %%

df_lunar = pd.read_csv("data/lunar-round2.csv")
df_lunar = df_lunar.rename(
    columns={
        "Group: sac-baseline-lunarlander-v3 - episode_return": "Baseline",
        "Group: sac-baseline-lunarlander-v3 - episode_return__MIN": "Baseline_Min",
        "Group: sac-baseline-lunarlander-v3 - episode_return__MAX": "Baseline_Max",
        "Group: sac-diaa-lunarlander-v3 - episode_return": "DIAA",
        "Group: sac-diaa-lunarlander-v3 - episode_return__MIN": "DIAA_Min",
        "Group: sac-diaa-lunarlander-v3 - episode_return__MAX": "DIAA_Max",
        "Group: sac-iaa-lunarlander-v3 - episode_return": "IAA",
        "Group: sac-iaa-lunarlander-v3 - episode_return__MIN": "IAA_Min",
        "Group: sac-iaa-lunarlander-v3 - episode_return__MAX": "IAA_Max",
        "Group: sac-finetune-lunarlander-v3 - episode_return": "Finetune",
        "Group: sac-finetune-lunarlander-v3 - episode_return__MIN": "Finetune_Min",
        "Group: sac-finetune-lunarlander-v3 - episode_return__MAX": "Finetune_Max",
    },
)
df_lunar["Environment"] = "LLWE"

# %%

df_curveroad = pd.read_csv("data/curve-merge-learning-curves.csv")
df_curveroad = df_curveroad.dropna()
df_curveroad = df_curveroad.rename(
    columns={
        "Group: sac-baseline - episode_return": "Baseline",
        "Group: sac-baseline - episode_return__MIN": "Baseline_Min",
        "Group: sac-baseline - episode_return__MAX": "Baseline_Max",
        "Group: finetune - episode_return": "Finetune",
        "Group: finetune - episode_return__MIN": "Finetune_Min",
        "Group: finetune - episode_return__MAX": "Finetune_Max",
        "Group: iaa - episode_return": "IAA",
        "Group: iaa - episode_return__MIN": "IAA_Min",
        "Group: iaa - episode_return__MAX": "IAA_Max",
        "Group: diaa - episode_return": "DIAA",
        "Group: diaa - episode_return__MIN": "DIAA_Min",
        "Group: diaa - episode_return__MAX": "DIAA_Max",
    },
)
df_curveroad["Environment"] = "CM"

# %%

df_tintersection = pd.read_csv("data/t-inter.csv")
df_tintersection = df_tintersection.dropna()
df_tintersection = df_tintersection.rename(
    columns={
        "Group: sac-baseline - episode_return": "Baseline",
        "Group: sac-baseline - episode_return__MIN": "Baseline_Min",
        "Group: sac-baseline - episode_return__MAX": "Baseline_Max",
        "Group: finetune - episode_return": "Finetune",
        "Group: finetune - episode_return__MIN": "Finetune_Min",
        "Group: finetune - episode_return__MAX": "Finetune_Max",
        "Group: iaa - episode_return": "IAA",
        "Group: iaa - episode_return__MIN": "IAA_Min",
        "Group: iaa - episode_return__MAX": "IAA_Max",
        "Group: diaa - episode_return": "DIAA",
        "Group: diaa - episode_return__MIN": "DIAA_Min",
        "Group: diaa - episode_return__MAX": "DIAA_Max",
    },
)
df_tintersection["Environment"] = "T-Int."

# %%

df_merge_turn = pd.read_csv("data/merge-turn.csv")
df_merge_turn = df_merge_turn.dropna()
df_merge_turn = df_merge_turn.rename(
    columns={
        "Group: sac-baseline - episode_return": "Baseline",
        "Group: sac-baseline - episode_return__MIN": "Baseline_Min",
        "Group: sac-baseline - episode_return__MAX": "Baseline_Max",
        "Group: finetune - episode_return": "Finetune",
        "Group: finetune - episode_return__MIN": "Finetune_Min",
        "Group: finetune - episode_return__MAX": "Finetune_Max",
        "Group: iaa - episode_return": "IAA",
        "Group: iaa - episode_return__MIN": "IAA_Min",
        "Group: iaa - episode_return__MAX": "IAA_Max",
        "Group: diaa - episode_return": "DIAA",
        "Group: diaa - episode_return__MIN": "DIAA_Min",
        "Group: diaa - episode_return__MAX": "DIAA_Max",
    },
)
df_merge_turn["Environment"] = "MT"

# %%

combined_df = pd.concat(
    [
        df_lunar,
        df_bipedal,
        df_tintersection,
        df_curveroad,
        df_merge_turn,
    ],
    ignore_index=True,
)

# Apply a rolling mean to smooth the curves, calculated per environment
window_size = 10
algorithms = ["Baseline", "Finetune", "IAA", "DIAA"]

# Columns to apply smoothing on
cols_to_smooth = []
for algo in algorithms:
    cols_to_smooth.extend([algo, f"{algo}_Min", f"{algo}_Max"])

# Group by environment and apply rolling mean
for col in cols_to_smooth:
    if col in combined_df.columns:
        # Use transform to apply rolling mean within each group and align results
        combined_df[col] = combined_df.groupby("Environment")[col].transform(
            lambda x: x.rolling(window=window_size, min_periods=1).mean(),
        )

# %%

algorithms = ["Baseline", "Finetune", "IAA", "DIAA"]
for algo in algorithms:
    # Estimate standard deviation from the range
    std_dev_est = combined_df[f"{algo}_Max"] - combined_df[f"{algo}_Min"]
    # Calculate standard error and add it as a new column
    combined_df[f"{algo}_SE"] = std_dev_est / 4

# %% Sanity check plot

# Specify the algorithms you want to plot
algorithms = ["Baseline", "Finetune", "IAA", "DIAA"]  # Extend as needed
colors = sns.color_palette("Set2", 8)  # 8 discrete colors from "Set2"

# Iterate over each algorithm
for algo, color in zip(algorithms, colors, strict=False):
    # Plot the main line
    plt.plot(df_tintersection["Step"], df_tintersection[algo], color=color, label=algo)

    # Shade the min and max
    plt.fill_between(df_tintersection["Step"], df_tintersection[f"{algo}_Min"], df_tintersection[f"{algo}_Max"], color=color, alpha=0.2)

plt.xlabel("Step")
plt.ylabel("Episodic Returns")
plt.tight_layout()
plt.show()


# %%


def generate_learningcurve_facets(df: pd.DataFrame) -> None:
    """Generate a faceted learning curve plot, separated by environment."""
    # 1. Define algorithms and melt the DataFrame to a "tidy" format.
    #    'Environment' is now an identifier variable.
    algorithms = ["Baseline", "DIAA", "IAA", "Finetune"]
    df_long = pd.melt(
        df,
        id_vars=["Step", "Environment"],
        value_vars=algorithms,
        var_name="Algorithm",
        value_name="Episodic Returns",
    )

    # 2. Create the faceted plot using sns.relplot.
    #    This function returns a FacetGrid object.
    g = sns.relplot(
        data=df_long,
        x="Step",
        y="Episodic Returns",
        hue="Algorithm",
        col="Environment",  # This creates the columns of subplots
        kind="line",
        palette="Set2",
        height=4,  # Height of each facet in inches
        aspect=1.0,  # Aspect ratio of each facet
        facet_kws={"sharey": False},
        linewidth=1,
    )

    # 3. Add the shaded min/max regions to each subplot (facet).
    #    We need to iterate through the axes of the FacetGrid.
    for env_name, ax in g.axes_dict.items():
        # Filter the original wide-format DataFrame for the specific environment
        df_env = df[df["Environment"] == env_name]

        # Get the color mapping from the plot's legend
        handles, labels = g.axes.flat[0].get_legend_handles_labels()
        color_map = {label: handle.get_color() for label, handle in zip(labels, handles, strict=False)}

        # Add a shaded region for each algorithm
        for algo in algorithms:
            color = color_map[algo]
            ax.fill_between(
                df_env["Step"],
                df_env[f"{algo}"] - df_env[f"{algo}_SE"],
                df_env[f"{algo}"] + df_env[f"{algo}_SE"],
                color=color,
                alpha=0.2,
            )
    g.set_titles(col_template="{col_name}")
    sns.move_legend(
        g,
        "lower center",
        bbox_to_anchor=(0.5, -0.05),  # Position it horizontally centered, just above the plots
        ncols=len(algorithms),  # Display all items in a single row
        title=None,  # Remove the legend title
        frameon=False,  # Remove the legend box frame
    )

    # 4. Set overall title and save the figure
    g.tight_layout()
    g.savefig("learning-curves-facet.pdf", format="pdf")
    plt.show()


# Generate the plot
generate_learningcurve_facets(combined_df)

# %%
