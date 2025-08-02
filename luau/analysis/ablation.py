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
def generate_learningcurve_facets(df: pd.DataFrame) -> None:
    """Generate a faceted learning curve plot, separated by environment."""
    # 1. Define algorithms and melt the DataFrame to a "tidy" format.
    #    'Environment' is now an identifier variable.
    algorithms = ["DIAA", "IAA"]
    df_long = pd.melt(
        df,
        id_vars=["Step", "Threshold"],
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
        col="Threshold",  # This creates the columns of subplots
        kind="line",
        palette="Set2",
        height=5,  # Height of each facet in inches
        col_wrap=3,
        aspect=1.2,  # Aspect ratio of each facet
        linewidth=1.5,
    )

    # 3. Add the shaded min/max regions to each subplot (facet).
    #    We need to iterate through the axes of the FacetGrid.
    for env_name, ax in g.axes_dict.items():
        # Filter the original wide-format DataFrame for the specific environment
        df_env = df[df["Threshold"] == env_name]

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
    g.set_titles(col_template="Initial Threshold: {col_name}")
    g._legend.remove()  # noqa: SLF001

    # 4. Set overall title and save the figure
    g.savefig("ablation-lc.pdf", format="pdf")
    plt.show()


# Generate the plot
sns.set_theme(context="paper", font_scale=2.5, font="Times New Roman")
generate_learningcurve_facets(combined_df)

# %%

df_threshold = pd.concat([df_threshold_25, df_threshold_75], ignore_index=True)
algorithms = ["DIAA", "IAA"]
df_long = pd.melt(
    df_threshold,
    id_vars=["Step", "Threshold"],
    value_vars=algorithms,
    var_name="Algorithm",
    value_name="Threshold Value",
)

# %%
# 2. Create the faceted plot using sns.relplot.
#    This function returns a FacetGrid object.
g = sns.relplot(
    data=df_long,
    x="Step",
    y="Threshold Value",
    hue="Algorithm",
    col="Threshold",  # This creates the columns of subplots
    kind="line",
    palette="Set2",
    height=5,  # Height of each facet in inches
    col_wrap=3,
    aspect=1.2,  # Aspect ratio of each facet
    linewidth=1.5,
)
g.set_titles(col_template="Initial Threshold: {col_name}")
sns.move_legend(
    g,
    "lower center",
    bbox_to_anchor=(0.35, -0.10),  # Position it horizontally centered, just above the plots
    ncols=len(algorithms),  # Display all items in a single row
    title=None,  # Remove the legend title
    frameon=False,  # Remove the legend box frame
)
g.savefig("ablation-threshold.pdf", format="pdf")
plt.show()

# %%
