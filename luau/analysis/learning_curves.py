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

# %%
df_bipedal = pd.read_csv("bipedal-learning-curves.csv")
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
df_bipedal["Environment"] = "Bipedal Walker: Hardcore Mode"
df_bipedal.head(10)

# %%

df_lunar = pd.read_csv("lunar-learning-curves.csv")
df_lunar = df_lunar.rename(
    columns={
        "Group: sac-baseline-lunarlander-v3 - episode_return": "Baseline",
        "Group: sac-baseline-lunarlander-v3 - episode_return__MIN": "Baseline_Min",
        "Group: sac-baseline-lunarlander-v3 - episode_return__MAX": "Baseline_Max",
        "Group: sac-finetune-lunarlander-v3 - episode_return": "Finetune",
        "Group: sac-finetune-lunarlander-v3 - episode_return__MIN": "Finetune_Min",
        "Group: sac-finetune-lunarlander-v3 - episode_return__MAX": "Finetune_Max",
        "Group: sac-iaa-lunarlander-v3 - episode_return": "IAA",
        "Group: sac-iaa-lunarlander-v3 - episode_return__MIN": "IAA_Min",
        "Group: sac-iaa-lunarlander-v3 - episode_return__MAX": "IAA_Max",
        "Group: sac-diaa-lunarlander-v3 - episode_return": "DIAA",
        "Group: sac-diaa-lunarlander-v3 - episode_return__MIN": "DIAA_Min",
        "Group: sac-diaa-lunarlander-v3 - episode_return__MAX": "DIAA_Max",
    },
)
df_lunar["Environment"] = "Lunar Lander: Wind Enabled"
df_lunar.head(10)

# %%

combined_df = pd.concat([df_bipedal, df_lunar], ignore_index=True)

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
        height=5,  # Height of each facet in inches
        aspect=1.0,  # Aspect ratio of each facet
        facet_kws={"sharey": False},
    )

    # 3. Add the shaded min/max regions to each subplot (facet).
    #    We need to iterate through the axes of the FacetGrid.
    for env_name, ax in g.axes_dict.items():
        # Filter the original wide-format DataFrame for the specific environment
        df_env = df[df["Environment"] == env_name]

        # Get the color mapping from the plot's legend
        handles, labels = ax.get_legend_handles_labels()
        color_map = {label: handle.get_color() for label, handle in zip(labels, handles, strict=False)}

        # Add a shaded region for each algorithm
        for algo in algorithms:
            if algo in color_map:  # Ensure algorithm is plotted
                color = color_map[algo]
                ax.fill_between(
                    df_env["Step"],
                    df_env[f"{algo}_Min"],
                    df_env[f"{algo}_Max"],
                    color=color,
                    alpha=0.2,
                )
    sns.move_legend(
        g,
        "lower center",
        bbox_to_anchor=(0.5, 1),  # Position it horizontally centered, just above the plots
        ncols=len(algorithms),  # Display all items in a single row
        title=None,  # Remove the legend title
        frameon=False,  # Remove the legend box frame
    )

    # 4. Set overall title and save the figure
    g.tight_layout()
    g.savefig("bipedal_walker_facets.pdf", format="pdf")
    plt.show()


# Generate the plot
generate_learningcurve_facets(combined_df)

# %%
