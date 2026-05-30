# %%

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


pd.set_option("display.max_columns", None)
sns.set_theme(style="whitegrid", palette="pastel")

# %%
df_bipedal = pd.read_csv("data/bipedal_round3.csv")
df_bipedal = df_bipedal.rename(
    columns={
        "Group: tgrl - episode_return": "TGRL",
        "Group: tgrl - episode_return__MIN": "TGRL_Min",
        "Group: tgrl - episode_return__MAX": "TGRL_Max",
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

df_lunar = pd.read_csv("data/lunar_round3.csv")
df_lunar = df_lunar.rename(
    columns={
        "Group: tgrl - episode_return": "TGRL",
        "Group: tgrl - episode_return__MIN": "TGRL_Min",
        "Group: tgrl - episode_return__MAX": "TGRL_Max",
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

df_curveroad = pd.read_csv("data/curvemerge_round2.csv")
df_curveroad = df_curveroad.dropna()
df_curveroad = df_curveroad.rename(
    columns={
        "Group: tgrl - episode_return": "TGRL",
        "Group: tgrl - episode_return__MIN": "TGRL_Min",
        "Group: tgrl - episode_return__MAX": "TGRL_Max",
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

df_tintersection = pd.read_csv("data/t_int_round2.csv")
df_tintersection = df_tintersection.dropna()
df_tintersection = df_tintersection.rename(
    columns={
        "Group: tgrl - episode_return": "TGRL",
        "Group: tgrl - episode_return__MIN": "TGRL_Min",
        "Group: tgrl - episode_return__MAX": "TGRL_Max",
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

df_merge_turn = pd.read_csv("data/merge_turn_round2.csv")
df_merge_turn = df_merge_turn.dropna()
df_merge_turn = df_merge_turn.rename(
    columns={
        "Group: tgrl - episode_return": "TGRL",
        "Group: tgrl - episode_return__MIN": "TGRL_Min",
        "Group: tgrl - episode_return__MAX": "TGRL_Max",
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
algorithms = ["Baseline", "Finetune", "IAA", "DIAA", "TGRL"]

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

algorithms = ["Baseline", "Finetune", "IAA", "DIAA", "TGRL"]
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


# Fixed color per algorithm so the same method has the same color across both figures.
_SET2 = sns.color_palette("Set2", 5)
ALGO_COLORS = {
    "Baseline": _SET2[0],
    "DIAA": _SET2[1],
    "IAA": _SET2[2],
    "Finetune": _SET2[3],
    "TGRL": _SET2[4],
}


def generate_learningcurve_facets(
    df: pd.DataFrame,
    environments: list[str],
    algorithms: list[str],
    filename: str,
    col_wrap: int,
    aspect: float = 1.2,
    xlim_overrides: dict[str, tuple[float, float]] | None = None,
) -> None:
    """
    Generate a faceted learning curve plot for the given environments, with independent y-axes.

    ``xlim_overrides`` maps an environment name to an ``(xmin, xmax)`` tuple to truncate that
    facet's x-axis (e.g., to highlight the jumpstart). Other facets keep their full range.
    """
    xlim_overrides = xlim_overrides or {}
    df = df[df["Environment"].isin(environments)]
    df_long = pd.melt(
        df,
        id_vars=["Step", "Environment"],
        value_vars=algorithms,
        var_name="Algorithm",
        value_name="Episodic Returns",
    )

    g = sns.relplot(
        data=df_long,
        x="Step",
        y="Episodic Returns",
        hue="Algorithm",
        hue_order=algorithms,
        col="Environment",
        col_order=environments,
        kind="line",
        palette=ALGO_COLORS,
        height=10,
        aspect=aspect,
        col_wrap=col_wrap,
        linewidth=2,
        facet_kws={"sharey": False, "sharex": False},  # independent x and y axes per facet
    )

    # Add shaded min/max regions to each subplot
    for env_name, ax in g.axes_dict.items():
        df_env = df[df["Environment"] == env_name]
        for algo in algorithms:
            ax.fill_between(
                df_env["Step"],
                df_env[f"{algo}"] - df_env[f"{algo}_SE"],
                df_env[f"{algo}"] + df_env[f"{algo}_SE"],
                color=ALGO_COLORS[algo],
                alpha=0.2,
            )
        if env_name in xlim_overrides:
            ax.set_xlim(*xlim_overrides[env_name])
    g.set_titles(col_template="{col_name}")

    # Move legend below the plots, spread horizontally
    sns.move_legend(
        g,
        "lower center",
        bbox_to_anchor=(0.5, -0.05),
        ncols=len(algorithms),
        title=None,
        frameon=False,
    )

    g.savefig(filename, format="pdf", bbox_inches="tight")
    plt.show()


# Generate the plots
sns.set_theme(context="paper", font_scale=6, font="Times New Roman")

# All 5 experiments in a single row (all now include TGRL)
generate_learningcurve_facets(
    combined_df,
    environments=["LLWE", "BWHM", "T-Int.", "CM", "MT"],
    algorithms=["Baseline", "DIAA", "IAA", "Finetune", "TGRL"],
    filename="learning-curves-facet.pdf",
    col_wrap=5,
)

# %%
