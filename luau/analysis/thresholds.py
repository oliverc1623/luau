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
df_bipedal = pd.read_csv("data/bipedal-thresholds.csv")
df_bipedal = df_bipedal.rename(
    columns={
        "Group: sac-iaa - introspection_threshold": "IAA",
        "Group: sac-iaa - introspection_threshold__MIN": "IAA_Min",
        "Group: sac-iaa - introspection_threshold__MAX": "IAA_Max",
        "Group: sac-diaa - introspection_threshold": "DIAA",
        "Group: sac-diaa - introspection_threshold__MIN": "DIAA_Min",
        "Group: sac-diaa - introspection_threshold__MAX": "DIAA_Max",
    },
)
df_bipedal["Environment"] = "BWHM"

# %%

df_lunar = pd.read_csv("data/lunar-thresholds.csv")
df_lunar = df_lunar.rename(
    columns={
        "Group: sac-diaa-lunarlander-v3 - introspection_threshold": "DIAA",
        "Group: sac-diaa-lunarlander-v3 - introspection_threshold__MIN": "DIAA_Min",
        "Group: sac-diaa-lunarlander-v3 - introspection_threshold__MAX": "DIAA_Max",
        "Group: sac-iaa-lunarlander-v3 - introspection_threshold": "IAA",
        "Group: sac-iaa-lunarlander-v3 - introspection_threshold__MIN": "IAA_Min",
        "Group: sac-iaa-lunarlander-v3 - introspection_threshold__MAX": "IAA_Max",
    },
)
df_lunar["Environment"] = "LLWE"

# %%

df_curveroad = pd.read_csv("data/curvemerge-thresholds.csv")
df_curveroad = df_curveroad.dropna()
df_curveroad = df_curveroad.rename(
    columns={
        "Group: iaa - introspection_threshold": "IAA",
        "Group: iaa - introspection_threshold__MIN": "IAA_Min",
        "Group: iaa - introspection_threshold__MAX": "IAA_Max",
        "Group: diaa - introspection_threshold": "DIAA",
        "Group: diaa - introspection_threshold__MIN": "DIAA_Min",
        "Group: diaa - introspection_threshold__MAX": "DIAA_Max",
    },
)
df_curveroad["Environment"] = "CM"

# %%

df_tintersection = pd.read_csv("data/tintersection-thresholds.csv")
df_tintersection = df_tintersection.dropna()
df_tintersection = df_tintersection.rename(
    columns={
        "Group: iaa - introspection_threshold": "IAA",
        "Group: iaa - introspection_threshold__MIN": "IAA_Min",
        "Group: iaa - introspection_threshold__MAX": "IAA_Max",
        "Group: diaa - introspection_threshold": "DIAA",
        "Group: diaa - introspection_threshold__MIN": "DIAA_Min",
        "Group: diaa - introspection_threshold__MAX": "DIAA_Max",
    },
)
df_tintersection["Environment"] = "T-Int."

# %%

df_merge_turn = pd.read_csv("data/merge-turn-thresholds.csv")
df_merge_turn = df_merge_turn.dropna()
df_merge_turn = df_merge_turn.rename(
    columns={
        "Group: iaa - introspection_threshold": "IAA",
        "Group: iaa - introspection_threshold__MIN": "IAA_Min",
        "Group: iaa - introspection_threshold__MAX": "IAA_Max",
        "Group: diaa - introspection_threshold": "DIAA",
        "Group: diaa - introspection_threshold__MIN": "DIAA_Min",
        "Group: diaa - introspection_threshold__MAX": "DIAA_Max",
    },
)
df_merge_turn["Environment"] = "MT"

# %%

combined_df = pd.concat(
    [
        df_bipedal,
        df_lunar,
        df_curveroad,
        df_tintersection,
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

# Estimate the standard error from the min/max range for each algorithm.
for algo in ["IAA", "DIAA"]:
    if f"{algo}_Max" in combined_df.columns and f"{algo}_Min" in combined_df.columns:
        combined_df[f"{algo}_SE"] = (combined_df[f"{algo}_Max"] - combined_df[f"{algo}_Min"]) / 4


# %%

"""Generate a faceted learning curve plot, separated by environment."""
# 1. Define algorithms and melt the DataFrame to a "tidy" format.
#    'Environment' is now an identifier variable.
algorithms = ["DIAA", "IAA"]
df_long = pd.melt(
    combined_df,
    id_vars=["Step", "Environment"],
    value_vars=algorithms,
    var_name="Algorithm",
    value_name="Threshold Value",
)

# %%

# 2. Create the faceted plot using sns.relplot.
#    This function returns a FacetGrid object.
palette = sns.color_palette("Set2", len(algorithms))
color_map = dict(zip(algorithms, palette, strict=False))

g = sns.relplot(
    data=df_long,
    x="Step",
    y="Threshold Value",
    hue="Algorithm",
    hue_order=algorithms,
    palette=color_map,
    col="Environment",  # This creates the columns of subplots
    kind="line",
    height=3,  # Height of each facet in inches
    aspect=1.0,  # Aspect ratio of each facet
    facet_kws={"sharey": False},
)

# Shade +/- SE around each algorithm's threshold curve, per facet.
for env_name, ax in g.axes_dict.items():
    df_env = combined_df[combined_df["Environment"] == env_name]
    for algo in algorithms:
        if f"{algo}_SE" in df_env.columns:
            ax.fill_between(
                df_env["Step"],
                df_env[algo] - df_env[f"{algo}_SE"],
                df_env[algo] + df_env[f"{algo}_SE"],
                color=color_map[algo],
                alpha=0.2,
            )

sns.move_legend(g, "upper center", bbox_to_anchor=(0.45, 0.1), ncol=2)

g.set_titles(col_template="{col_name}")
g.savefig("threshold-facet.pdf", format="pdf")
plt.show()

# %%
