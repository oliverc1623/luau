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
df_bipedal = pd.read_csv("bipedal-thresholds.csv")
df_bipedal = df_bipedal.rename(
    columns={
        "Group: sac-iaa - introspection_threshold": "IAA",
        "Group: sac-diaa - introspection_threshold": "DIAA",
    },
)
df_bipedal["Environment"] = "Bipedal Walker: \nHardcore Mode"

# %%

df_lunar = pd.read_csv("lunar-thresholds.csv")
df_lunar = df_lunar.rename(
    columns={
        "Group: sac-diaa-lunarlander-v3 - introspection_threshold": "DIAA",
        "Group: sac-iaa-lunarlander-v3 - introspection_threshold": "IAA",
    },
)
df_lunar["Environment"] = "Lunar Lander: \nWind Enabled"

# %%

df_curveroad = pd.read_csv("curvemerge-thresholds.csv")
df_curveroad = df_curveroad.dropna()
df_curveroad = df_curveroad.rename(
    columns={
        "Group: iaa - introspection_threshold": "IAA",
        "Group: diaa - introspection_threshold": "DIAA",
    },
)
df_curveroad["Environment"] = "Curve Road, \nDense Traffic"

# %%

df_tintersection = pd.read_csv("tintersection-thresholds.csv")
df_tintersection = df_tintersection.dropna()
df_tintersection = df_tintersection.rename(
    columns={
        "Group: iaa - introspection_threshold": "IAA",
        "Group: diaa - introspection_threshold": "DIAA",
    },
)
df_tintersection["Environment"] = "T Intersection, \nDense Traffic"

# %%

df_merge_turn = pd.read_csv("merge-turn-thresholds.csv")
df_merge_turn = df_merge_turn.dropna()
df_merge_turn = df_merge_turn.rename(
    columns={
        "Group: iaa - introspection_threshold": "IAA",
        "Group: diaa - introspection_threshold": "DIAA",
    },
)
df_merge_turn["Environment"] = "Merge Turn, \nDense Traffic"

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
g = sns.relplot(
    data=df_long,
    x="Step",
    y="Threshold Value",
    hue="Algorithm",
    col="Environment",  # This creates the columns of subplots
    kind="line",
    height=3,  # Height of each facet in inches
    aspect=1.0,  # Aspect ratio of each facet
    facet_kws={"sharey": False},
    col_wrap=3,
)

g.set_titles(col_template="{col_name}")
g.savefig("threshold-facet.pdf", format="pdf")
sns.move_legend(g, "lower right")
plt.tight_layout()
plt.show()

# %%
