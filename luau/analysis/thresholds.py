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
df_bipedal = pd.read_csv("bipedal-thresholds.csv")
df_bipedal = df_bipedal.rename(
    columns={
        "Group: sac-iaa - introspection_threshold": "IAA",
        "Group: sac-diaa - introspection_threshold": "DIAA",
    },
)
df_bipedal["Environment"] = "Bipedal Walker: Hardcore Mode"

# %%

df_lunar = pd.read_csv("lunar-thresholds.csv")
df_lunar = df_lunar.rename(
    columns={
        "Group: sac-diaa-lunarlander-v3 - introspection_threshold": "DIAA",
        "Group: sac-iaa-lunarlander-v3 - introspection_threshold": "IAA",
    },
)
df_lunar["Environment"] = "Lunar Lander: Wind Enabled"

# %%

df_curveroad = pd.read_csv("curvemerge-thresholds.csv")
df_curveroad = df_curveroad.dropna()
df_curveroad = df_curveroad.rename(
    columns={
        "Group: iaa - introspection_threshold": "IAA",
        "Group: diaa - introspection_threshold": "DIAA",
    },
)
df_curveroad["Environment"] = "Curve Road, Dense Traffic"

# %%

df_tintersection = pd.read_csv("tintersection-thresholds.csv")
df_tintersection = df_tintersection.dropna()
df_tintersection = df_tintersection.rename(
    columns={
        "Group: iaa - introspection_threshold": "IAA",
        "Group: diaa - introspection_threshold": "DIAA",
    },
)
df_tintersection["Environment"] = "T Intersection, Dense Traffic"

# %%

combined_df = pd.concat(
    [
        df_bipedal,
        df_lunar,
        df_curveroad,
        df_tintersection,
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
    palette="Set2",
    height=3.5,  # Height of each facet in inches
    aspect=1.0,  # Aspect ratio of each facet
    facet_kws={"sharey": False},
    linewidth=1,
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
g.savefig("threshold-facet.pdf", format="pdf")
plt.show()

# %%
