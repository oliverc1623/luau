# %%
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


sns.set_theme(style="darkgrid")  # you can also pick 'darkgrid', 'white', etc.
# Set font family to Times New Roman
plt.rcParams["font.size"] = 52
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["font.family"] = "Times New Roman"
plt.figure(figsize=(2, 1))

num_driving_actions = 2

# %%

bdf = pd.read_csv("../BipedalWalker-v3_kswwn4xa_actions.csv")
bdf = bdf[bdf["Action Dimension"] == "Action Dim 3"]
bdf["env_id"] = "BipedalWalker-v3"

ldf = pd.read_csv("../LunarLander-v3_akn48jga_actions.csv")
ldf = ldf[ldf["Action Dimension"] == "Action Dim 2"]
ldf["env_id"] = "LunarLander-v3"

tdf = pd.read_csv("../driving-envs/TIntersection-v0_rfcac5sk_actions.csv")
tdf["env_id"] = "TIntersection-v0"
tdf = tdf[tdf["Action Dimension"] == "steering"]

cydf = pd.read_csv("../driving-envs/CyCurveMerge-v0_wjbish1a_actions.csv")
cydf["env_id"] = "CyCurveMerge-v0"
cydf = cydf[cydf["Action Dimension"] == "throttle"]

df_combined = pd.concat([tdf, cydf, bdf, ldf], ignore_index=True)

# %%
sns.set_theme(style="darkgrid")  # you can also pick 'darkgrid', 'white', etc.
# Set font family to Times New Roman
plt.rcParams["font.size"] = 52
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["font.family"] = "Times New Roman"
plt.figure(figsize=(2, 1))
sns.set_context("talk")

g = sns.displot(
    data=df_combined,
    x="Value",
    hue="Source",
    col="env_id",
    kind="kde",
    fill=True,
    alpha=0.4,
    palette={"Teacher": "#0A5FF5", "DIAA Student": "#f5a00a"},
    height=3,
    aspect=1.0,
    col_wrap=2,
)

g.axes[0].text(-1.0, 0.3, "Steering Angle", fontsize=14)
g.axes[1].text(-1.0, 0.3, "Driving Throttle", fontsize=14)
g.axes[2].text(-1.0, 0.3, "Hip Throttle", fontsize=14)
g.axes[3].text(-1.0, 0.3, "Lateral Throttle", fontsize=14)
g.set_axis_labels("Action Value")
g.set_titles(col_template="{col_name}")
g.savefig("action_dist.pdf", dpi=300, bbox_inches="tight")
plt.show()

# %%
