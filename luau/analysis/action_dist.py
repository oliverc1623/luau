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

num_driving_actions = 2

# %%

tdf = pd.read_csv("../driving-envs/TIntersection-v0_rfcac5sk_actions.csv")
tdf["env_id"] = "TIntersection-v0"
tdf = tdf[tdf["Action Dimension"] == "steering"]
cydf = pd.read_csv("../driving-envs/CyCurveMerge-v0_wjbish1a_actions.csv")
cydf["env_id"] = "CyCurveMerge-v0"
cydf = cydf[cydf["Action Dimension"] == "throttle"]

df_combined = pd.concat([tdf, cydf])

# %%

g = sns.displot(
    data=df_combined,
    x="Value",
    hue="Source",
    col="env_id",
    kind="kde",
    fill=True,
    alpha=0.4,
    palette={"Teacher": "blue", "DIAA Student": "red"},
    height=4,
    aspect=1.2,
    col_wrap=min(num_driving_actions, 3),
)
g.axes[0].set_xlabel("Steering Actions")
g.axes[1].set_xlabel("Throttle Actions")

plt.show()

# %%
