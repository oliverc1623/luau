# %%
import pandas as pd
import seaborn as sns


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
g.set_axis_labels("Action Value")

# %%
