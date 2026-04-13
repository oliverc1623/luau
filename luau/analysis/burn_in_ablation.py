# %%
"""Burn-in ablation: pull runs from W&B, group by env_id and burn_in, plot facets."""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import wandb


WANDB_PROJECT = "luau"
EXP_NAME = "burnin_ablation"
ENV_ID = "LunarLander-v3"
METRICS = ["advice", "episode_return"]
SMOOTH_WINDOW = 10

# %%

api = wandb.Api()
runs = list(
    api.runs(
        WANDB_PROJECT,
        filters={"config.exp_name": EXP_NAME, "config.env_id": ENV_ID},
    ),
)
print(f"Found {len(runs)} runs for exp_name={EXP_NAME}, env_id={ENV_ID}")

records = []
for run in runs:
    cfg = run.config
    env_id = cfg.get("env_id")
    burn_in = cfg.get("burn_in")
    if env_id is None or burn_in is None:
        continue
    history = run.history(keys=METRICS, samples=500, pandas=True)
    if history.empty:
        continue
    history["env_id"] = env_id
    history["burn_in"] = burn_in
    history["run_id"] = run.id
    records.append(history)
print(f"Collected history from {len(records)} runs")

df = pd.concat(records, ignore_index=True)
df = df.rename(columns={"_step": "Step", "advice": "Advice", "episode_return": "Episodic Returns"})

# %%

# Average across seeds within (env_id, burn_in) at each step.
agg = df.groupby(["env_id", "burn_in", "Step"], as_index=False)[["Advice", "Episodic Returns"]].mean().sort_values(["env_id", "burn_in", "Step"])

# Smooth per group.
for col in ["Advice", "Episodic Returns"]:
    agg[col] = agg.groupby(["env_id", "burn_in"])[col].transform(
        lambda x: x.rolling(window=SMOOTH_WINDOW, min_periods=1).mean(),
    )

agg["burn_in"] = agg["burn_in"].astype(str)

# %%


def facet_plot(df: pd.DataFrame, value_col: str, ylabel: str, out_pdf: str) -> None:
    """Facet by env_id, color by burn_in."""
    g = sns.relplot(
        data=df,
        x="Step",
        y=value_col,
        hue="burn_in",
        col="env_id",
        kind="line",
        palette="Set2",
        height=4.2,
        aspect=1.5,
        col_wrap=3,
        linewidth=2,
        facet_kws={"sharey": False, "sharex": False},
    )
    g.set_titles(col_template="{col_name}")
    g.set_axis_labels("Step", ylabel)
    sns.move_legend(
        g,
        "lower center",
        bbox_to_anchor=(0.5, -0.05),
        ncols=df["burn_in"].nunique(),
        title="Burn-in",
        frameon=False,
    )
    g.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.show()


sns.set_theme(context="paper", font_scale=2.2, font="Times New Roman", style="darkgrid")

facet_plot(agg, "Episodic Returns", "Episodic Returns", "burn-in-ablation-returns.pdf")
facet_plot(agg, "Advice", "Advice", "burn-in-ablation-advice.pdf")

# %%
