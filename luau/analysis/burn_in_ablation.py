# %%
"""Burn-in ablation: pull runs from W&B, group by env_id and burn_in, plot facets."""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import wandb


WANDB_PROJECT = "luau"
EXP_NAME = "burnin_ablation"
ENV_IDS = ["LunarLander-v3", "BipedalWalker-v3"]
METRICS = ["advice", "episode_return"]
SMOOTH_WINDOW = 10

# %%

api = wandb.Api()
runs = list(
    api.runs(
        WANDB_PROJECT,
        filters={"config.exp_name": EXP_NAME, "config.env_id": {"$in": ENV_IDS}},
    ),
)
print(f"Found {len(runs)} runs for exp_name={EXP_NAME}, env_ids={ENV_IDS}")

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

# Smooth per (run_id) so seaborn can compute error bands across seeds.
agg = df.sort_values(["env_id", "burn_in", "run_id", "Step"]).copy()
for col in ["Advice", "Episodic Returns"]:
    agg[col] = agg.groupby(["env_id", "burn_in", "run_id"])[col].transform(
        lambda x: x.rolling(window=SMOOTH_WINDOW, min_periods=1).mean(),
    )

agg["burn_in"] = agg["burn_in"].astype(str)

# %%


def facet_plot(df: pd.DataFrame, value_col: str, ylabel: str, out_pdf: str, xlim: tuple | None = None) -> None:
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
        errorbar="se",
        facet_kws={"sharey": False, "sharex": False},
    )
    if xlim is not None:
        for ax in g.axes.flat:
            ax.set_xlim(xlim)
    g.set_titles(col_template="{col_name}")
    g.set_axis_labels("Step", ylabel)
    g.figure.subplots_adjust(bottom=0.25)
    sns.move_legend(
        g,
        "upper center",
        bbox_to_anchor=(0.35, 0.05),
        ncols=df["burn_in"].nunique(),
        title="Burn-in",
        frameon=False,
    )
    g.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.show()


sns.set_theme(context="paper", font_scale=2.2, font="Times New Roman", style="darkgrid")

facet_plot(agg, "Episodic Returns", "Episodic Returns", "burn-in-ablation-returns.pdf")
facet_plot(agg, "Advice", "Advice", "burn-in-ablation-advice.pdf", xlim=(0, 400_000))

# %%
