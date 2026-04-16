# %%
"""
Combined burn-in + decay ablation: 1x4 figure of advice rate.

Cols 0-1: burn-in ablation (LunarLander, BipedalWalker)
Cols 2-3: decay ablation   (LunarLander, BipedalWalker)
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import wandb


WANDB_PROJECT = "luau"
BURN_IN_EXP = "burnin_ablation"
DECAY_EXP = "decay_ablation"
ENV_IDS = ["LunarLander-v3", "BipedalWalker-v3"]
METRICS = ["advice", "episode_return"]
SMOOTH_WINDOW = 10
BURN_IN_XLIM = (0, 400_000)
OUT_PDF = "burn-in-decay-ablation-advice.pdf"
OUT_PDF_RETURNS = "burn-in-decay-ablation-returns.pdf"

# %%


def fetch_runs(exp_name: str, hyperparam_key: str) -> pd.DataFrame:
    """Pull W&B runs for an ablation experiment and return a smoothed dataframe."""
    api = wandb.Api()
    runs = list(
        api.runs(
            WANDB_PROJECT,
            filters={"config.exp_name": exp_name, "config.env_id": {"$in": ENV_IDS}},
        ),
    )
    print(f"Found {len(runs)} runs for exp_name={exp_name}")

    records = []
    for run in runs:
        cfg = run.config
        env_id = cfg.get("env_id")
        value = cfg.get(hyperparam_key)
        if env_id is None or value is None:
            continue
        history = run.history(keys=METRICS, samples=500, pandas=True)
        if history.empty:
            continue
        history["env_id"] = env_id
        history[hyperparam_key] = value
        history["run_id"] = run.id
        records.append(history)
    print(f"Collected history from {len(records)} runs for {exp_name}")

    df = pd.concat(records, ignore_index=True)
    df = df.rename(columns={"_step": "Step", "advice": "Advice", "episode_return": "Episodic Returns"})

    df = df.sort_values(["env_id", hyperparam_key, "run_id", "Step"]).copy()
    for col in ["Advice", "Episodic Returns"]:
        df[col] = df.groupby(["env_id", hyperparam_key, "run_id"])[col].transform(
            lambda x: x.rolling(window=SMOOTH_WINDOW, min_periods=1).mean(),
        )
    df[hyperparam_key] = df[hyperparam_key].astype(str)
    return df


burn_in_df = fetch_runs(BURN_IN_EXP, "burn_in")
decay_df = fetch_runs(DECAY_EXP, "introspection_decay")

# %%

sns.set_theme(context="paper", font_scale=1.5, font="Times New Roman", style="darkgrid")

fig, axes = plt.subplots(1, 4, figsize=(18, 4.2), sharex=False, sharey=False)

# (dataframe, hue column, legend title, xlim, env, ax index)
panel_specs = [
    (burn_in_df, "burn_in", "Burn-in", BURN_IN_XLIM, "LunarLander-v3", 0),
    (burn_in_df, "burn_in", "Burn-in", BURN_IN_XLIM, "BipedalWalker-v3", 1),
    (decay_df, "introspection_decay", "Decay", None, "LunarLander-v3", 2),
    (decay_df, "introspection_decay", "Decay", None, "BipedalWalker-v3", 3),
]

for df, hue_key, legend_title, xlim, env_id, col_idx in panel_specs:
    ax = axes[col_idx]
    sub = df[df["env_id"] == env_id]
    sns.lineplot(
        data=sub,
        x="Step",
        y="Advice",
        hue=hue_key,
        palette="Set2",
        linewidth=2,
        errorbar="se",
        ax=ax,
    )
    if xlim is not None:
        ax.set_xlim(xlim)
    ax.set_title(env_id)
    ax.set_xlabel("")
    ax.set_ylabel("")
    # Keep the legend only on the rightmost panel of each hyperparameter pair.
    if col_idx in (1, 3):
        ax.legend(
            title=legend_title,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
        )
    else:
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()

# Group annotations above the two hyperparameter pairs.
fig.text(0.28, 1.02, "Burn-in ablation", ha="center", va="bottom", fontsize=24)
fig.text(0.72, 1.02, "Decay ablation", ha="center", va="bottom", fontsize=24)

fig.supxlabel("Step")
fig.supylabel("Advice")
fig.tight_layout()
fig.savefig(OUT_PDF, format="pdf", bbox_inches="tight")
plt.show()

# %%

# 2x2 returns figure: row 0 burn-in, row 1 decay; cols are envs.

fig2, axes2 = plt.subplots(2, 2, figsize=(12, 8), sharex=False, sharey=False)

row_specs_returns = [
    (burn_in_df, "burn_in", "Burn-in", None),
    (decay_df, "introspection_decay", "Decay", None),
]

for row_idx, (df, hue_key, legend_title, xlim) in enumerate(row_specs_returns):
    for col_idx, env_id in enumerate(ENV_IDS):
        ax = axes2[row_idx, col_idx]
        sub = df[df["env_id"] == env_id]
        sns.lineplot(
            data=sub,
            x="Step",
            y="Episodic Returns",
            hue=hue_key,
            palette="Set2",
            linewidth=2,
            errorbar="se",
            ax=ax,
        )
        if xlim is not None:
            ax.set_xlim(xlim)
        if row_idx == 0:
            ax.set_title(env_id)
        ax.set_xlabel("")
        ax.set_ylabel("")
        if col_idx == len(ENV_IDS) - 1:
            ax.legend(
                title=legend_title,
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                frameon=False,
            )
        else:
            legend = ax.get_legend()
            if legend is not None:
                legend.remove()

fig2.supxlabel("Step")
fig2.supylabel("Episodic Returns")
fig2.tight_layout()
fig2.savefig(OUT_PDF_RETURNS, format="pdf", bbox_inches="tight")
plt.show()

# %%
