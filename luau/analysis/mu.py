# %%

from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import wandb


pd.set_option("display.max_columns", None)
sns.set_theme(style="whitegrid", palette="pastel")

# %% Load the mu-ablation runs from WandB (project "luau", group "mu").

WANDB_PROJECT = "luau"
GROUP = "mu"
METRICS = ["introspection_threshold", "episode_return"]
METRIC_LABELS = {
    "introspection_threshold": "Introspection Threshold",
    "episode_return": "Episodic Returns",
}
GRID_POINTS = 1000
SMOOTH_WINDOW = 10

api = wandb.Api()
runs = api.runs(WANDB_PROJECT, filters={"group": GROUP})

# Bucket each seed's curve by (threshold_lr, metric).
seed_curves: dict[tuple[float, str], list[tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
for r in runs:
    threshold_lr = r.config.get("threshold_lr")
    hist = r.history(keys=METRICS, samples=GRID_POINTS * 5)
    if hist.empty:
        continue
    for metric in METRICS:
        if metric not in hist.columns:
            continue
        steps = hist["_step"].to_numpy(float)
        vals = hist[metric].to_numpy(float)
        finite = np.isfinite(steps) & np.isfinite(vals)
        if finite.any():
            seed_curves[(threshold_lr, metric)].append((steps[finite], vals[finite]))

threshold_lrs = sorted({tlr for (tlr, _metric) in seed_curves})
mus = [str(tlr) for tlr in threshold_lrs]
print(f"Fetched {len(runs)} runs; threshold_lr groups: {threshold_lrs}")

# %%


def smooth(values: np.ndarray) -> np.ndarray:
    """Rolling-mean smoothing of a curve."""
    return pd.Series(values).rolling(window=SMOOTH_WINDOW, min_periods=1).mean().to_numpy()


# Build a wide dataframe (one column per mu for the mean, plus a `{mu}_SE` column),
# faceted by metric -- mirroring the layout of ``learning_curves.py``.
frames = []
for metric in METRICS:
    all_curves = [c for tlr in threshold_lrs for c in seed_curves.get((tlr, metric), [])]
    grid = np.linspace(max(s[0] for s, _ in all_curves), min(s[-1] for s, _ in all_curves), GRID_POINTS)
    data = {"Step": grid, "Metric": METRIC_LABELS[metric]}
    for tlr in threshold_lrs:
        stacked = np.array([smooth(np.interp(grid, s, v)) for s, v in seed_curves[(tlr, metric)]])
        mean = stacked.mean(axis=0)
        std = stacked.std(axis=0, ddof=1) if stacked.shape[0] > 1 else np.zeros_like(mean)
        data[str(tlr)] = mean
        # Standard error across the four seeds.
        data[f"{tlr}_SE"] = std / np.sqrt(stacked.shape[0])
    frames.append(pd.DataFrame(data))

combined_df = pd.concat(frames, ignore_index=True)

# %%

# Fixed color per mu so the same threshold_lr keeps its color across facets.
MU_COLORS = dict(zip(mus, sns.color_palette("Set2", len(mus)), strict=False))


def generate_mu_facets(
    df: pd.DataFrame,
    metrics: list[str],
    mu_values: list[str],
    filename: str,
    aspect: float = 1.3,
) -> None:
    """Generate a one-row faceted plot of each metric, with a line per threshold_lr (mu)."""
    df_long = pd.melt(
        df,
        id_vars=["Step", "Metric"],
        value_vars=mu_values,
        var_name="mu",
        value_name="Value",
    )

    g = sns.relplot(
        data=df_long,
        x="Step",
        y="Value",
        hue="mu",
        hue_order=mu_values,
        col="Metric",
        col_order=metrics,
        kind="line",
        palette=MU_COLORS,
        height=6,
        aspect=aspect,
        linewidth=2,
        facet_kws={"sharey": False, "sharex": False},  # independent x and y axes per facet
    )

    # Add shaded +/- SE regions, and use the metric name as each facet's y-axis label.
    for metric_name, ax in g.axes_dict.items():
        df_env = df[df["Metric"] == metric_name]
        for mu in mu_values:
            ax.fill_between(
                df_env["Step"],
                df_env[mu] - df_env[f"{mu}_SE"],
                df_env[mu] + df_env[f"{mu}_SE"],
                color=MU_COLORS[mu],
                alpha=0.2,
            )
        ax.set_ylabel(metric_name)
    g.set_titles(col_template="")

    # Move legend below the plots, spread horizontally.
    sns.move_legend(
        g,
        "lower center",
        bbox_to_anchor=(0.5, -0.15),
        ncols=len(mu_values),
        title=r"Threshold LR ($\mu$)",
        frameon=False,
    )

    g.savefig(filename, format="pdf", bbox_inches="tight")
    plt.show()


# Generate the plot
sns.set_theme(context="paper", font_scale=2.5, font="Times New Roman")

generate_mu_facets(
    combined_df,
    metrics=[METRIC_LABELS[m] for m in METRICS],
    mu_values=mus,
    filename="mu-ablation.pdf",
)

# %%
