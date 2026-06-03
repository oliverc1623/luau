# %%

from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import wandb


pd.set_option("display.max_columns", None)
sns.set_theme(style="whitegrid", palette="pastel")

# %% Load the abs_diff curves from WandB (project "luau-final").
#
# We group runs by their WandB ``Group`` and keep the BipedalWalker (BWHM) and
# LunarLander (LLWE) experiments. Only the DIAA and IAA groups log ``abs_diff``
# (the introspection signal |Q1 - Q2|), so the baselines drop out naturally.

WANDB_PROJECT = "luau-final"
METRIC = "abs_diff"
GRID_POINTS = 1000
SMOOTH_WINDOW = 10

ENV_LABELS = {"BipedalWalker-v3": "BWHM", "LunarLander-v3": "LLWE"}
ENVIRONMENTS = ["LLWE", "BWHM"]


def algo_from_group(group: str) -> str | None:
    """Normalize a WandB group name to a canonical algorithm label (DIAA only)."""
    if "diaa" in (group or "").lower():
        return "DIAA"
    return None


api = wandb.Api()
runs = api.runs(WANDB_PROJECT)

# Bucket each seed's abs_diff curve by (environment, algorithm).
seed_curves: dict[tuple[str, str], list[tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
for r in runs:
    env = ENV_LABELS.get(r.config.get("env_id"))
    algo = algo_from_group(r.group)
    if env is None or algo is None:
        continue
    hist = r.history(keys=[METRIC], samples=GRID_POINTS * 5)
    if METRIC not in hist.columns or hist.empty:
        continue
    steps = hist["_step"].to_numpy(float)
    vals = hist[METRIC].to_numpy(float)
    finite = np.isfinite(steps) & np.isfinite(vals)
    if finite.any():
        seed_curves[(env, algo)].append((steps[finite], vals[finite]))

algorithms = sorted({a for (_env, a) in seed_curves})
print(f"Fetched {len(runs)} runs; (env, algo) groups: {sorted(seed_curves)}")

# %%


def smooth(values: np.ndarray) -> np.ndarray:
    """Rolling-mean smoothing of a curve."""
    return pd.Series(values).rolling(window=SMOOTH_WINDOW, min_periods=1).mean().to_numpy()


# Build a wide dataframe (one column per algorithm for the mean, plus a `{algo}_SE`
# column), faceted by environment -- mirroring the layout of ``learning_curves.py``.
frames = []
for env in ENVIRONMENTS:
    all_curves = [c for a in algorithms for c in seed_curves.get((env, a), [])]
    grid = np.linspace(max(s[0] for s, _ in all_curves), min(s[-1] for s, _ in all_curves), GRID_POINTS)
    data = {"Step": grid, "Environment": env}
    for algo in algorithms:
        curves = seed_curves.get((env, algo), [])
        if not curves:
            continue
        stacked = np.array([smooth(np.interp(grid, s, v)) for s, v in curves])
        mean = stacked.mean(axis=0)
        std = stacked.std(axis=0, ddof=1) if stacked.shape[0] > 1 else np.zeros_like(mean)
        data[algo] = mean
        # Standard error across the four seeds.
        data[f"{algo}_SE"] = std / np.sqrt(stacked.shape[0])
    frames.append(pd.DataFrame(data))

combined_df = pd.concat(frames, ignore_index=True)

# %%

# Fixed color per algorithm, consistent with the other figures.
_SET2 = sns.color_palette("Set2", 5)
ALGO_COLORS = {"DIAA": _SET2[1], "IAA": _SET2[2]}


def generate_absdiff_facets(
    df: pd.DataFrame,
    environments: list[str],
    algos: list[str],
    filename: str,
    aspect: float = 1.3,
) -> None:
    """Generate a one-row faceted plot of abs_diff per environment, with a line per algorithm."""
    df_long = pd.melt(
        df,
        id_vars=["Step", "Environment"],
        value_vars=algos,
        var_name="Algorithm",
        value_name="Absolute Q-value Difference",
    )

    g = sns.relplot(
        data=df_long,
        x="Step",
        y="Absolute Q-value Difference",
        hue="Algorithm",
        hue_order=algos,
        col="Environment",
        col_order=environments,
        kind="line",
        palette=ALGO_COLORS,
        height=6,
        aspect=aspect,
        linewidth=2,
        facet_kws={"sharey": False, "sharex": False},  # independent x and y axes per facet
    )

    # Add shaded +/- SE regions to each subplot.
    for env_name, ax in g.axes_dict.items():
        df_env = df[df["Environment"] == env_name]
        for algo in algos:
            if algo in df_env.columns:
                ax.fill_between(
                    df_env["Step"],
                    df_env[algo] - df_env[f"{algo}_SE"],
                    df_env[algo] + df_env[f"{algo}_SE"],
                    color=ALGO_COLORS[algo],
                    alpha=0.2,
                )
    g.set_titles(col_template="{col_name}")

    # Move legend below the plots, spread horizontally.
    sns.move_legend(
        g,
        "lower center",
        bbox_to_anchor=(0.5, -0.1),
        ncols=len(algos),
        title=None,
        frameon=False,
    )

    g.savefig(filename, format="pdf", bbox_inches="tight")
    plt.show()


# Generate the plot
sns.set_theme(context="paper", font_scale=2.5, font="Times New Roman")

generate_absdiff_facets(
    combined_df,
    environments=ENVIRONMENTS,
    algos=algorithms,
    filename="abs-diff.pdf",
)

# %%
