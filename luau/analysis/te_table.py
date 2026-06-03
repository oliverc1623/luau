# %%

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import wandb


sns.set_theme(style="darkgrid")  # you can also pick 'darkgrid', 'white', etc.
# Set font family to Times New Roman
plt.rcParams["font.size"] = 34
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["font.family"] = "Times New Roman"
plt.figure(figsize=(5, 4))
sns.set_context("talk")

# %%
df_bipedal = pd.read_csv("data/bipedal-learning-curves.csv")
df_bipedal = df_bipedal.rename(
    columns={
        "Group: sac-baseline - episode_return": "Baseline",
        "Group: sac-baseline - episode_return__MIN": "Baseline_Min",
        "Group: sac-baseline - episode_return__MAX": "Baseline_Max",
        "Group: sac-finetune - episode_return": "Finetune",
        "Group: sac-finetune - episode_return__MIN": "Finetune_Min",
        "Group: sac-finetune - episode_return__MAX": "Finetune_Max",
        "Group: sac-iaa - episode_return": "IAA",
        "Group: sac-iaa - episode_return__MIN": "IAA_Min",
        "Group: sac-iaa - episode_return__MAX": "IAA_Max",
        "Group: sac-diaa - episode_return": "DIAA",
        "Group: sac-diaa - episode_return__MIN": "DIAA_Min",
        "Group: sac-diaa - episode_return__MAX": "DIAA_Max",
    },
)
df_bipedal["Environment"] = "Bipedal Walker: Hardcore Mode"

# %%

df_lunar = pd.read_csv("data/lunar-round2.csv")
df_lunar = df_lunar.rename(
    columns={
        "Group: sac-baseline-lunarlander-v3 - episode_return": "Baseline",
        "Group: sac-baseline-lunarlander-v3 - episode_return__MIN": "Baseline_Min",
        "Group: sac-baseline-lunarlander-v3 - episode_return__MAX": "Baseline_Max",
        "Group: sac-diaa-lunarlander-v3 - episode_return": "DIAA",
        "Group: sac-diaa-lunarlander-v3 - episode_return__MIN": "DIAA_Min",
        "Group: sac-diaa-lunarlander-v3 - episode_return__MAX": "DIAA_Max",
        "Group: sac-iaa-lunarlander-v3 - episode_return": "IAA",
        "Group: sac-iaa-lunarlander-v3 - episode_return__MIN": "IAA_Min",
        "Group: sac-iaa-lunarlander-v3 - episode_return__MAX": "IAA_Max",
        "Group: sac-finetune-lunarlander-v3 - episode_return": "Finetune",
        "Group: sac-finetune-lunarlander-v3 - episode_return__MIN": "Finetune_Min",
        "Group: sac-finetune-lunarlander-v3 - episode_return__MAX": "Finetune_Max",
    },
)
df_lunar["Environment"] = "Lunar Lander: Wind Enabled"

# %%

df_curveroad = pd.read_csv("data/curve-merge-learning-curves.csv")
df_curveroad = df_curveroad.dropna()
df_curveroad = df_curveroad.rename(
    columns={
        "Group: sac-baseline - episode_return": "Baseline",
        "Group: sac-baseline - episode_return__MIN": "Baseline_Min",
        "Group: sac-baseline - episode_return__MAX": "Baseline_Max",
        "Group: finetune - episode_return": "Finetune",
        "Group: finetune - episode_return__MIN": "Finetune_Min",
        "Group: finetune - episode_return__MAX": "Finetune_Max",
        "Group: iaa - episode_return": "IAA",
        "Group: iaa - episode_return__MIN": "IAA_Min",
        "Group: iaa - episode_return__MAX": "IAA_Max",
        "Group: diaa - episode_return": "DIAA",
        "Group: diaa - episode_return__MIN": "DIAA_Min",
        "Group: diaa - episode_return__MAX": "DIAA_Max",
    },
)
df_curveroad["Environment"] = "Curve Road, Dense Traffic"

# %%

df_tintersection = pd.read_csv("data/t-inter.csv")
df_tintersection = df_tintersection.dropna()
df_tintersection = df_tintersection.rename(
    columns={
        "Group: sac-baseline - episode_return": "Baseline",
        "Group: sac-baseline - episode_return__MIN": "Baseline_Min",
        "Group: sac-baseline - episode_return__MAX": "Baseline_Max",
        "Group: finetune - episode_return": "Finetune",
        "Group: finetune - episode_return__MIN": "Finetune_Min",
        "Group: finetune - episode_return__MAX": "Finetune_Max",
        "Group: iaa - episode_return": "IAA",
        "Group: iaa - episode_return__MIN": "IAA_Min",
        "Group: iaa - episode_return__MAX": "IAA_Max",
        "Group: diaa - episode_return": "DIAA",
        "Group: diaa - episode_return__MIN": "DIAA_Min",
        "Group: diaa - episode_return__MAX": "DIAA_Max",
    },
)
df_tintersection["Environment"] = "T Intersection, Dense Traffic"

# %%

df_merge_turn = pd.read_csv("data/merge-turn.csv")
df_merge_turn = df_merge_turn.dropna()
df_merge_turn = df_merge_turn.rename(
    columns={
        "Group: sac-baseline - episode_return": "Baseline",
        "Group: sac-baseline - episode_return__MIN": "Baseline_Min",
        "Group: sac-baseline - episode_return__MAX": "Baseline_Max",
        "Group: finetune - episode_return": "Finetune",
        "Group: finetune - episode_return__MIN": "Finetune_Min",
        "Group: finetune - episode_return__MAX": "Finetune_Max",
        "Group: iaa - episode_return": "IAA",
        "Group: iaa - episode_return__MIN": "IAA_Min",
        "Group: iaa - episode_return__MAX": "IAA_Max",
        "Group: diaa - episode_return": "DIAA",
        "Group: diaa - episode_return__MIN": "DIAA_Min",
        "Group: diaa - episode_return__MAX": "DIAA_Max",
    },
)
df_merge_turn["Environment"] = "Merge Turn, Dense Traffic"

# %%

combined_df = pd.concat(
    [
        df_lunar,
        df_bipedal,
        df_tintersection,
        df_curveroad,
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

tmp_df = df_merge_turn  # skip first 7 points for lunar lander
steps = tmp_df["Step"].to_numpy()
transfer_rewards = tmp_df["DIAA"].to_numpy()
baseline_rewards = tmp_df["Baseline"].astype(float).to_numpy()

min_reward = min(np.min(transfer_rewards), np.min(baseline_rewards))
normalized_transfer = (transfer_rewards - min_reward).astype(float)
normalized_baseline = (baseline_rewards - min_reward).astype(float)

# 3. Compute AUC with the trapezoidal rule
auc_transfer = np.trapz(normalized_transfer, steps)
auc_baseline = np.trapz(normalized_baseline, steps)

# 4. Compute transfer efficacy
transfer_efficacy = (auc_transfer - auc_baseline) / auc_baseline

print("AUC (transfer method):", auc_transfer)
print("AUC (baseline):       ", auc_baseline)
print(f"Transfer Efficacy:    {transfer_efficacy:.2f}")

# jumpstart percentage increase
jumpstart = ((normalized_transfer[0] - normalized_baseline[0]) / normalized_baseline[0]) * 100
print(f"Jumpstart (% increase): {jumpstart:.2f}%")

# final reward percentage increase
final_reward_increase = ((normalized_transfer[-1] - normalized_baseline[-1]) / normalized_baseline[-1]) * 100
print(f"Final Reward (% increase): {final_reward_increase:.2f}%")

# %%
(normalized_transfer[0] - normalized_baseline[0]) / normalized_baseline[0] * 100

# %% Per-seed TE / jumpstart / final-reward from WandB (project: luau-final)
#
# Pulls the raw per-seed ``episode_return`` curves directly from WandB so we can
# report a genuine across-seed std (instead of estimating it from MIN/MAX bands).
#
#   * Box2D (LLWE, BWHM): runs are grouped by ``Group`` and filtered by ``env_id``.
#   * MetaDrive (T-Int., CM, MT): runs are grouped by ``Group`` and filtered by ``map``.
#
# For each transfer method we compute the metric per seed against the *mean*
# baseline curve, then report mean +/- std across that method's seeds.

WANDB_PROJECT = "luau-final"
METRIC = "episode_return"
GRID_POINTS = 1200
SMOOTH_WINDOW = 10

# Output ordering for the table.
ENV_ORDER = ["LLWE", "BWHM", "T-Int.", "CM", "MT"]
ENV_HEADERS = {
    "LLWE": "LLWE",
    "BWHM": "BWHM",
    "T-Int.": "T-Inters.",
    "CM": "Curve Merge",
    "MT": "Merge Turn",
}
METHOD_ORDER = ["IAA", "DIAA", "Finetune", "TGRL"]


def algo_from_group(group: str) -> str | None:
    """Normalize a WandB group name to a canonical algorithm label."""
    g = (group or "").lower()
    if "baseline" in g:
        return "Baseline"
    if "finetune" in g:
        return "Finetune"
    if "diaa" in g:  # check before "iaa" since "iaa" is a substring of "diaa"
        return "DIAA"
    if "iaa" in g:
        return "IAA"
    if "tgrl" in g:
        return "TGRL"
    return None


def env_from_run(env_id: str | None, env_map: str | None) -> str | None:
    """Assign a run to one of the five target tasks (env_id for Box2D, map for MetaDrive)."""
    if env_id == "LunarLander-v3":
        return "LLWE"
    if env_id == "BipedalWalker-v3":
        return "BWHM"
    if env_map == "T":
        return "T-Int."
    if env_map == "Cy":
        return "CM"
    if env_map == "yT":
        return "MT"
    return None


def smooth(values: np.ndarray) -> np.ndarray:
    """Rolling-mean smoothing, matching the learning-curve figures."""
    return pd.Series(values).rolling(window=SMOOTH_WINDOW, min_periods=1).mean().to_numpy()


# Collect per-(env, algo) lists of (steps, returns) seed curves.
api = wandb.Api()
seed_curves: dict[tuple[str, str], list[tuple[np.ndarray, np.ndarray]]] = defaultdict(list)

runs = api.runs(WANDB_PROJECT)
print(f"Fetching {len(runs)} runs from {WANDB_PROJECT}...")
for r in runs:
    algo = algo_from_group(r.group)
    env = env_from_run(r.config.get("env_id"), r.config.get("map"))
    if algo is None or env is None:
        continue
    hist = r.history(keys=[METRIC], samples=GRID_POINTS * 5)
    if hist.empty:
        continue
    steps = hist["_step"].to_numpy(float)
    vals = hist[METRIC].to_numpy(float)
    # Drop steps with NaN returns (e.g. logged before learning starts).
    finite = np.isfinite(steps) & np.isfinite(vals)
    if not finite.any():
        continue
    seed_curves[(env, algo)].append((steps[finite], vals[finite]))

for (env, algo), curves in sorted(seed_curves.items()):
    print(f"  {env:8s} {algo:9s}: {len(curves)} seeds")


# %% Compute metrics with across-seed std.


def metric_auc(steps: np.ndarray, vals: np.ndarray) -> float:
    """Area under the (min-shifted) return curve via the trapezoidal rule."""
    return float(np.trapz(vals, steps))


def mean_std(values: list[float]) -> tuple[float, float]:
    """Return (mean, sample std) of a list, with std=0 for a single sample."""
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), (float(arr.std(ddof=1)) if arr.size > 1 else 0.0)


results: dict[tuple[str, str], dict[str, tuple[float, float]]] = {}

for env in ENV_ORDER:
    base_curves = seed_curves.get((env, "Baseline"), [])
    if not base_curves:
        print(f"[warn] no baseline runs for {env}; skipping")
        continue

    # Common step grid for this environment (overlap across every seed curve here).
    env_curves = [c for (e, _algo), cs in seed_curves.items() if e == env for c in cs]
    grid_start = max(s[0] for s, _ in env_curves)
    grid_end = min(s[-1] for s, _ in env_curves)
    grid = np.linspace(grid_start, grid_end, GRID_POINTS)

    base_interp = np.array([smooth(np.interp(grid, s, v)) for s, v in base_curves])
    base_mean = base_interp.mean(axis=0)

    for method in METHOD_ORDER:
        method_curves = seed_curves.get((env, method), [])
        if not method_curves:
            continue
        te_vals, js_vals, fr_vals = [], [], []
        for s, v in method_curves:
            transfer = smooth(np.interp(grid, s, v))
            # TE: ratio of areas under the min-shifted curves (standard definition).
            min_reward = min(transfer.min(), base_mean.min())
            auc_b = metric_auc(grid, base_mean - min_reward)
            te_vals.append((metric_auc(grid, transfer - min_reward) - auc_b) / auc_b)
            # Jumpstart / final reward: percentage change vs. baseline at the first/last
            # step, on the raw curves (abs denominator keeps "improvement" positive and
            # avoids the singularity when the min-shifted baseline starts at ~0, e.g. LLWE).
            js_vals.append((transfer[0] - base_mean[0]) / abs(base_mean[0]) * 100)
            fr_vals.append((transfer[-1] - base_mean[-1]) / abs(base_mean[-1]) * 100)

        results[(env, method)] = {
            "TE": mean_std(te_vals),
            "Jumpstart": mean_std(js_vals),
            "Final Reward": mean_std(fr_vals),
        }
        te = results[(env, method)]["TE"]
        print(f"{env:8s} {method:9s} TE={te[0]:+.2f}+/-{te[1]:.2f}  (n={len(method_curves)})")


# %% Write the LaTeX table (mean +/- std) to a .tex file.

METRIC_ROWS = ["TE", "Jumpstart", "Final Reward"]
# Display labels (abbreviations defined in the caption).
METHOD_LABELS = {"IAA": "IAA", "DIAA": "DIAA", "Finetune": "FT", "TGRL": "TGRL"}
METRIC_LABELS = {"TE": "TE", "Jumpstart": "JS", "Final Reward": "FR"}


def diaa_is_best(env: str, metric: str) -> bool:
    """Return whether DIAA has the highest mean for this (env, metric) across all methods."""
    means = {m: results[(env, m)][metric][0] for m in METHOD_ORDER if (env, m) in results and metric in results[(env, m)]}
    return bool(means) and max(means, key=means.get) == "DIAA"


def fmt_cell(env: str, method: str, metric: str, *, with_std: bool) -> str:
    """Format a LaTeX math cell, bolding DIAA's value when it is best across methods."""
    res = results.get((env, method))
    if res is None or metric not in res:
        return "--"
    mean, std = res[metric]
    suffix = "" if metric == "TE" else r"\%"
    digits = 2 if metric == "TE" else 1
    mean_str = f"{mean:.{digits}f}"
    if method == "DIAA" and diaa_is_best(env, metric):
        mean_str = rf"\mathbf{{{mean_str}}}"
    if with_std:
        # std as a small subscript keeps the table within the text width.
        return rf"${mean_str}_{{\pm {std:.{digits}f}{suffix}}}$"
    return f"${mean_str}{suffix}$"


def build_table(*, with_std: bool, label: str, caption: str) -> str:
    """Build a full LaTeX ``table*`` of the metrics, with or without the std term."""
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        # Subscript std is narrower but still benefits from tighter column padding.
        *([r"\setlength{\tabcolsep}{4pt}"] if with_std else []),
        r"\begin{tabular}{ll" + "l" * len(ENV_ORDER) + "}",
        r"\hline",
        r"\textbf{Method} & \textbf{Metric} & " + " & ".join(rf"\textbf{{{ENV_HEADERS[e]}}}" for e in ENV_ORDER) + r" \\ \hline",
    ]
    for method in METHOD_ORDER:
        for i, metric in enumerate(METRIC_ROWS):
            row_label = rf"\multirow{{{len(METRIC_ROWS)}}}{{*}}{{{METHOD_LABELS[method]}}}" if i == 0 else ""
            cells = " & ".join(fmt_cell(e, method, metric, with_std=with_std) for e in ENV_ORDER)
            lines.append(f"{row_label} & {METRIC_LABELS[metric]} & {cells} \\\\")
        lines.append(r"\hline")
    lines += [
        r"\end{tabular}",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\end{table*}",
    ]
    return "\n".join(lines) + "\n"


_DEFS = (
    r"Transfer efficacy (TE), jumpstart (JS), and final reward (FR) for IAA, DIAA, "
    r"Fine-tune (FT), and TGRL across the five target tasks. All metrics are computed "
    r"relative to the SAC baseline. \textbf{Bold} entries mark metrics where DIAA achieves "
    r"the best value across all methods."
)

# Main-body table: means only (compact enough for the text width).
mean_table = build_table(
    with_std=False,
    label="tbl:results",
    caption=_DEFS + r" Per-seed standard deviations are reported in Appendix Table~\ref{tbl:results-std}.",
)
# Appendix table: full mean $\pm$ std across four seeds.
std_table = build_table(
    with_std=True,
    label="tbl:results-std",
    caption=_DEFS + r" Values are mean $\pm$ standard deviation across four seeds.",
)

for fname, content in [
    ("te_table_results.tex", mean_table),
    ("te_table_results_std.tex", std_table),
]:
    Path(fname).write_text(content)
    print(f"Wrote {fname}")

# %%
