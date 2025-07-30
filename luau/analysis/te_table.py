# %%

import matplotlib.pyplot as plt
import numpy as np
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

tmp_df = df_lunar  # skip first 7 points for lunar lander
steps = tmp_df["Step"].to_numpy()
transfer_rewards = tmp_df["DIAA"].to_numpy()
baseline_rewards = tmp_df["Baseline"].astype(float).to_numpy()

min_reward = min(np.min(transfer_rewards), np.min(baseline_rewards))
normalized_transfer = transfer_rewards - min_reward
normalized_baseline = baseline_rewards - min_reward

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
