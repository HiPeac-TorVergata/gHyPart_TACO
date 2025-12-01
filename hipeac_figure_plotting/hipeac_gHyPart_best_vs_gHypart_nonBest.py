#!/usr/bin/env python3
# -------------------------------------------------------------------------
# This is the Tor Vergata team version of hipeac_gHyPart_best_vs_gHypart_nonBest.py 
# used for the reproducibility study of gHyPart for HiPEAC Students Challenge 2025.
#
# There is no original code in author's work similar to this script
# -------------------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# -----------------------------
# Load data
# -----------------------------
df_standard = pd.read_csv("../hipeac_results/hipeac_ghypart_NVIDIA_GeForce_RTX_4090_t32_2025-09-14.csv")
df_best = pd.read_csv("../hipeac_results/hipeac_ghypart_all_best_NVIDIA_GeForce_RTX_4090_t32_2025-11-29.csv")


# Remove cut metrics
df_best = df_best[[c for c in df_best.columns if "_cut" not in c]]

df_best = df_best.rename(columns={
    "k=2_perf": "k=2",
    "k=3_perf": "k=3",
    "k=4_perf": "k=4",
})

df_speedup = pd.DataFrame({
    "id": df_standard["id"],
    "dataset": df_standard["dataset"],
    "hedges": df_best["hedges"],  # take hedges from df_best
})


print(df_standard)
print(df_best)

# Compute speedups
for k in ["k=2", "k=3", "k=4"]:
    df_speedup[f"speedup_{k}"] = df_best[k] / df_standard[k]


print(df_speedup)

# -----------------------------
# Single plot: all k curves independently sorted
# -----------------------------
fig, ax = plt.subplots(figsize=(30,11))

colors = {"k=2": "r", "k=3": "g", "k=4": "b"}

for k in ["k=2", "k=3", "k=4"]:
    col = f"speedup_{k}"
    
    # Sort independently
    y = np.sort(df_speedup[col].values)
    x = np.arange(len(y))
    
    # Plot curve
    ax.plot(x, y, label=f"{k}", color=colors[k], alpha=0.8)
    
    # Count datasets with speedup > 1
    count_gt1 = np.sum(y > 1)
    ax.axvline(len(y) - count_gt1, color=colors[k], linestyle=':', alpha=0.7,
               label=f"{k}: #Speedup>1 = {count_gt1}")

# -----------------------------
# Styling
# -----------------------------
ax.set_yscale("symlog", linthresh=0.05)
ax.set_ylim(bottom=0.01)  # ensures the y-axis starts from a small positive number
ax.set_xlabel("Sorted dataset index (each k independently)", fontsize=40,fontweight='bold')
ax.set_ylabel("Speedup Best / Standard",fontsize=40,fontweight='bold')
ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
ax.legend(fontsize=40)

ax.tick_params(axis='y', which='major', labelsize=36)
ax.tick_params(axis='x', which='major', labelsize=36, rotation=0)
# Save and show
plt.tight_layout()
plt.savefig("figures/ghypart_speedup_all_sorted.pdf", dpi=300)
plt.show()