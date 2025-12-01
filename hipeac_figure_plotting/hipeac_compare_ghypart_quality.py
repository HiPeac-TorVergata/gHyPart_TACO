#!/usr/bin/env python3
# -------------------------------------------------------------------------------
# Tor Vergata team version of compare_ghypart_quality.py
# Used for the reproducibility study of gHyPart (HiPEAC Students Challenge 2025).
#
# There is no original code in author's work similar to this script
# -------------------------------------------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# -------------------------------------------------------------------------------
# Load CSV while removing any "Unnamed" columns and trimming dataset names
# -------------------------------------------------------------------------------
def load_csv(filename):
    df = pd.read_csv(filename)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]  # drop ghost columns
    df.columns = df.columns.str.strip(',')                # remove trailing commas
    df['dataset'] = df['dataset'].str.strip()             # cleanup dataset name
    return df

# -------------------------------------------------------------------------------
# Compare the quality of gHyPart vs gHyPart_B
# Assumes BOTH CSV files contain the column "total_hyperedges"
# -------------------------------------------------------------------------------
def compare_quality(file_g, file_b):

    # Load cleaned input files
    df_g = load_csv(file_g)
    df_b = load_csv(file_b)

    # Remove performance metrics
    df_g = df_g[[c for c in df_g.columns if "_perf" not in c]]
    df_b = df_b[[c for c in df_b.columns if "_perf" not in c]]

    # Rename GHyPart columns
    df_g = df_g.rename(columns={
        "k=2_cut": "k=2",
        "k=3_cut": "k=3",
        "k=4_cut": "k=4",
    })

    # Rename BiPart columns
    df_b = df_b.rename(columns={
        "k=2_cut": "k=2",
        "k=3_cut": "k=3",
        "k=4_cut": "k=4",
    })

    # Merge by dataset
    merged = pd.merge(
        df_g,
        df_b,
        on=["dataset", "id", "hedges"],
        suffixes=("_g", "_b")
    )


    # Compute differences and normalized differences
    for k in ["k=2", "k=3", "k=4"]:
        merged[f"diff_{k}"] = merged[f"{k}_b"] - merged[f"{k}_g"]
        merged[f"diff_{k}_norm"] = merged[f"diff_{k}"] / merged["hedges"]
    # Print summary
    print("\n=== Normalized Differences (B - G) / hedges ===\n")
    

    print(
        merged[
            ["dataset", "hedges",
             "diff_k=2_norm", "diff_k=3_norm", "diff_k=4_norm"]
        ].to_string(index=False)
    )

    # -------------------------------
    # Single plot: 3 curves, each sorted independently
    # -------------------------------
    fig, ax = plt.subplots(figsize=(20,8))

    colors = {"k=2": "r", "k=3": "g", "k=4": "b"}

    for k in ["k=2", "k=3", "k=4"]:
        col = f"diff_{k}_norm"

        # Sort independently
        sorted_df = merged.sort_values(col).reset_index(drop=True)

        ax.plot(
            range(len(sorted_df)),
            sorted_df[col],
            label=f"{k}",
            color=colors[k],
            alpha=1
        )
        

    ax.set_xlabel("Sorted dataset index (each k independently sorted)",fontsize=20,fontweight='bold')
    ax.set_ylabel("Normalized difference",fontsize=20,fontweight='bold')
    ax.set_yscale("symlog", linthresh=1e-4)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)

    ax.tick_params(axis='y', which='major', labelsize=36)
    ax.tick_params(axis='x', which='major', labelsize=36, rotation=0)

    ax.legend(fontsize=40)

    # Save
    plt.savefig("figures/ghypart_quality_plot_all_sorted.png", dpi=300)
    print("Saved: ghypart_quality_plot_all_sorted.png")

    plt.show()



# -------------------------------------------------------------------------------
# Main execution
# -------------------------------------------------------------------------------
if __name__ == "__main__":

    file_ghypart = "../hipeac_results/hipeac_ghypart_all_best_NVIDIA_GeForce_RTX_4090_t32_2025-11-29.csv"
    file_ghypart_b = "../hipeac_results/hipeac_ghypart_B_all_best_NVIDIA_GeForce_RTX_4090_t32_2025-11-30.csv"

    if not os.path.exists(file_ghypart):
        print(f"Error: {file_ghypart} not found")
        sys.exit(1)

    if not os.path.exists(file_ghypart_b):
        print(f"Error: {file_ghypart_b} not found")
        sys.exit(1)

    compare_quality(file_ghypart, file_ghypart_b)
