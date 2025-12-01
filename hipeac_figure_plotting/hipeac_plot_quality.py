#!/usr/bin/env python3
# -------------------------------------------------------------------------
# This is the Tor Vergata team version of hipeac_plot_quality.py used for the 
# reproducibility study of gHyPart for HiPEAC Students Challenge 2025.
#
# There is no original code in author's work similar to this script
# -------------------------------------------------------------------------
"""
hipeac_plot_quality.py

Script aggiornato per calcolare e plottare metriche sui cut normalizzati e boxplot
con riferimento al partizionatore `bipart`.

Input (Expected CSV files):
  * hipeac_ghypart_all_best_NVIDIA_GeForce_RTX_4090_t32_2025-11-29.csv
      header: id,dataset,k=2,k=3,k=4,hedges
  * hipeac_bipart_quality_AMD_Ryzen_5_3600_6-Core_Processor_t12_2025-11-29.csv
      header: id,dataset,k=2,k=3,k=4,hedges
  * hipeac_mtkahypar_quality_Intel_Xeon_Gold_6142_t64_2025-11-26.csv
      header: id,dataset,k=2,k=3,k=4,hedges

Output:
  * boxplot PNG (default: quality_quantification_comparison.pdf)

Uso:
  python3 hipeac_plot_quality.py

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


DEFAULT_FILES = {
    'ghypart': '../hipeac_results/hipeac_ghypart_all_best_NVIDIA_GeForce_RTX_4090_t32_2025-11-29.csv',
    'bipart': '../hipeac_results/hipeac_bipart_quality_AMD_Ryzen_5_3600_6-Core_Processor_t12_2025-11-29.csv',
    'mtkahypar': '../hipeac_results/hipeac_mtkahypar_quality_Intel_Xeon_Gold_6142_t64_2025-11-26.csv',
    'hmetis': '../hipeac_results/hipeac_prof_hmetis.csv'
}


def count_metric_thresholds_per_configuration(df: pd.DataFrame, reference: str = 'ghypart'):
    """
    Compute and print comparative statistics of all tools relative to a reference tool.
    Handles non-numeric values (e.g., 'segfault') by discarding those rows and prints info.
    """
    print(f"Comparative stats relative to {reference}:")

    # Get all reference columns
    ref_cols = [c for c in df.columns if c.startswith(reference + '_k')]

    for ref_col in ref_cols:
        k = ref_col.split('_k')[-1]

        # All other tools for the same k
        other_cols = [c for c in df.columns if c.endswith(f"_k{k}") and not c.startswith(reference)]
        print(f"\nComparing {reference}_k{k} against: {other_cols}")

        for col in other_cols:
            try:
                # Extract relevant sub-DataFrame
                sub_df = df[['dataset', 'hedges', ref_col, col]].copy()
                
                # Convert to numeric safely
                sub_df[ref_col] = pd.to_numeric(sub_df[ref_col], errors='coerce')
                sub_df[col] = pd.to_numeric(sub_df[col], errors='coerce')
                sub_df['hedges'] = pd.to_numeric(sub_df['hedges'], errors='coerce')


                # Identify rows that will be dropped
                dropped_rows = sub_df[sub_df.isna().any(axis=1)]
                if not dropped_rows.empty:
                    print(f"\nRows with non-numeric values that will be discarded for {col} vs {ref_col}:")
                    print(dropped_rows[['dataset', ref_col, col, 'hedges']])                

                # Drop rows with NaN
                sub_df = sub_df.dropna()

                # Compute metric difference
                metric_diff = (sub_df[ref_col] - sub_df[col]) / sub_df["hedges"]


                n_total = len(metric_diff)
                if n_total == 0:
                    print(f"{col} vs {reference}_k{k}: No valid data to compare.")
                    continue

                n_abs_less_05 = (metric_diff.abs() < 0.005).sum()
                n_greater_05 = (metric_diff > 0.005).sum()
                n_less_minus05 = (metric_diff < -0.005).sum()

                print(f"{reference}_k{k} vs {col}: total={n_total}, "
                      f"Comparable={n_abs_less_05 / n_total * 100:.1f}%, "
                      f"Worse={n_greater_05 / n_total * 100:.1f}%, "
                      f"Better={n_less_minus05 / n_total * 100:.1f}%")

                # Cases with |metric_diff| > 1
                high_metric = sub_df[metric_diff.abs() > 1]
                if not high_metric.empty:
                    print(f"\nCases with |metric_diff| > 1 for {col} vs {reference}_k{k}:")
                    for idx, row in high_metric.iterrows():
                        print(f"Dataset={row['dataset']}, {col}={row[col]}, {reference}_k{k}={row[ref_col]}, "
                              f"hedges={row['hedges']}, metric_diff={metric_diff[idx]:.3f}")

            except Exception as e:
                print(f"\nERROR computing {col} vs {reference}_k{k}: {e}")
                sample_problem = df[['dataset', ref_col, col, 'hedges']].head(10)
                print("Sample data for debugging:")
                print(sample_problem)


def plot_box_standard_detailed(df: pd.DataFrame, out_pdf: str, reference: str = 'ghypart'):
    """
    Plot boxplots of partitioning quality difference relative to a reference tool.
    Automatically handles non-numeric values by discarding problematic rows.
    """
    import matplotlib.pyplot as plt

    # Mapping for nicer names
    name_mapping = {
        'ghypart': 'gHyPart',
        'bipart': 'BiPart',
        'mtkahypar': 'mt-KaHyPar',
        'hmetis': 'hMetis'
    }

    all_data = []
    labels = []

    # Reference columns
    ref_cols = [c for c in df.columns if c.startswith(reference + '_k')]
    
    for ref_col in ref_cols:
        k = ref_col.split('_k')[-1]

        # Other tools for the same k
        other_cols = [c for c in df.columns if c.endswith(f"_k{k}") and not c.startswith(reference)]

        for col in other_cols:
            # Extract sub-DataFrame for safety
            sub_df = df[['hedges', ref_col, col]].copy()
            sub_df[ref_col] = pd.to_numeric(sub_df[ref_col], errors='coerce')
            sub_df[col] = pd.to_numeric(sub_df[col], errors='coerce')
            sub_df['hedges'] = pd.to_numeric(sub_df['hedges'], errors='coerce')

            # Drop rows with any NaNs
            sub_df = sub_df.dropna()

            # Compute metric difference safely
            metric_diff = (sub_df[ref_col] - sub_df[col]) / sub_df['hedges']

            all_data.append(metric_diff.values)
            labels.append(f"{name_mapping.get(col.split('_')[0], col.split('_')[0])} (k={k})")

    fig, ax = plt.subplots(figsize=(30, 11))

    # Create boxplot
    bp = ax.boxplot(
        all_data,
        labels=labels,
        patch_artist=True,
        showmeans=False,
        showfliers=False,
        whis=2
    )

    # Set y-axis label
    ax.set_ylabel("Partitioning Quality Difference",
                fontsize=30, fontweight='bold', labelpad=20)
    # Set tick label fonts
    for tick in ax.get_xticklabels():
        tick.set_fontsize(30)
        tick.set_rotation(45)
        tick.set_fontweight("bold")

    for tick in ax.get_yticklabels():
        tick.set_fontsize(40)
        tick.set_fontweight("bold")


    plt.tight_layout()
    plt.savefig(f"figures/{out_pdf}", dpi=300)
    print(f"Saved boxplot to figures/{out_pdf}")




def read_csv_safe(path: str) -> pd.DataFrame:
    """Read a CSV, strip column names, ensure 'id' exists."""
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    # Ensure id column exists
    if 'id' not in df.columns:
        df.insert(0, 'id', range(len(df)))

    # Strip dataset names if present
    if 'dataset' in df.columns:
        df['dataset'] = df['dataset'].astype(str).str.strip()

    # Ensure hedges column exists
    if 'hedges' not in df.columns:
        df['hedges'] = pd.NA

    return df



# -------------------------------------------------------------------------
# This scripts reads quality results from all works in order to compare 
# them respectively to gHyPart. Note that results present in Table 3 inside
# the report are collected from the output of this script. Moreover Figure 8
# is generated as plot at the end of the script.
#
# Note also that this scripts generates merged_all_quality.csv file.
# -------------------------------------------------------------------------
if __name__ == '__main__':
    # Load CSVs
    ghypart = read_csv_safe(DEFAULT_FILES['ghypart'])
    bipart = read_csv_safe(DEFAULT_FILES['bipart'])
    mtkahypar = read_csv_safe(DEFAULT_FILES['mtkahypar'])
    hmetis = read_csv_safe(DEFAULT_FILES['hmetis'])

    print("-----------------------------------------------------------")
    print("Original Dataset")
    print(f"ghypart: {ghypart.shape}")
    print(f"bipart: {bipart.shape}")
    print(f"mtkahypar: {mtkahypar.shape}")
    print(f"hmetis: {hmetis.shape}")

    # --- Clean ghypart ---
    # Keep only the columns ending with '_cut' + id, dataset, hedges
    cut_cols = [c for c in ghypart.columns if c.endswith('_cut')]
    ghypart = ghypart[['id', 'dataset', 'hedges'] + cut_cols].copy()

    # Rename columns: ghypart_kX_cut -> ghypart_kX
    rename_map = {col: f"ghypart_k{col.split('=')[1]}" for col in cut_cols}
    ghypart = ghypart.rename(columns=rename_map)

    # Remove any leftover '_cut' just in case
    ghypart = ghypart.rename(columns=lambda x: x.replace('_cut',''))

    # --- Clean bipart ---
    cut_cols = [c for c in bipart.columns if c.startswith('k=')]
    bipart = bipart[['id', 'dataset', 'hedges'] + cut_cols].copy()
    bipart = bipart.rename(columns=lambda x: x.replace('k=', 'bipart_k'))

    # --- Clean mtkahypar ---
    cut_cols = [c for c in mtkahypar.columns if c.startswith('k=')]
    mtkahypar = mtkahypar[['id', 'dataset', 'hedges'] + cut_cols].copy()
    mtkahypar = mtkahypar.rename(columns=lambda x: x.replace('k=', 'mtkahypar_k'))

    # --- Clean hmetis ---
    extime_cols = [c for c in hmetis.columns if c.startswith('extimek')]
    hmetis = hmetis.drop(columns=extime_cols)
    cut_cols = [c for c in hmetis.columns if c.startswith('cutk')]
    rename_dict = {c: f"hmetis_k{c[-1]}" for c in cut_cols}
    hmetis = hmetis.rename(columns=rename_dict)
    hmetis = hmetis[['id', 'dataset', 'hedges'] + list(rename_dict.values())]

    # --- Ensure numeric hedges ---
    for df_name, df in zip(['ghypart','bipart','mtkahypar','hmetis'],
                           [ghypart, bipart, mtkahypar, hmetis]):

        df['hedges'] = pd.to_numeric(df['hedges'], errors='coerce')
        nan_rows = df[df['hedges'].isna()]
        if not nan_rows.empty:
            print(f"{df_name} has invalid hedges:")
            print(nan_rows[['id','dataset','hedges']])
        # Fill NaNs with 0 or drop
        df['hedges'] = df['hedges'].fillna(0).astype(int)

    # --- Merge all on id, dataset, hedges ---
    merged = ghypart.merge(bipart, on=['id','dataset','hedges'], how='outer') \
                     .merge(mtkahypar, on=['id','dataset','hedges'], how='outer') \
                     .merge(hmetis, on=['id','dataset','hedges'], how='outer')

    print(f"Merged shape: {merged.shape}")
    merged.to_csv("merged_all_quality.csv", index=False)
    print("Saved merged CSV: merged_all_quality.csv")
    print("-----------------------------------------------------------")

    print("")
    print("")
    print("-----------------------------------------------------------")
    print("Compare results")

    count_metric_thresholds_per_configuration(merged, reference="ghypart")
    print("-----------------------------------------------------------")

    plot_box_standard_detailed(merged, "quality_quantification_comparison.pdf", reference="ghypart")
