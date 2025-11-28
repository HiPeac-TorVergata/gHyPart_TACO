#!/usr/bin/env python3
"""
plot_cut_normalized.py

Script aggiornato per calcolare e plottare metriche sui cut normalizzati e boxplot
con riferimento al partizionatore `bipart`.

Input (file CSV attesi nella stessa cartella):
  * hipeac_ghypart_quality_NVIDIA_GeForce_RTX_2070_2025-11-25.csv
      header: id,dataset,k=2,k=3,k=4,hedges
  * hipeac_bipart_quality_Intel_Xeon_Gold_6142_t64_2025-11-26.csv
      header: id,dataset,k=2,k=3,k=4,hedges
  * hipeac_mtkahypar_quality_Intel_Xeon_Gold_6142_t64_2025-11-26.csv
      header: id,dataset,k=2,k=3,k=4,hedges

Output:
  * boxplot PNG (default: boxplot_relative_to_bipart.png)

Uso:
  python3 plot_cut_normalized.py --out boxplot.png

"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

DEFAULT_FILES = {
    'ghypart': '../results/hipeac_ghypart_quality_NVIDIA_GeForce_RTX_2070_2025-11-25.csv',
    'bipart': '../results/hipeac_bipart_quality_Intel_Xeon_Gold_6142_t64_2025-11-26.csv',
    'mtkahypar': '../results/hipeac_mtkahypar_quality_Intel_Xeon_Gold_6142_t64_2025-11-26.csv',
    'hmetis': '../results/prof_hmetis.csv'
}

def read_hmetis_file(path: Path, source_name: str = 'hmetis') -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    cut_cols = ['cutk2', 'cutk3', 'cutk4']
    id_vars = ['dataset', 'hedges']

    long = df.melt(id_vars=id_vars, value_vars=cut_cols, var_name='k_col', value_name='cut')
    long['k'] = long['k_col'].map(lambda x: int(x.replace('cutk','')))

    long['source'] = source_name
    long['cut'] = pd.to_numeric(long['cut'], errors='coerce')
    long['hedges'] = pd.to_numeric(long['hedges'], errors='coerce')
    long['dataset'] = long['dataset'].astype(str)
    return long[['dataset','k','cut','hedges','source']]

def parse_k(colname: str) -> int:
    s = str(colname).strip()
    if '=' in s:
        try:
            return int(s.split('=')[1])
        except Exception:
            pass
    # fallback: take last digit
    for ch in reversed(s):
        if ch.isdigit():
            return int(ch)
    raise ValueError(f"Unable to parse k from column name '{colname}'")


def read_quality_file(path: Path, source_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    df.columns = [c.strip() for c in df.columns]

    possible_k_cols = [c for c in df.columns if c.startswith('k')]
    if not possible_k_cols:

        possible_k_cols = [c for c in df.columns if 'k' in c.lower() and any(d in c for d in '234')]

    id_vars = [c for c in ['id', 'dataset', 'hedges'] if c in df.columns]
    if 'dataset' not in df.columns and 'id' in df.columns:

        df = df.rename(columns={'id': 'dataset'})
        if 'dataset' not in id_vars:
            id_vars.append('dataset')

    long = df.melt(id_vars=[c for c in id_vars if c in df.columns], value_vars=possible_k_cols,
                   var_name='k_col', value_name='cut')

    long['k'] = long['k_col'].map(parse_k)

    for c in ['dataset', 'hedges']:
        if c not in long.columns:
            long[c] = np.nan
    long['source'] = source_name

    long['cut'] = pd.to_numeric(long['cut'], errors='coerce')
    long['hedges'] = pd.to_numeric(long['hedges'], errors='coerce')
    long['dataset'] = long['dataset'].astype(str)
    return long[['dataset', 'k', 'cut', 'hedges', 'source']]


def build_master(ghyp_df: pd.DataFrame, bipart_df: pd.DataFrame, mtk_df: pd.DataFrame):
    all_df = pd.concat([ghyp_df, bipart_df, mtk_df], ignore_index=True)

    hedges_by_dataset = {}
    tmp = all_df[['dataset', 'hedges']].dropna()
    for ds, val in zip(tmp['dataset'], tmp['hedges']):
        if ds not in hedges_by_dataset:
            try:
                hedges_by_dataset[ds] = float(val)
            except Exception:
                hedges_by_dataset[ds] = np.nan

    all_df['hedges_global'] = all_df['dataset'].map(hedges_by_dataset)

    def compute_norm(r):
        try:
            if pd.isna(r['hedges_global']) or r['hedges_global'] == 0:
                return np.nan
            return float(r['cut']) / float(r['hedges_global'])
        except Exception:
            return np.nan

    all_df['metric'] = all_df.apply(compute_norm, axis=1)
    return all_df, hedges_by_dataset


def count_metric_thresholds_per_configuration(all_df: pd.DataFrame, reference: str = 'hmetis') -> None:
    """
    Conta e stampa statistiche della metrica comparativa rispetto a un reference.
    Inoltre stampa tutti i casi in cui |metric_diff| > 1 con informazioni dettagliate.
    """
    # seleziona le righe del reference
    ref = all_df[all_df['source'] == reference][['dataset', 'k', 'cut']].rename(columns={'cut': 'cut_ref'})

    # merge con tutti gli altri
    merged = all_df.merge(ref, on=['dataset', 'k'], how='left', validate='many_to_one')

    # calcola la metrica comparativa normalizzata
    merged['metric_diff'] = (merged['cut'] - merged['cut_ref']) / merged['hedges_global']

    # escludi il reference stesso
    merged = merged[merged['source'] != reference]
    merged = merged.dropna(subset=['metric_diff'])

    # etichette per configurazione
    merged['label'] = merged['source'] + '_k' + merged['k'].astype(int).astype(str)
    labels = sorted(merged['label'].unique())

    print(f"Comparative stats relative to {reference}:")
    for lab in labels:
        vals = merged[merged['label'] == lab]['metric_diff']
        n_total = len(vals)
        n_abs_less_05 = (vals.abs() < 0.005).sum()
        n_greater_05 = (vals > 0.005).sum()
        n_less_minus05 = (vals < -0.005).sum()
        print(
            f"{lab}: total = {n_total}, Comparable = {n_abs_less_05 / n_total * 100:.1f}%, "
            f"Worse = {n_greater_05 / n_total * 100:.1f}%, Better = {n_less_minus05 / n_total * 100:.1f}%"
        )

    # stampa dettagli dei casi con metrica > 1
    high_metric = merged[merged['metric_diff'].abs() > 1]
    if not high_metric.empty:
        print("\nCases with |metric_diff| > 1:")
        for idx, row in high_metric.iterrows():
            print(f"Dataset: {row['dataset']}, Source: {row['source']}, k={row['k']}, "
                  f"cut={row['cut']}, cut_ref={row['cut_ref']}, hedges={row['hedges_global']}, "
                  f"metric_diff={row['metric_diff']:.3f}")


def plot_box_standard_detailed(all_df: pd.DataFrame, out_pdf: Path, reference: str = 'hmetis'):
    name_mapping = {
        'ghypart': 'gHyPart',
        'bipart': 'BiPart',
        'mtkahypar': 'mt-KaHyPar',
        'hmetis': 'hMetis'
    }

    # seleziona il riferimento
    ref = all_df[all_df['source'] == reference][['dataset','k','cut']].rename(columns={'cut':'cut_ref'})
    merged = all_df.merge(ref, on=['dataset','k'], how='left', validate='many_to_one')
    merged['metric_diff'] = (merged['cut'] - merged['cut_ref']) / merged['hedges_global']
    merged = merged[merged['source'] != reference]
    merged = merged.dropna(subset=['metric_diff'])

    # etichette originali
    merged['label'] = merged['source'] + '_k' + merged['k'].astype(int).astype(str)
    labels = sorted(merged['label'].unique(), key=lambda x: (x.split('_k')[0], int(x.split('_k')[1])))
    groups = [merged[merged['label'] == lab]['metric_diff'].values for lab in labels]

    # etichette formattate
    formatted_labels = [f"{name_mapping.get(lab.split('_k')[0], lab.split('_k')[0])} (k={lab.split('_k')[1]})"
                        for lab in labels]

    # imposta parametri globali font come nel tuo script di riferimento
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['legend.fontsize'] = 32
    plt.rcParams['legend.frameon'] = True

    fig, ax = plt.subplots(figsize=(30, 11))  # figura ampia come nello script di riferimento
    plt.style.use('seaborn-v0_8-whitegrid')

    # boxplot
    bp = ax.boxplot(groups,
                    patch_artist=True,
                    showfliers=False,
                    widths=0.6,
                    boxprops=dict(facecolor='skyblue', edgecolor='black', alpha=0.7),
                    whiskerprops=dict(color='black', linewidth=3),
                    capprops=dict(color='black', linewidth=3),
                    medianprops=dict(color='darkblue', linewidth=3),
                    meanprops=dict(marker='D', markeredgecolor='black', markerfacecolor='red'),
                    showmeans=True)

    ax.set_xticklabels(formatted_labels, rotation=45, ha='center', fontsize=50)
    ax.set_ylabel("Partitioning Quality Difference", fontsize=40, fontweight='bold', labelpad=45)
    ax.tick_params(axis='y', which='major', labelsize=30)
    ax.tick_params(axis='x', which='major', labelsize=30)


    # spessori dei bordi
    spines = ax.spines
    spines['top'].set_linewidth(5)
    spines['bottom'].set_linewidth(5)
    spines['left'].set_linewidth(5)
    spines['right'].set_linewidth(5)

    plt.tight_layout()
    plt.savefig(out_pdf, dpi=300, bbox_inches='tight')
    print(f"Saved styled box plot to {out_pdf}")




def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--ghypart', default=DEFAULT_FILES['ghypart'], help='CSV ghypart')
    parser.add_argument('--bipart', default=DEFAULT_FILES['bipart'], help='CSV bipart')
    parser.add_argument('--mtkahypar', default=DEFAULT_FILES['mtkahypar'], help='CSV mt-kahypar')
    parser.add_argument('--hmetis', default=DEFAULT_FILES['hmetis'], help='CSV hmetis')
    parser.add_argument('--out', default='boxplot_relative_to_bipart.pdf', help='Output PDF file for boxplot')
    parser.add_argument('--ref', default='hmetis', help='Reference partitioner (default: hmetis)')
    parser.add_argument('--standard-box-out', default='boxplot_standard.pdf', help='Output PDF file for standard boxplot')
    args = parser.parse_args(argv)


    paths = {k: Path(v) for k, v in [('ghypart', args.ghypart), ('bipart', args.bipart), ('mtkahypar', args.mtkahypar), ('hmetis', args.hmetis)]}
    for k, p in paths.items():
        if not p.exists():
            print(f"Warning: file for {k} not found at {p} — continuing but this tool will miss data.")


    ghyp = read_quality_file(paths['ghypart'], 'ghypart') if paths['ghypart'].exists() else pd.DataFrame(columns=['dataset','k','cut','hedges','source'])
    bip = read_quality_file(paths['bipart'], 'bipart') if paths['bipart'].exists() else pd.DataFrame(columns=['dataset','k','cut','hedges','source'])
    mtk = read_quality_file(paths['mtkahypar'], 'mtkahypar') if paths['mtkahypar'].exists() else pd.DataFrame(columns=['dataset','k','cut','hedges','source'])
    hmetis = read_hmetis_file(paths['hmetis'], 'hmetis') if paths['hmetis'].exists() else pd.DataFrame(columns=['dataset','k','cut','hedges','source'])


    all_df, hedges_map = build_master(ghyp, bip, mtk)
    if not hmetis.empty:
        all_df = pd.concat([all_df, hmetis], ignore_index=True)

    count_metric_thresholds_per_configuration(all_df, reference=args.ref)

    plot_box_standard_detailed(all_df, Path(args.standard_box_out), reference=args.ref)





if __name__ == '__main__':
    main()
