# Figure Plotting
In this folder, are reported all the scripts used to create the figures in the report

## Script Description
All scripts have hard coded file from which they take data used inside plots. All file paths are present at the beginning of the file, so that there is clarity on which file which script is using. Of course remember to run scripts inside `hipeac_scripts` to regenerate `.csv` files and change file path manually inside `hipeac_figure_plotting` in order to reproduce results on different hardware.


## Figure Generation Description
In the list above is reported the relation between figures present in the report and the scripts that generates them.
| Element | Source File | Source Script |
|----------------|:--------:|:--------:|
| Figure 1   | `hipeac_figure_new_compare_gpus.pdf`  |`hipeac_comp_all_gpu.py` |
| Figure 2   | `hipeac_figure_11_NVIDIA_GeForce_RTX_4090_2025-12-01.pdf` | `hipeac_perf_comp_all_works_2way.py` |
| Figure 3   | Taken directly from the original paper (Figure 11) | `perf_comp_all_works_2way.py` |
| Figure 4   | `benchmarks_all_patterns_top100_3090_2025-12-01.pdf` | `hipeac_benchmarks_all.py` |
| Figure 5   | Taken directly from the original paper (Figure 12) | `benchmarks_all.py` |
| Figure 6   | `work_perf_comp_all_NVIDIA_GeForce_RTX_4090_2025-12-01.pdf` | Generated using the script `hipeac_perf_comp_all_works_kway.py` |
| Figure 7   | Taken directly from the original paper (Figure 18) | `perf_comp_all_works_kway.py` |
| Figure 8   | `quality_quantification_comparison.pdf` | `hipeac_plot_quality.py` |
| Figure 9   | `ghypart_speedup_all_sorted.pdf` | `hipeac_gHyPart_best_vs_gHypart_nonBest.py` |
| Figure 10  | `ghypart_quality_plot_all_sorted.png` | `hipeac_compare_ghypart_quality.py` |
| Table 3    | Script stdout | `hipeac_plot_quality.py` |
