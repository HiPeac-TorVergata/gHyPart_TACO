import matplotlib.pyplot as plt
import os
import pandas as pd
import matplotlib
if os.environ.get('DISPLAY', '') == '':
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gmean
import datetime
import subprocess

cpu_name = "unknown"
gpu_name = "unknown"

# -------------------------------------------------------------------------
# Detect CPU model name
# -------------------------------------------------------------------------
with open("/proc/cpuinfo") as f:
    for line in f:
        if "model name" in line:
            print(line.strip())
            cpu_name = line.strip().split(":")[1].split("CPU @")[0].strip()
            cpu_name = cpu_name.replace(" ","_").replace("(R)","").replace("(","").replace(")","")
            break


# -------------------------------------------------------------------------
# Detect GPU model name using nvidia-smi command. Note this could not work
# inside HPC context where modules environment doesn't let users use 
# nvidia-smi.
# -------------------------------------------------------------------------
try:
    result = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        encoding="utf-8"
    )
    gpus = [line.strip() for line in result.splitlines() if line.strip()]
    if gpus:

        for i, gpu in enumerate(gpus):
            print(f"GPU {i}: {gpu}")
            if i == 0:
                gpu_name = gpu
        gpu_name = gpu_name.replace(" ","_")
    else:
       gpu_name = 'not_present'
except FileNotFoundError:
    gpu_name = 'not_present'





current_date = datetime.date.today()

file1 = '../figure_plotting/results/bipart_perf_xeon_4214R_24cores_t12.csv'
file2 = '../figure_plotting/results/mtkahypar_perf_xeon_4214R_24cores_t12.csv'
file4 = '../figure_plotting/results/prof_hmetis_xeon_4214R_all.csv'


colors = ["#FF0000", "#00FF00", "#0000FF", "#FFFF00", 
          "#800080", "#ffc0cb", "#ffa500", "#808080", 
          "#007fff", "#8e52dc", "#ff7f50", "#ff4500", 
          "#396e04", "#ff00ff", "#ff69b4", "#489177",
          "#CC99FF", "#FFCC99", "#99CCFF", "#CCFF99",]


def perf_comp_all_with_2way(k,gpu):
    file3  = ""
    if gpu == "1060":
        file3 = '../hipeac_results/hipeac_ghypart_NVIDIA_GeForce_GTX_1060_3GB_t8_2025-11-22.csv'
    if gpu == "2070":
        file3 = '../hipeac_results/hipeac_ghypart_NVIDIA_GeForce_RTX_2070_t12_2025-11-21.csv'
    if gpu == "q5000":
        file3 = '../hipeac_results/hipeac_ghypart_QUADRO_RTX_5000_2025-11-22.csv'
    if gpu == "4090":
        file3 = '../hipeac_results/hipeac_ghypart_NVIDIA_GeForce_RTX_4090_t32_2025-09-14.csv'
    data1 = pd.read_csv(file1)
    data2 = pd.read_csv(file2)
    data3 = pd.read_csv(file3)
    data4 = pd.read_csv(file4)

    data1['BiPart']=data1['k=2']
    data2['Mt-KaHyPar-SDet']=data2['k=2']
    data3['gHyPart']=data3['k=2']
    data4['part2_time']
    merged_data = pd.concat([data1['BiPart'],data2['Mt-KaHyPar-SDet'],data3['gHyPart'], data4['part2_time']], axis=1)
    merged_data = merged_data[merged_data['gHyPart']!= 0.0]
    
    merged_data['g_ratio'] = merged_data['BiPart'] / merged_data['gHyPart'] # merged_data['best']
    merged_data = merged_data.sort_values(by='g_ratio')
    y1 = merged_data['BiPart']
    y6 = merged_data['part2_time']
    y2 = merged_data['Mt-KaHyPar-SDet']
    y4 = merged_data['gHyPart']
    
    print (y1.sort_index(),y2.sort_index(),y6.sort_index(),y4.sort_index())
    x = merged_data

    index_list = range(len(x))
    
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['legend.fontsize'] = 32
    plt.rcParams['legend.frameon'] =  True

    fig, ax = plt.subplots(figsize=(30, 11))
    ylim = 100
    ax.set_yscale('log', base=10)  
    ax.set_ylim(0, ylim)  
    ax.tick_params(axis='y', which='major', labelsize=48)
    ax.set_ylabel('Speedup over BiPart', va='center', fontsize=60, fontweight='bold', labelpad=45) 
    ax.set_xlabel('Hypergraphs', va='center', fontsize=60, fontweight='bold', labelpad=40)
    
    xlim = 580
    ax.set_xlim(0, xlim)
    app_ticks = [0, 100, 200, 300, 400, len(x)-1, 505]
    ax.xaxis.set_ticks([0, 100, 200, 300, 400, len(x)-1])
    ax.tick_params(axis='x', which='major', labelsize=50, rotation=0)
    
    dot1 = ax.plot(index_list, y1/y1, marker='+', linestyle='-', label='BiPart', color="#808080", linewidth=5)
    dot6 = ax.plot(index_list, y1/y6, marker='o', markersize=6, linestyle='-', label='hMetis', color=colors[16], linewidth=5)
    dot2 = ax.plot(index_list, y1/y2, marker='*', markersize=12, linestyle='-', label='Mt-KaHyPar', color="#396e04", linewidth=3)
    dot4 = ax.plot(index_list, y1/y4, marker='^', markersize=6, linestyle='-', label='ɢHʏPᴀʀᴛ', color=colors[8], linewidth=5)
    
    last_center = 540
    bar1_pos = last_center - 14
    bar2_pos = last_center - 5
    bar3_pos = last_center + 4
    bar4_pos = last_center + 13
    bar5_pos = last_center + 22
    bar6_pos = last_center + 31
    width = 8
    bar1 = ax.bar(bar1_pos, gmean(y1/y1), width=width, label='', edgecolor='black', color="#808080", hatch='//')
    bar6 = ax.bar(bar2_pos, gmean(y1/y6), width=width, label='', edgecolor='black', color=colors[16], hatch='.')
    bar2 = ax.bar(bar3_pos, gmean(y1/y2), width=width, label='', edgecolor='black', color="#396e04", hatch='.')
    bar4 = ax.bar(bar5_pos, gmean(y1/y4), width=width, label='', edgecolor='black', color=colors[8], hatch='o')
    
    ax.axvline([len(x) + 15], color='grey', linestyle='dashed', linewidth=2)

    ax.yaxis.grid(True, color='gray', linestyle = '--', linewidth = 0.5)  
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontweight('bold')
        
    last_label_pos = 550
    xticks = list(ax.get_xticks()) + [last_label_pos]
    xticklabels = list(ax.get_xticklabels()) + ['GMean']
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)

    xtick_labels = ax.get_xticklabels()
    last_xtick_label = xtick_labels[-1]
    last_xtick_label.set_fontsize(50)
    last_xtick_label.set_rotation(0)

    last_value1 = gmean(y1/y1)
    last_value2 = gmean(y1/y6)
    last_value3 = gmean(y1/y2)
    last_value5 = gmean(y1/y4)
    ax.text(bar1_pos, last_value1 + 0.5, "[{:.2f}]".format(last_value1), color="#808080", ha='center', fontsize=35, rotation=90)
    ax.text(bar2_pos, gmean(y1/y6) + .5, "[{:.2f}]".format(gmean(y1/y6)), color=colors[16], ha='center', fontsize=35, rotation=90)
    ax.text(bar3_pos+0.5, last_value3 + 0.5, "[{:.2f}]".format(last_value3), color="#396e04", ha='center', fontsize=35, rotation=90)
    ax.text(bar5_pos+0.5, last_value5 + 6, "[{:.2f}]".format(last_value5), color=colors[8], ha='center', fontsize=35, rotation=90)
    
    handles, labels = ax.get_legend_handles_labels()
    legend = ax.legend(handles, labels, bbox_to_anchor=(0.45, 0.88), loc='center', ncol=3, fontsize=35, frameon=True)
    frame = legend.get_frame()
    frame.set_edgecolor('#808080')  
    frame.set_linewidth(2)  
    frame.set_alpha(1)  

    spines = ax.spines
    spines['top'].set_linewidth(5)
    spines['bottom'].set_linewidth(5)
    spines['left'].set_linewidth(5)
    spines['right'].set_linewidth(5)


    plt.tight_layout()

    outfile = f'figures/hipeac_figure_11_{gpu_name}_{current_date}.pdf'


    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.show()


# -------------------------------------------------------------------------
# This script generates original Figure 11. Please note that some changes to 
# this script should be done in order to collect data of gHyPart-B and 
# gHyPart-O.
# -------------------------------------------------------------------------
if __name__ == '__main__':
    gpus=["1060","2070","4090","q5000"]
    for i in gpus:
        perf_comp_all_with_2way(2,i)
    
