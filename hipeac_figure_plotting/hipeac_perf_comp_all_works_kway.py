import matplotlib.pyplot as plt
import os
import pandas as pd
import matplotlib
import os
if os.environ.get('DISPLAY', '') == '':
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gmean
import datetime
import subprocess

current_date = datetime.date.today()


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




file = '../figure_plotting/results/works_all_comp_3090_2024-01-08.csv'
file12 = '../figure_plotting/results/prof_hmetis_xeon_4214R_all.csv'
file13 = '../hipeac_results/hipeac_ghypart_NVIDIA_GeForce_RTX_4090_t32_2025-09-14.csv'
file14 = '../figure_plotting/results/bipart_perf_xeon_4214R_t12_2024-10-25.csv'
file15 = '../figure_plotting/results/mtkahypar_perf_xeon_4214R_24cores_t12_241020.csv'

data12 = pd.read_csv(file12)
data13 = pd.read_csv(file13)
data14 = pd.read_csv(file14)
data15 = pd.read_csv(file15)


colors = ["#FF0000", "#00FF00", "#0000FF", "#FFFF00", 
          "#800080", "#ffc0cb", "#ffa500", "#808080", 
          "#007fff", "#8e52dc", "#ff7f50", "#ff4500", 
          "#396e04", "#ff00ff", "#ff69b4", "#489177",
          "#CC99FF", "#FFCC99", "#99CCFF", "#CCFF99",]


def perf_comp_all_with_kway():
    bpk2 = data14['k=2']
    hmk2 = data12['part2_time']
    mtk2 = data15['k=2']
    ghk2 = data13['k=2']
    
    bpk3 = data14['k=3']
    hmk3 = data12['part3_time']
    mtk3 = data15['k=3']
    ghk3 = data13['k=3']
    
    bpk4 = data14['k=4']
    hmk4 = data12['part4_time']
    mtk4 = data15['k=4']
    ghk4 = data13['k=4']
    
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['legend.fontsize'] = 32
    plt.rcParams['legend.frameon'] =  True

    fig, ax = plt.subplots(figsize=(25, 11))
    ylim = 100
    ax.set_yscale('log', base=10)  
    ax.set_ylim(0, ylim)  
    ax.yaxis.set_ticks([0.1, 1, 10, ylim])  
    ax.tick_params(axis='y', which='major', labelsize=48)
    ax.set_ylabel('Speedup over BiPart', va='center', fontsize=50, fontweight='bold', labelpad=45)  
    ax.set_xlabel('', va='center', fontsize=40, fontweight='bold', labelpad=40)
    
    ax.set_xlim(0, 10)
    x_positions = [1.2, 5.0, 8.7]
    x_labels = ['k=2', 'k=3', 'k=4']
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels)
    ax.tick_params(axis='x', which='major', labelsize=40, rotation=0)
    
    width = .5
    bar1 = ax.bar(2*0+.5, gmean(bpk2/bpk2), width=width, label='BiPart', edgecolor='black', color="#808080", hatch='//')
    bar2 = ax.bar(2*0+1, gmean(bpk2/hmk2), width=width, label='hMetis', edgecolor='black', color=colors[5], hatch='.')
    bar3 = ax.bar(2*0+1.5, gmean(bpk2/mtk2), width=width, label='Mt-KaHyPar', edgecolor='black', color="#396e04", hatch='o')
    bar4 = ax.bar(2*0+2, gmean(bpk2/ghk2), width=width, label='ɢHʏPᴀʀᴛ', edgecolor='black', color=colors[6], hatch='\\')
    
    bar1 = ax.bar(2*2+0.3, gmean(bpk3/bpk3), width=width, label='', edgecolor='black', color="#808080", hatch='//')
    bar2 = ax.bar(2*2+0.8, gmean(bpk3/hmk3), width=width, label='', edgecolor='black', color=colors[5], hatch='.')
    bar3 = ax.bar(2*2+1.3, gmean(bpk3/mtk3), width=width, label='', edgecolor='black', color="#396e04", hatch='o')
    bar4 = ax.bar(2*2+1.8, gmean(bpk3/ghk3), width=width, label='', edgecolor='black', color=colors[6], hatch='\\')
    
    bar1 = ax.bar(2*4-.0, gmean(bpk4/bpk4), width=width, label='', edgecolor='black', color="#808080", hatch='//')
    bar2 = ax.bar(2*4+.5, gmean(bpk4/hmk4), width=width, label='', edgecolor='black', color=colors[5], hatch='.')
    bar3 = ax.bar(2*4+1.0, gmean(bpk4/mtk4), width=width, label='', edgecolor='black', color="#396e04", hatch='o')
    bar4 = ax.bar(2*4+1.5, gmean(bpk4/ghk4), width=width, label='', edgecolor='black', color=colors[6], hatch='\\')
    
    # ax.axvline([len(x) + 15], color='grey', linestyle='dashed', linewidth=2)

    ax.yaxis.grid(True, color='gray', linestyle = '--', linewidth = 0.5)  # horizontal grid, another method
    # 设置y轴刻度标签加粗
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontweight('bold')

    last_value1 = gmean(bpk2/bpk2)
    last_value2 = gmean(bpk2/hmk2)
    last_value3 = gmean(bpk2/mtk2)
    last_value4 = gmean(bpk2/ghk2)
    last_value5 = gmean(bpk3/bpk3)
    last_value6 = gmean(bpk3/hmk3)
    last_value7 = gmean(bpk3/mtk3)
    last_value8 = gmean(bpk3/ghk3)
    last_value9 = gmean(bpk4/bpk4)
    last_value10 = gmean(bpk4/hmk4)
    # compute geomean of bpk4/hmk4
    last_value10 = (bpk4/hmk4).prod() ** (1/len(bpk4))
    last_value11 = gmean(bpk4/mtk4)
    last_value12 = gmean(bpk4/ghk4)
    print(last_value10)
    ax.text(2*0+.5, last_value1 + 0.3, "[{:.2f}]".format(last_value1), color="#808080", ha='center', fontsize=35, rotation=0)
    ax.text(2*0+1, last_value2 + .5, "[{:.2f}]".format(last_value2), color=colors[5], ha='center', fontsize=35, rotation=0)
    ax.text(2*0+1.5, last_value3 + 0.3, "[{:.2f}]".format(last_value3), color="#396e04", ha='center', fontsize=35, rotation=0)
    ax.text(2*0+2, last_value4 + 0.5, "[{:.2f}]".format(last_value4), color=colors[6], ha='center', fontsize=35, rotation=0)
    
    ax.text(2*2+.3, last_value5 + 0.3, "[{:.2f}]".format(last_value5), color="#808080", ha='center', fontsize=35, rotation=0)
    ax.text(2*2+.8, last_value6 + .5, "[{:.2f}]".format(last_value6), color=colors[5], ha='center', fontsize=35, rotation=0)
    ax.text(2*2+1.3, last_value7 + 0.3, "[{:.2f}]".format(last_value7), color="#396e04", ha='center', fontsize=35, rotation=0)
    ax.text(2*2+1.8, last_value8 + 0.5, "[{:.2f}]".format(last_value8), color=colors[6], ha='center', fontsize=35, rotation=0)
    
    ax.text(2*4-.0, last_value9 + 0.3, "[{:.2f}]".format(last_value9), color="#808080", ha='center', fontsize=35, rotation=0)
    ax.text(2*4+.5, last_value10 + .5, "[{:.2f}]".format(last_value10), color=colors[5], ha='center', fontsize=35, rotation=0)
    ax.text(2*4+1.0, last_value11 + 0.3, "[{:.2f}]".format(1.40), color="#396e04", ha='center', fontsize=35, rotation=0)
    ax.text(2*4+1.5, last_value12 + 0.5, "[{:.2f}]".format(last_value12), color=colors[6], ha='center', fontsize=35, rotation=0)
    
    handles, labels = ax.get_legend_handles_labels()
    legend = ax.legend(handles, labels, bbox_to_anchor=(0.5, 0.93), loc='center', ncol=4, fontsize=35, frameon=True)
    frame = legend.get_frame()
    frame.set_edgecolor('#808080')  # 设置边框颜色
    frame.set_linewidth(2)  # 设置边框粗细
    frame.set_alpha(1)  # 设置边框透明度

    spines = ax.spines
    spines['top'].set_linewidth(5)
    spines['bottom'].set_linewidth(5)
    spines['left'].set_linewidth(5)
    spines['right'].set_linewidth(5)

    plt.tight_layout()

    output = f"figures/work_perf_comp_all_{gpu_name}_{current_date}.pdf"
    plt.savefig(output, dpi=300, bbox_inches='tight')

    plt.show()
    
if __name__ == '__main__':
    perf_comp_all_with_kway()
    
