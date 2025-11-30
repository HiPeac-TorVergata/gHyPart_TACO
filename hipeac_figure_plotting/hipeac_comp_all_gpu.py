import csv
import random
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import pandas as pd
# sys.path.append(".")
# import myplot
import matplotlib
import os
if os.environ.get('DISPLAY', '') == '':
    matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import FuncFormatter
from matplotlib.font_manager import FontProperties
import seaborn as sns
from scipy.stats import gmean
import datetime

current_date = datetime.date.today()

file1 = 'results/bipart_perf_xeon_4214R_24cores_t12.csv'






colors = ["#FF0000", "#00FF00", "#0000FF", "#FFFF00", 
          "#800080", "#ffc0cb", "#ffa500", "#808080", 
          "#007fff", "#8e52dc", "#ff7f50", "#ff4500", 
          "#396e04", "#ff00ff", "#ff69b4", "#489177",
          "#CC99FF", "#FFCC99", "#99CCFF", "#CCFF99",]


def compare_across_gpus(k):
    file1 = 'results/bipart_perf_xeon_4214R_24cores_t12.csv'
    file2 = '../results/hipeac_ghypart_NVIDIA_GeForce_GTX_1060_3GB_t8_2025-11-22.csv'
    file3 = '../results/hipeac_ghypart_NVIDIA_GeForce_RTX_2070_t12_2025-11-21.csv'
    file4 = '../results/hipeac_ghypart_QUADRO_RTX_5000_2025-11-22.csv'
    file5 = '../results/hipeac_ghypart_NVIDIA_GeForce_RTX_4090_t32_2025-09-14.csv'
    file6 = 'results/works_all_comp_3090_2024-01-08.csv'
    
    data1 = pd.read_csv(file1)
    data2 = pd.read_csv(file2)
    data3 = pd.read_csv(file3)
    data4 = pd.read_csv(file4)
    data5 = pd.read_csv(file5)
    data6 = pd.read_csv(file6)
    
    data1['BiPart'] = data1['k=2']
    data2['GTX-1060'] = data2['k=2']
    data3['RTX-2070'] = data3['k=2']
    data4['QUADRO-5000'] = data4['k=2']
    data5['RTX-4090'] = data5["k=2"]
    data6['RTX-3090'] = data6["gHyPart"]

    merged_data = pd.concat([data1['BiPart'],data2['GTX-1060'] ,data3['RTX-2070'], data4['QUADRO-5000'],data5['RTX-4090'],data6['RTX-3090']], axis=1)
    print(merged_data)
    
    #return
    merged_data['g_ratio'] = merged_data['BiPart'] / merged_data['RTX-4090'] # merged_data['best']
    merged_data = merged_data.sort_values(by='g_ratio')
    y1 = merged_data['BiPart']
    y6 = merged_data['GTX-1060']
    y2 = merged_data['RTX-2070']
    y3 = merged_data['QUADRO-5000']
    y4 = merged_data['RTX-4090']
    y5 = merged_data['RTX-3090']
    
    x = merged_data

    index_list = range(len(x))
    
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['legend.fontsize'] = 32
    plt.rcParams['legend.frameon'] =  True

    fig, ax = plt.subplots(figsize=(30, 11))
    ylim = 100
    #ax.set_yscale('log', base=10)  # 设置纵坐标为log2刻度
    #ax.set_ylim(0, ylim)  # 设置纵坐标起始值为0
    #ax.yaxis.set_ticks([0.1, 1, 10, 100, ylim])  # 设置刻度值
    # ax.yaxis.set_ticks([0.1, 1, 10, ylim])  # 设置刻度值
    ax.tick_params(axis='y', which='major', labelsize=48)
    ax.set_ylabel('Speedup over BiPart', va='center', fontsize=60, fontweight='bold', labelpad=45)  # 设置纵坐标title
    ax.set_xlabel('Hypergraphs', va='center', fontsize=60, fontweight='bold', labelpad=40)
    
    xlim = 600
    # xlim = 2.2
    # ax.set_xlim(0,len(x)+1)
    ax.set_xlim(0, xlim)
    app_ticks = [0, 100, 200, 300, 400, len(x)-1, 505]
    ax.xaxis.set_ticks([0, 100, 200, 300, 400, len(x)-1])
    ax.tick_params(axis='x', which='major', labelsize=50, rotation=0)
    
    dot1 = ax.plot(index_list, y1/y1, marker='+', linestyle='-', label='BiPart', color="#808080", linewidth=5)
    dot6 = ax.plot(index_list, y1/y6, marker='o', markersize=6, linestyle='-', label='GTX-1060', color=colors[16], linewidth=5)
    dot2 = ax.plot(index_list, y1/y2, marker='*', markersize=12, linestyle='-', label='RTX-2070', color="#396e04", linewidth=3)
    dot3 = ax.plot(index_list, y1/y3, marker='x', linestyle='-', label='QUADRO-5000', color=colors[6], linewidth=5)
    dot4 = ax.plot(index_list, y1/y4, marker='^', markersize=6, linestyle='-', label='RTX-4090', color=colors[8], linewidth=5)
    dot5 = ax.plot(index_list, y1/y5, marker='o', markersize=6, linestyle='-', label='RTX-3090', color=colors[0], linewidth=5)
    
    last_center = 550
    bar1_pos = last_center - 24
    bar2_pos = last_center - 12
    bar3_pos = last_center 
    bar4_pos = last_center + 12
    bar5_pos = last_center + 24
    bar6_pos = last_center + 36
    width = 8
    
    y2_clean = y2.where(y2 != 0.0, y1)
    y6_clean = y6.where(y6 != 0.0, y1)
    y3_clean = y3.where(y3 != 0.0, y1)
    y4_clean = y4.where(y4 != 0.0, y1)
    y5_clean = y5.where(y5 != 0.0, y1)
    
    print(gmean(y1/y2_clean),gmean(y1/y6_clean),gmean(y1/y3_clean),gmean(y1/y4_clean))
    
    bar1 = ax.bar(bar1_pos, gmean(y1/y1), width=width, label='', edgecolor='black', color="#808080", hatch='//')
    bar6 = ax.bar(bar2_pos, gmean(y1/y6_clean), width=width, label='', edgecolor='black', color=colors[16], hatch='.')
    bar2 = ax.bar(bar3_pos, gmean(y1/y2_clean), width=width, label='', edgecolor='black', color="#396e04", hatch='.')
    bar3 = ax.bar(bar4_pos, gmean(y1/y3_clean), width=width, label='', edgecolor='black', color=colors[6], hatch='\\')
    bar4 = ax.bar(bar5_pos, gmean(y1/y4), width=width, label='', edgecolor='black', color=colors[8], hatch='o')
    bar5 = ax.bar(bar6_pos, gmean(y1/y5_clean), width=width, label='', edgecolor='black', color=colors[0], hatch='x')

    ax.axvline([len(x) + 15], color='grey', linestyle='dashed', linewidth=2)

    ax.yaxis.grid(True, color='gray', linestyle = '--', linewidth = 0.5)  # horizontal grid, another method
    # 设置y轴刻度标签加粗
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontweight('bold')
        
    last_label_pos = 550
    xticks = list(ax.get_xticks()) + [last_label_pos]
    xticklabels = list(ax.get_xticklabels()) + ['GMean']
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)

    xtick_labels = ax.get_xticklabels()
    # 设置最后一个xtick label的大小、角度和颜色
    last_xtick_label = xtick_labels[-1]
    last_xtick_label.set_fontsize(50)
    last_xtick_label.set_rotation(0)

    last_value1 = gmean(y1/y1)
    last_value2 = gmean(y1/y6_clean)
    last_value3 = gmean(y1/y2_clean)
    last_value4 = gmean(y1/y3_clean)
    last_value5 = gmean(y1/y4_clean)
    last_value6 = gmean(y1/y5_clean)
    
    ax.text(bar1_pos+1, last_value1+0.5, "[{:.2f}]".format(last_value1), color="#808080", ha='center', fontsize=35, rotation=90)
    ax.text(bar2_pos+1, last_value2+5, "[{:.2f}]".format(last_value2), color=colors[16], ha='center', fontsize=35, rotation=90)
    ax.text(bar3_pos+1, last_value3+10, "[{:.2f}]".format(last_value3), color="#396e04", ha='center', fontsize=35, rotation=90)
    ax.text(bar4_pos+1, last_value4+10, "[{:.2f}]".format(last_value4), color=colors[6], ha='center', fontsize=35, rotation=90)
    ax.text(bar5_pos+1, last_value5+10, "[{:.2f}]".format(last_value5), color=colors[8], ha='center', fontsize=35, rotation=90)
    ax.text(bar6_pos+1, last_value5+10, "[{:.2f}]".format(last_value6), color=colors[0], ha='center', fontsize=35, rotation=90)
    
    handles, labels = ax.get_legend_handles_labels()
    legend = ax.legend(handles, labels, bbox_to_anchor=(0.45, 0.88), loc='center', ncol=3, fontsize=35, frameon=True)
    frame = legend.get_frame()
    frame.set_edgecolor('#808080')  # 设置边框颜色
    frame.set_linewidth(2)  # 设置边框粗细
    frame.set_alpha(1)  # 设置边框透明度

    spines = ax.spines
    spines['top'].set_linewidth(5)
    spines['bottom'].set_linewidth(5)
    spines['left'].set_linewidth(5)
    spines['right'].set_linewidth(5)

    # 调整布局以适应标签
    plt.tight_layout()

    output = f'results/hipeac_figure_new_compare_gpus.pdf'

    # 保存图片
    plt.savefig(output, dpi=300, bbox_inches='tight')

    # 显示图形  
    plt.show()



if __name__ == '__main__':
    compare_across_gpus(2)
    
