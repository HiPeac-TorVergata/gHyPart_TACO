import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import matplotlib
import os
if os.environ.get('DISPLAY', '') == '':
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy.stats import gmean
import datetime

current_date = datetime.date.today()

file = 'results/benchmarks_all_patterns_real_hypergraphs_4060Ti_2024-01-08.csv'
file1= '../scripts/hipeac_ghypart_B_NVIDIA_GeForce_GTX_1060_3GB_t8_2025-11-25.csv'
file2 = 'results/bipart_perf_xeon_4214R_24cores_t12.csv'
file3 = '../results/hipeac_ghypart_NVIDIA_GeForce_GTX_1060_3GB_t8_2025-11-22.csv'
file4 = '../results/benchmarks_all_patterns_real_hypergraphs_4090_2025-11-25.csv'


data1 = pd.read_csv(file1)
data2 = pd.read_csv(file2)
data3 = pd.read_csv(file3)
data4 = pd.read_csv(file4)
data2["BiPart"] = data2["k=2"]
data1["gHyPart-B"] = data1["k=2"]
data3["gHyPart"] = data3["k=2"]
print(data1)

data = pd.concat([data2["BiPart"],data1["gHyPart-B"],data3["gHyPart"]],axis=1)
print(data)
data['ratio'] = data["BiPart"] / data["gHyPart"]
sorted_data = data.sort_values(by='ratio')
 
sorted_data.reset_index(drop=True, inplace=True)
data = sorted_data
x = data.iloc[:,0]
index_list = range(len(x))

plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
# plt.rcParams['legend.fontsize'] =  30
plt.rcParams['legend.frameon'] =  True

colors = ["#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#800080", "#ffc0cb", "#ffa500", "#808080", "#007fff", "#8e52dc"]

fig = plt.figure(figsize=(25, 9))
# spec = fig.add_gridspec(1, 2, width_ratios=[1.0, 1])
# for 4060Ti
spec = fig.add_gridspec(1, 2, width_ratios=[2.2, 1])
ax = fig.add_subplot(spec[0, 0])
ax1 = fig.add_subplot(spec[0, 1])

def myplot(data, markers, labels, color, markersize = 5):
    n = len(data)
    counts = []
    for k in range(n+1):
        count = np.sum(data >= k/n)
        counts.append(count)
    print(counts)
    print(len(counts))
    cdf = np.asarray(counts) / n
    cdf = cdf[::-1]  # 倒转CDF的顺序
    print(cdf)
    percentiles = np.linspace(0, 100, n+1)
    percentiles = percentiles[::-1]  # 倒转百分位数的顺序
    print(percentiles)
    # print(np.sum(data['ratio'] <= 0.99999999))
    ax.plot(percentiles, cdf, marker=markers, markersize = markersize, linestyle='-', label=labels, color=color, linewidth=2)

cmap = plt.cm.Blues
Colors = cmap(np.linspace(0.5, 1, 6))

data['speedupBiPart'] = data['BiPart'] / data['gHyPart']
sorted_data = data.sort_values(by='speedupBiPart')
sorted_data.reset_index(drop=True, inplace=True)
y1 = sorted_data['BiPart'] / sorted_data['BiPart']
y2 = sorted_data['BiPart'] / sorted_data['gHyPart-B']
y3 = sorted_data['BiPart'] / sorted_data['gHyPart']

ylim = 1e3
ax.set_yscale('log', base=10)  # 设置纵坐标为log2刻度
ax.set_ylim(0, ylim)  # 设置纵坐标起始值为0
ax.yaxis.set_ticks([0.1, 1, 10, 100, 1000]) #, fontsize=40)  # 设置刻度值
ax.tick_params(axis='y', which='major', labelsize=40)
ax.set_ylabel('Speedup over BiPart', va='center', fontsize=45, fontweight='bold', labelpad=45)  # 设置纵坐标title
ax.set_xlabel('Hypergraphs', va='center', fontsize=45, fontweight='bold', labelpad=40)

print(len(x))
xlim = 730
ax.set_xlim(0, xlim)
ax.xaxis.set_ticks([0, 250, len(x)-1]) #, fontsize=40)
ax.set_xticks([0, 250, len(x)-1])
ax.tick_params(axis='x', which='major', labelsize=40, rotation=0)
dot1 = ax.plot(index_list, y1, marker='+', linestyle='-', label='BiPart', color="#808080", linewidth=2)
dot2 = ax.plot(index_list, y2, marker='x', linestyle='-', label='ɢHʏPᴀʀᴛ-B', color=colors[6], linewidth=2)
dot3 = ax.plot(index_list, y3, marker='^', linestyle='-', label='ɢHʏPᴀʀᴛ', color=colors[8], linewidth=2)

ax.axvline([len(x) + 50], color='grey', linestyle='dashed', linewidth=2)

last_center = 590
bar1_pos = last_center + 5
bar2_pos = last_center + 50
bar3_pos = last_center + 95
width = 40
y2_clean = sorted_data['BiPart'] / sorted_data['gHyPart-B'].where(sorted_data['gHyPart-B']!=0.0,sorted_data['BiPart'])
y3_clean = sorted_data['BiPart'] / sorted_data['gHyPart'].where(sorted_data['gHyPart']!=0.0,sorted_data['BiPart'])

bar1 = ax.bar(bar1_pos, gmean(y1), width=width, label='', edgecolor='black', color="#808080", hatch='//')
bar2 = ax.bar(bar2_pos, gmean(y2_clean), width=width, label='', edgecolor='black', color=colors[6], hatch='\\')
bar3 = ax.bar(bar3_pos, gmean(y3_clean), width=width, label='', edgecolor='black', color=colors[8], hatch='o')

ax.yaxis.grid(True, color='gray', linestyle = '--', linewidth = 0.5)  # horizontal grid, another method
# 设置y轴刻度标签加粗
for tick in ax.yaxis.get_major_ticks():
    tick.label1.set_fontweight('bold')
    
last_label_pos = 640
xticks = list(ax.get_xticks()) + [last_label_pos]
# xticks = []#[0, 250, len(x)-1, last_label_pos]
print(xticks)
# xticklabels = list(ax.get_xticklabels()) + ['GMean']
xticklabels = list(ax.get_xticks()) + ['GMean']
print(xticklabels)
# ax.set_xticks(xticks)
ax.xaxis.set_ticks(xticks) #, fontsize=40)
ax.set_xticklabels(xticklabels)

xtick_labels = ax.get_xticklabels()
# 设置最后一个xtick label的大小、角度和颜色
last_xtick_label = xtick_labels[-1]
last_xtick_label.set_fontsize(40)
last_xtick_label.set_rotation(0)

ax.text(bar1_pos, gmean(y1) + 0.5, "[{:.2f}]".format(gmean(y1)), color="#808080", ha='center', fontsize=30, rotation=90)
ax.text(bar2_pos+0.5, gmean(y2_clean) + 5, "[{:.2f}]".format(gmean(y2_clean)), color=colors[6], ha='center', fontsize=30, rotation=90)
ax.text(bar3_pos, gmean(y3_clean) + 5, "[{:.2f}]".format(gmean(y3_clean)), color=colors[8], ha='center', fontsize=30, rotation=90)

width = 1

labels = ['Ported', 'Re-trained']
myticks = [-0.1, 1.5]

ax1.set_yscale('linear')
ax1.set_ylim(0, 1.01)
ax1.tick_params(axis='y', which='major', labelsize=32)
ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, _: '{:.0%}'.format(x)))
ax1.set_ylabel('Performance Normalized \nto ɢHʏPᴀʀᴛ-O', va='center', fontsize=40, fontweight='bold', labelpad=30)  # 设置纵坐标title
ax1.set_xlabel('', va='center', fontsize=30, fontweight='bold', labelpad=100)
ax1.set_xlim(-1.0 * width + .0, len(labels) + 0.4)
ax1.set_xticks(ticks=[i for i in range(len(labels))], rotation=0, fontsize=38, labels=labels, fontweight='bold')

ticks = []

ax1.set_xticks(myticks)  # 设置X轴刻度位置为0和100
ax1.set_xticklabels(labels)


print(data4)
data['3090'] = data["gHyPart"] / data4['gHyPart']
norm_3090 = data['3090'].tolist()
norm_3090 = [1 if x > 1.0 else x for x in norm_3090]

ax1.bar(-0.1, gmean(norm_3090), width=width, label='4090', edgecolor='black', color=Colors[2], hatch='')
ax1.bar(1.5, gmean(data["gHyPart"]), width=width, label='1060', edgecolor='black', color=Colors[4], hatch='')


ax1.text(-0.1, gmean(norm_3090) + 0.01, "{:.1%}".format(gmean(norm_3090)), color='black', ha='center', fontsize=36, rotation=0)
ax1.text(1.5, gmean(data["gHyPart"]) + 0.01, "{:.1%}".format(gmean(data["gHyPart"])), color='black', ha='center', fontsize=36, rotation=0)


ax.yaxis.grid(True, color='gray', linestyle = '--', linewidth = 0.5)  # horizontal grid, another method
# 设置y轴刻度标签加粗
for tick in ax.yaxis.get_major_ticks():
    tick.label1.set_fontweight('bold')
    
ax1.yaxis.grid(True, color='gray', linestyle = '--', linewidth = 0.5)  # horizontal grid, another method
# 设置y轴刻度标签加粗
for tick in ax1.yaxis.get_major_ticks():
    tick.label1.set_fontweight('bold')
    

handles, labels = ax.get_legend_handles_labels()
legend = ax.legend(handles, labels, bbox_to_anchor=(0.45, 0.92), loc='center', ncol=3, fontsize=30, frameon=True)
frame = legend.get_frame()
frame.set_edgecolor('#808080')  # 设置边框颜色
frame.set_linewidth(1)  # 设置边框粗细
frame.set_alpha(1)  # 设置边框透明度
frame = legend.get_frame()
frame.set_edgecolor('#808080')  # 设置边框颜色
frame.set_linewidth(2)  # 设置边框粗细
frame.set_alpha(1)  # 设置边框透明度

spines = ax.spines
spines['top'].set_linewidth(5)
spines['bottom'].set_linewidth(5)
spines['left'].set_linewidth(5)
spines['right'].set_linewidth(5)  

spines1 = ax1.spines
spines1['top'].set_linewidth(5)
spines1['bottom'].set_linewidth(5)
spines1['left'].set_linewidth(5)
spines1['right'].set_linewidth(5)  

# 调整布局以适应标签
plt.tight_layout()
plt.subplots_adjust(wspace=0.35) 

output = f'results/benchmarks_all_patterns_allversions_4060Ti_{current_date}.pdf'
output = f'results/benchmarks_all_patterns_allversions_1060.pdf'
# 保存图片
plt.savefig(output, dpi=300, bbox_inches='tight')

# 显示图形  
plt.show()
