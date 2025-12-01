import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import matplotlib
import os
if os.environ.get('DISPLAY', '') == '':
    matplotlib.use('Agg')
import seaborn as sns
from scipy.stats import gmean
import datetime
import seaborn as sns

current_date = datetime.date.today()

file = '../figure_plotting/results/benchmarks_all_patterns_top100_3090_2024-01-08.csv'

data = pd.read_csv(file)
sorted_data = data.iloc[:].sort_values(by=data.columns[31])
data['ratio'] = data.iloc[:,30][:] / data.iloc[:,29][:]
sorted_data = pd.concat([sorted_data, data.iloc[:]])
sorted_data = sorted_data.head(100)
sorted_data.reset_index(drop=True, inplace=True)
data = sorted_data
x = data.iloc[:,0]

y1 = data.iloc[:,31]
y2 = data.iloc[:,32]


index_list = range(len(x))
def find_next_power(value, base):
    power = 1
    while power < value:
        power *= base
    return power

plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['legend.fontsize'] =  30
plt.rcParams['legend.frameon'] =  True

colors = ["#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#800080", "#ffc0cb", "#ffa500", "#808080", "#007fff", "#8e52dc"]
sns.set_palette(colors)

fig, ax = plt.subplots(figsize=(30, 11))

max_value = max(y1.max(), y2.max())  
min_value = min(y1.min(), y2.min()) 
ylim = find_next_power(max_value, 10)

second_min_value = y1.nsmallest(2).iloc[-1]

ax.set_yscale('log', base=10)  
ax.set_ylim(0, ylim) 
ax.tick_params(axis='y', which='major', labelsize=50)
ax.set_ylabel('Speedup over ɢHʏPᴀʀᴛ-B', va='center', fontsize=48, fontweight='bold', labelpad=50)  
ax.set_xlabel('Hypergraphs', va='center', fontsize=50, fontweight='bold', labelpad=40)


xlim = 550
powerlaw = 75
topk = 100
filter = len(x)-1 
ax.set_xlim(0,len(x)+12)
ax.xaxis.set_ticks([0, 20, 40, 60, 80, topk-1])
ax.tick_params(axis='x', which='major', labelsize=50, rotation=0)

ax.set_xticks([0.7, 25.7, 50.7, 75.7, 99.7])  
ax.set_xticklabels(['0', '25', '50', '75', '99'])
width = .8
mycolors = ['#014F9C', '#3C79B4', '#78A3CC', '#B3CDE4', '#EEF7FC']
bar1 = ax.bar(x=[i-width+1.5 for i in range(len(index_list[:-1]))], height=y1[:-1], bottom=0, color=mycolors[0], edgecolor='black', alpha=1, width=width, hatch='', label='')

area = np.trapz(y1[:] / y2[:], index_list[:])


ax.bar(len(x) + 6.5, gmean(y1), width=width, label='', edgecolor='black', color=mycolors[0], hatch='')

ax.axvline([len(x) + 1], color='grey', linestyle='dashed', linewidth=2)

ax.yaxis.grid(True, color='gray', linestyle = '--', linewidth = 0.5) 
for tick in ax.yaxis.get_major_ticks():
    tick.label1.set_fontweight('bold')
    
last_label_pos = len(x)+7
xticks = list(ax.get_xticks()) + [last_label_pos]
xticklabels = list(ax.get_xticklabels()) + ['GMean']
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels)

xtick_labels = ax.get_xticklabels()
last_xtick_label = xtick_labels[-1]
last_xtick_label.set_fontsize(50)  
last_xtick_label.set_rotation(0)  

last_value1 = y1.iloc[-1]

ax.text(len(x) + 6.5, gmean(y1) + 1, "{:.2f}".format(gmean(y1)), color='black', ha='center',rotation=0,fontsize=40)

ax.text(len(x)-3, max(y1[:])+50, "{:.2f}".format(max(y1[:])), color='black', ha='center', rotation=0, fontsize=40)

print(last_value1)

spines = ax.spines
spines['top'].set_linewidth(5)
spines['bottom'].set_linewidth(5)
spines['left'].set_linewidth(5)
spines['right'].set_linewidth(5)  

plt.tight_layout()

output = f'figures/benchmarks_all_patterns_top100_3090_{current_date}.pdf'

plt.savefig(output, dpi=300, bbox_inches='tight') 
plt.show()
