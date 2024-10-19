import numpy as np
import matplotlib.pyplot as plt
import os
import re
import pandas as pd
import os

fig, ax = plt.subplots()
plt.tick_params(labelsize=12)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
plt.rcParams["text.usetex"] = True
plt.rcParams['text.latex.preamble'] = r"\usepackage{times}"
font_legend = {'family': 'Times New Roman', 'size': 10}
font_label = {'family': 'Times New Roman', 'size': 14}
plt.rcParams["font.family"] = "Times New Roman"




def files_detail(directory):
    csv_count = 0
    flag = 0
    for item in os.listdir(directory):
        item = os.path.join(directory, item)
        if os.path.isfile(item):
            number = extract_numbers(item)
            if number == 0:
                flag = 1
            if item.endswith('.csv'):
                csv_count += 1
    prob_list = []
    for item in os.listdir(directory):
        item = os.path.join(directory, item)
        if os.path.isfile(item):
            csv = pd.read_csv(item)
            prob_list = csv.iloc[:, 0]
            break
    return csv_count, prob_list, flag

def extract_numbers(filename):
    pattern = re.compile(r'N_r_(\d+)')
    match = pattern.search(filename)
    if match:
        number = int(match.group(1))
    else:
        number = -1
    return number

def pattern_detect(directory):
    pattern_mm = re.compile(r'_mm$')
    match_mm = pattern_mm.search(directory)
    pattern_loc = re.compile(r'_loc$')
    match_loc = pattern_loc.search(directory)
    if match_mm:
        mm = 1
    else:
        mm = 0
    if match_loc:
        loc = 1
    else:
        loc = 0
    return mm, loc


def render(directory):
    mm, loc = pattern_detect(directory)
    baseline_label = r"null"
    enhanced_baseline_label = baseline_label
    if mm:
        baseline_label = r"Baseline 1"
        enhanced_baseline_label = r"Enhanced Baseline 1 when $N_\mathrm{r}$ = "
    if loc:
        baseline_label = r"Baseline 2 when $N_\mathrm{r}$ = 1"
        enhanced_baseline_label = r"Enhanced Baseline 2 when $N_\mathrm{r}$ = "
    DMBT_label = r"DMBT Method when $N_\mathrm{r}$ = "
    barWidth = 0.07
    capsize = 3
    elinewidth = 0.8
    csv_count, prob_list, flag = files_detail(directory)
    bars = np.zeros([len(prob_list), csv_count*2-1 if flag else csv_count*2])
    stds = np.zeros([len(prob_list), csv_count*2-1 if flag else csv_count*2])
    number_list = []
    idx = 0
    plt.grid(alpha=0.4,linestyle='--')
    for item in os.listdir(directory):
        item = os.path.join(directory, item)
        if item.endswith('.csv'):
            if os.path.isfile(item):
                number = extract_numbers(item)
                csv = pd.read_csv(item)
                if number == 0:
                    csv_asrp = csv.loc[:, ["S3 power error","S4 power error","S5 power error","S8 power error"]]
                    stds[:, idx] = csv_asrp.std(axis=1)
                    bars[:, idx] = csv_asrp.mean(axis=1)
                    idx = idx + 1
                else:
                    number_list.append(number)
                    csv_asrp = csv.loc[:, ["S3 power error", "S4 power error", "S5 power error", "S8 power error"]]
                    stds[:, idx] = csv_asrp.std(axis=1)
                    bars[:, idx] = csv_asrp.mean(axis=1)
                    idx = idx + 1
                    csv_asrp = csv.loc[:, ["S3 power correct", "S4 power correct", "S5 power correct", "S8 power correct"]]
                    stds[:, idx] = csv_asrp.std(axis=1)
                    bars[:, idx] = csv_asrp.mean(axis=1)
                    idx = idx + 1

    baseline_colors = ['white']
    light_colors = [
        'lightblue', 'lightgreen', 'lightcoral', 'lightgoldenrodyellow',
        'lightpink', 'lightsteelblue', 'lightskyblue', 'lightyellow',
        'palegreen', 'paleturquoise', 'palevioletred']
    dark_colors = [
        'blue', 'green', 'maroon', 'goldenrod',
        'deeppink', 'darkblue', 'navy', 'darkgoldenrod',
        'darkgreen', 'darkslateblue', 'purple']
    i = 0
    k = 0
    for j in range(csv_count):
        if flag:
            bar_i = bars[:, i]
            std_i = stds[:, i]
            r_i = np.arange(len(bar_i)) + barWidth * i
            plt.bar(r_i, bar_i, width=barWidth, color=baseline_colors[0], edgecolor='black',
                    yerr=std_i, capsize=capsize, error_kw={'linewidth': elinewidth}, label=baseline_label)
            i = i + 1
            flag = 0
        else:
            bar_i = bars[:, i]
            std_i = stds[:, i]
            r_i = np.arange(len(bar_i)) + barWidth * i
            plt.bar(r_i, bar_i, width=barWidth, color=light_colors[k], edgecolor='black',
                    yerr=std_i, capsize=capsize, error_kw={'linewidth': elinewidth}, label=enhanced_baseline_label+str(number_list[k]))
            i = i + 1
            bar_i = bars[:, i]
            std_i = stds[:, i]
            r_i = np.arange(len(bar_i)) + barWidth * i
            plt.bar(r_i, bar_i, width=barWidth, color=dark_colors[k], edgecolor='black',
                    yerr=std_i, capsize=capsize, error_kw={'linewidth': elinewidth}, label=DMBT_label+str(number_list[k]))
            i = i + 1
            k = k + 1

    plt.xticks([r + barWidth * bars.shape[0] for r in range(bars.shape[0])], prob_list)
    plt.ylabel('Average Normalized Received Beam Power', font_label)
    plt.xlabel("Noise Probability", font_label)
    plt.legend()
    ax.spines['right'].set_visible(True)
    ax.spines['top'].set_visible(True)
    plt.savefig(os.path.join(directory, 'results.jpg'), dpi=800)  # 保存为图片jpg格式
