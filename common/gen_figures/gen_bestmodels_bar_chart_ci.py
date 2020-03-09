import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
import math
from matplotlib import rc

rc('text', usetex=False)

csv_fullname = './results/exp_sl_ssl_bestmodels_chart_most_ci_result.csv'
plt.figure(num=None, dpi=600, facecolor='w', edgecolor='k', figsize=(4.5, 3))

method_name_map = {'sl': 'SL$^+$', 'ict': 'ICT', 'pimodel': '$\Pi$ model', 'mixmatch': 'MixMatch', 'semixup': 'Semixup'}
method_list = ['SL$^+$', '$\Pi$ model', 'ICT', 'MixMatch', 'Semixup']
x_cols = np.arange(0, 4)
print(x_cols)
width = 0.1
gap = 0.03
mid_offset = 2 * (gap + width)
method_2_idx = {'SL$^+$': 0, '$\Pi$ model': 1, 'ICT': 2, 'MixMatch': 3, 'Semixup': 4}
plt.rcParams.update({'font.size': 8})

method_x = []
method_2_coord = {}

for i_mt, method_name in enumerate(method_name_map):
    method_name = method_name_map[method_name]
    acc_list = []
    bounds = []
    for n_l in [50, 100, 500, 1000]:
        df = pd.read_csv(csv_fullname, sep=',')
        row_list = []

        for i, row in df.iterrows():
            if not row['method_name']:  # or math.isnan(row['method_name']):
                if row['device']:
                    m = re.match(r'.+bullx(.+)$', row['device'])
                    if m is not None:
                        if m.group(1) in method_name_map:
                            row['method_name'] = method_name_map[m.group(1)]
                            row_list.append(row)
                        else:
                            raise ValueError(f'Cannot find {m.group(1)} in {method_name_map}')
                    else:
                        raise ValueError('Cannot find method_name in {}'.format(row['device']))
            else:
                row['method_name'] = method_name_map[row['method_name']]
                row_list.append(row)

        df_update = pd.DataFrame(row_list)

        for i, row in df_update.iterrows():

            found = False
            if row['n_labels'] == n_l and row['method_name'] == method_name:
                acc_list.append(row['acc_test'])
                ub = row['acc_ci_ub'] - row['acc_test']
                lb = - row['acc_ci_lb'] + row['acc_test']
                bounds.append([lb, ub])
                found = True
                break

        if not found:
            raise ValueError(f"Not found {row['n_labels']} {row['method_name']}")

    bounds = np.transpose(np.array(bounds))
    _x = x_cols - mid_offset + method_2_idx[method_name] * (width + gap)
    plt.bar(_x, acc_list, width, label=method_name, yerr=bounds, linewidth=0.005, capsize=1.5)
    method_2_coord[method_name] = {'x': _x, 'y': np.array(acc_list), 'ub': np.array(bounds)[1, :]}

    if i_mt == 5:
        break

# Manually input based on trained result of SL on full OAI setting
acc_fulloai = 0.709
plt.plot([-0.4, 3.4], [acc_fulloai, acc_fulloai], '--k', linewidth=1, label='SL (full OAI)')

plt.ylim([0.42, .73])
plt.legend(loc="upper left", ncol=1, bbox_to_anchor=(0.01, 0.98))
plt.xlabel('Number of labeled data per KL grade')
plt.ylabel('Balanced multi-class accuracy')
x_ticks = [50, 100, 500, 1000]
plt.xticks(x_cols, x_ticks)


def annotate_method_pair(m1, m2, ind_matches, texts):
    for i, inds in enumerate(ind_matches):
        x = (m1['x'][inds[0]] + m2['x'][inds[1]]) / 2.0
        y = max(m1['y'][inds[0]], m2['y'][inds[1]])
        plt.annotate(texts[i], xy=(x, y), zorder=10, rotation=-90)


def mark_significance(m, inds, texts):
    print(m)
    for i, ind in enumerate(inds):
        x = m['x'][ind] - 1 / 3.0 * width
        y = m['y'][ind] + m['ub'][ind] + 0.01
        plt.annotate(texts[i], xy=(x, y), rotation=90)


# Manually input base on output of common/significance_tests/wilcoxon_test.py
mark_significance(method_2_coord['MixMatch'], [0, 1, 2, 3], ["", "**", "", "*"])
mark_significance(method_2_coord['ICT'], [0, 1, 2, 3], ["", "**", "**", "**"])
mark_significance(method_2_coord['$\Pi$ model'], [0, 1, 2, 3], ["", "*", "", "*"])
mark_significance(method_2_coord['SL$^+$'], [0, 1, 2, 3], ["", "**", "**", "*"])

plt.grid(True, axis='y', linestyle='-', linewidth=0.3, zorder=100)

output_plot_filename = "./comparison_sl_ssl_ci.pdf"
plt.savefig(output_plot_filename, format="pdf")
