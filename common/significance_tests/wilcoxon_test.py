import os
import re
import torch
import numpy as np
import pandas as pd
import argparse
from scipy.stats import wilcoxon


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_meta_file', type=str,
                        default='../results/exp_bestmodels_wilcoxon_test_result.csv',
                        help='Path of experimental meta file')
    parser.add_argument('--seed', type=int, default=12345, help='Random seed')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    return args


# Create a dict to map method maps to
def extract_data_for_wilcoxon_test(df, method_names, method_to_nlabel, metric_name='acc_test', n_folds=20):
    metric_by_method = dict()

    for method_name in method_names:
        print(f'> Compare to {method_name}')
        n_labels = method_to_nlabel[method_name]

        df_method = df[df['method_name'] == method_name]
        metric_by_method[method_name] = []

        for i_fold in range(n_folds):
            row = df_method[(df_method['n_labels'] == n_labels) & (df_method['fold_test'] == i_fold)]

            if len(row) == 1:
                row = row.iloc[0]
                metric_by_method[method_name].append(row[metric_name])
            else:
                raise ValueError(f'Not found exactly 1 row, but {len(row)}')
    return metric_by_method


def fill_methods_from_device(df):
    out_rows = []
    for ind, row in df.iterrows():
        if not row['method_name'] or np.isnan(row['method_name']):
            if row['device']:
                m = re.match(r".+bullx(\w+)$", row['device'])
                if m is None:
                    raise ValueError('Not found "bullx" in device.')
                else:
                    row['method_name'] = m.group(1)
            else:
                raise ValueError('Not found contents of device and method_name.')
        out_rows.append(row)

    return pd.DataFrame(out_rows)


if __name__ == '__main__':
    args = init_args()
    n_folds = 20
    input_meta_results_fullname = args.exp_meta_file
    output_meta_results_fullname = input_meta_results_fullname[:-4] + "_stats.csv"

    df = pd.read_csv(input_meta_results_fullname, sep=',')
    df = fill_methods_from_device(df)

    method_names = df['method_name'].unique().tolist()

    labels_list = [50, 100, 500, 1000]
    metric_name = 'acc_test'
    main_method = 'semixup'

    opponent_methods = set(method_names) - set([main_method])

    rows = []

    test_mode = 'greater'  # 'two-sided' # 'greater'

    print('Start Wilcoxon test...')
    method_to_nlabels = {}

    for n_label in labels_list:
        for method_name in method_names:
            method_to_nlabels[method_name] = n_label
        if "slfulloai" in method_to_nlabels:
            method_to_nlabels["slfulloai"] = 35000

        metrics_by_names = extract_data_for_wilcoxon_test(df, method_names, method_to_nlabels, metric_name,
                                                          n_folds=n_folds)
        if test_mode == 'greater':
            print('One-sided test results:')
        elif test_mode == 'two-sided':
            print('Two-sided test results:')
        else:
            raise ValueError(f'Not support mode {test_mode}')

        for op_method in opponent_methods:
            if op_method == "slfulloai":
                main_metrics = metrics_by_names[op_method]
                op_metrics = metrics_by_names[main_method]
            else:
                main_metrics = metrics_by_names[main_method]
                op_metrics = metrics_by_names[op_method]
            w, p = wilcoxon(main_metrics, op_metrics, zero_method='wilcox',
                            alternative=test_mode)
            print(f'[{op_method}][{n_label}] w = {w}, p-value = {p}')
            row = dict()
            row['n_labels'] = n_label
            row['main_method'] = main_method
            row['op_method'] = op_method
            row['w'] = w
            row['p_value'] = p
            rows.append(row)

    method_to_nlabels['semixup'] = 500
    method_to_nlabels['sl'] = 1000
    method_to_nlabels['pimodel'] = 1000
    method_to_nlabels['mixmatch'] = 1000
    method_to_nlabels['ict'] = 1000
    method_to_nlabels['slfulloai'] = 35000
    metrics_by_names = extract_data_for_wilcoxon_test(df, method_names, method_to_nlabels, metric_name, n_folds=n_folds)
    for op_method in opponent_methods:
        if len(metrics_by_names[main_method]) != len(metrics_by_names[op_method]):
            print(f'WARN: Method {main_method} and {op_method} have different sizes. Skip!')
            continue
        w, p = wilcoxon(metrics_by_names[main_method], metrics_by_names[op_method], zero_method='wilcox',
                        alternative=test_mode)
        print(f'[{op_method}][{method_to_nlabels[op_method]}] w = {w}, p-value = {p}')
        row = dict()
        row['n_labels'] = -1
        row['main_method'] = main_method
        row['op_method'] = op_method
        row['w'] = w
        row['p_value'] = p
        rows.append(row)

    saved_df = pd.DataFrame(rows)
    saved_df.to_csv(output_meta_results_fullname, sep='|', index=None)
