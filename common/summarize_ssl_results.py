import pandas as pd
import numpy as np

if __name__ == "__main__":
    input_csv = "./results/exp_sl_ssl_methods_result.csv"
    df = pd.read_csv(input_csv, sep=",")

    nlabels_list = [50, 100, 500, 1000]  # 2000, 3000
    map_max_nu_muls = {50: 6, 100: 6, 500: 6, 1000: 6, 2000: 3, 3000: 2}
    nu_muls = [1, 2, 3, 4, 5, 6]
    method_names = ['ict', 'pimodel', 'mixmatch', 'semixup']

    detailed_table = "\\toprule \n \
        \\multirow{2}{*}{\\textbf{Method}} & \\textbf{\\# unlabeled} & \\multicolumn{4}{c}{\\textbf{\\# labels / KL grade ($N / 5$)}} \\\\ \n \
        \\cmidrule{3-6}\n \
         & \\textbf{data} & \\textbf{50} & \\textbf{100} & \\textbf{500} & \\textbf{1000} \\\\ \n \
        \\midrule \\midrule \n "

    map_method_names = {'ict': 'ICT~\cite{verma2019interpolation}',
                        'pimodel': '$\Pi$ model~\cite{laine2016temporal}',
                        'mixmatch': 'MixMatch~\cite{berthelot2019mixmatch}',
                        'semixup': '\\textit{Semixup} (Ours)'}

    summary_table = ""

    metric = 'acc'
    reduce = 'mean'

    df_ext = []
    df_ext_accmax = []
    # Get results
    for n_labels in nlabels_list:
        max_nu_mul = map_max_nu_muls[n_labels] + 1
        for method_name in method_names:
            df_multi_runs = df[(df['n_labels'] == n_labels) &
                               (df['method_name'] == method_name) &
                               (df['seed'] != 54321)]
            accs = df_multi_runs['acc_test'].to_numpy()
            kappas = df_multi_runs['kappa_test'].to_numpy()

            row_accmax = {'n_labels': n_labels,
                          'method_name': method_name,
                          'method_name_pretty': map_method_names[method_name],
                          'accs': accs.tolist(),
                          'kappas': kappas.tolist(),
                          'acc_mean': accs.mean(),
                          'acc_std': accs.std(),
                          'acc_max': accs.max(),
                          'kappa_mean': kappas.mean(),
                          'kappa_std': kappas.std(),
                          'kappa_max': kappas.max(),
                          'target_max_all': ''}

            df_ext_accmax.append(row_accmax)

            for nu_mul in range(1, max_nu_mul):
                n_unlabels = n_labels * nu_mul

                df_multi_runs = df[(df['n_labels'] == n_labels) & (df['n_unlabels'] == n_unlabels)
                                   & (df['method_name'] == method_name) & (df['seed'] != 54321)]
                accs = df_multi_runs['acc_test'].to_numpy()
                kappas = df_multi_runs['kappa_test'].to_numpy()

                row = {'nu_mul': nu_mul,
                       'n_labels': n_labels,
                       'n_unlabels': n_unlabels,
                       'method_name': method_name,
                       'method_name_pretty': map_method_names[method_name],
                       'accs': accs.tolist(),
                       'kappas': kappas.tolist(),
                       'acc_mean': accs.mean(),
                       'acc_max': accs.max(),
                       'acc_std': accs.std(),
                       'kappa_mean': kappas.mean(),
                       'kappa_max': kappas.max(),
                       'kappa_std': kappas.std(),
                       'comb_key': f"{nu_mul}.{method_name}.{n_labels}",
                       'acc_mean_per_nu': '',
                       'acc_std_per_nu': '',
                       'acc_max_per_nu': '',
                       'kappa_mean_per_nu': '',
                       'kappa_std_per_nu': '',
                       'kappa_max_per_nu': '',
                       'target_metric_per_nu': ''}
                df_ext.append(row)

    df_ext = pd.DataFrame(df_ext)
    df_ext_accmax = pd.DataFrame(df_ext_accmax)

    # Format results
    for n_labels in nlabels_list:
        max_nu_mul = map_max_nu_muls[n_labels] + 1
        for nu_mul in range(1, max_nu_mul):
            n_unlabels = n_labels * nu_mul

            row_idx_sel = df_ext.index[(df_ext['n_labels'] == n_labels) & (df_ext['n_unlabels'] == n_unlabels)]
            _sorted_idx = np.argsort(df_ext.loc[row_idx_sel, f'{metric}_{reduce}'].to_numpy())
            bests_metric_ind = []
            bests_metric_ind.append(_sorted_idx[-1])
            bests_metric_ind.append(_sorted_idx[-2])
            for ind, row_ind_sel in enumerate(row_idx_sel):
                if metric == 'acc':
                    _mean = 100 * df_ext.loc[row_ind_sel, 'acc_mean'].item()
                    _std = 100 * df_ext.loc[row_ind_sel, 'acc_std'].item()
                    _max = 100 * df_ext.loc[row_ind_sel, 'acc_max'].item()
                elif metric == 'kappa':
                    _mean = 100 * df_ext.loc[row_ind_sel, 'kappa_mean'].item()
                    _std = 100 * df_ext.loc[row_ind_sel, 'kappa_std'].item()
                    _max = 100 * df_ext.loc[row_ind_sel, 'kappa_max'].item()
                else:
                    raise ValueError(f'Not support metric {metric}')

                if ind in bests_metric_ind:
                    if ind == bests_metric_ind[0]:
                        if reduce == 'mean':
                            df_ext.loc[
                                row_ind_sel, f'{metric}_{reduce}_per_nu'] = f"\\textbf{{{_mean:.01f}$\\pm${_std:.01f}}}"
                        else:
                            df_ext.loc[row_ind_sel, f'{metric}_{reduce}_per_nu'] = f"\\textbf{{{_max:.01f}}}"
                    else:
                        if reduce == 'mean':
                            df_ext.loc[
                                row_ind_sel, f'{metric}_{reduce}_per_nu'] = f"\\underline{{{_mean:.01f}$\\pm${_std:.01f}}}"
                        else:
                            df_ext.loc[row_ind_sel, f'{metric}_{reduce}_per_nu'] = f"\\underline{{{_max:.01f}}}"

                else:
                    if reduce == 'mean':
                        df_ext.loc[row_ind_sel, f'{metric}_{reduce}_per_nu'] = f"{_mean:.01f}$\\pm${_std:.01f}"
                    else:
                        df_ext.loc[row_ind_sel, f'{metric}_{reduce}_per_nu'] = f"{_max:.01f}"


    for n_labels in nlabels_list:
        for method_name in method_names:
            row_idx_sel = df_ext_accmax.index[(df_ext_accmax['n_labels'] == n_labels)]
            _sorted_idx = np.argsort(df_ext_accmax.loc[row_idx_sel, f'{metric}_max'].to_numpy())
            bests_metric_ind = []
            bests_metric_ind.append(_sorted_idx[-1])
            bests_metric_ind.append(_sorted_idx[-2])
            for ind, row_ind_sel in enumerate(row_idx_sel):
                _max = 100 * df_ext_accmax.loc[row_ind_sel, f'{metric}_max'].item()
                if ind == bests_metric_ind[0]:
                    df_ext_accmax.loc[row_ind_sel, f'target_max_all'] = f'\\textbf{{{_max:.01f}}}'
                elif ind == bests_metric_ind[1]:
                    df_ext_accmax.loc[row_ind_sel, f'target_max_all'] = f'\\secondbest{{{_max:.01f}}}'
                else:
                    df_ext_accmax.loc[row_ind_sel, f'target_max_all'] = f'{_max:.01f}'


    # Create the detailed table
    for i, nu_mul in enumerate(nu_muls):
        nu_mul_str = f'\multirow{{4}}{{*}}{{${nu_mul}N$}}'
        for method_i, method_name in enumerate(method_names):
            am_list = [map_method_names[method_name]]
            if method_i == 0:
                am_list.append(nu_mul_str)
            else:
                am_list.append('')
            for n_labels in nlabels_list:
                df_sel = df_ext[(df_ext['n_labels'] == n_labels) & (df_ext['nu_mul'] == nu_mul) & (
                        df_ext['method_name'] == method_name)]
                am_list.append(df_sel[f'{metric}_{reduce}_per_nu'].item())
            row_str = " & ".join(am_list)
            row_str += "\\\\ \n"
            detailed_table += row_str

        if i == len(nu_muls) - 1:
            detailed_table += "\\bottomrule \n"
        else:
            detailed_table += "\\midrule \n"

    print(detailed_table)
    print('\n\n')

    for method_name in method_names:
        am_list = [map_method_names[method_name]]
        for n_labels in nlabels_list:
            df_sel = df_ext_accmax[(df_ext_accmax['n_labels'] == n_labels) & (df_ext_accmax['method_name'] == method_name)]
            am_list.append(df_sel[f'target_max_all'].item())
        row_str = " & ".join(am_list)
        row_str += "\\\\ \n"
        summary_table += row_str

    print(summary_table)