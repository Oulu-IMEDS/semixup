import os
import pandas as pd
import re
import numpy as np

list_saved_dir = ['./all_runs/runs_sl_ssl/sl/',
                  './all_runs/runs_sl_ssl/ict/',
                  './all_runs/runs_sl_ssl/pimodel/',
                  './all_runs/runs_sl_ssl/mixmatch/',
                  './all_runs/runs_sl_ssl/semixup/']

meta_results_fullname = './results/exp_sl_ssl_methods.csv'


def parse_config_name(matches):
    configs = dict()

    configs["device"] = matches.group(1)

    configs["method_name"] = matches.group(2)
    configs['comment'] = matches.group(3)
    configs["d_lr"] = float(matches.group(4))
    configs["d_model"] = matches.group(5)
    configs["ndf"] = int(matches.group(6))
    configs["n_labels"] = int(matches.group(7))
    configs["n_unlabels"] = int(matches.group(8))
    configs["trans"] = matches.group(9)
    configs["sampling"] = matches.group(10)
    configs["fold_id"] = int(matches.group(11))
    return configs


def append_if_needed(groups_by_labels, n_labels, value):
    if str(n_labels) not in groups_by_labels:
        groups_by_labels[str(n_labels)] = [value]
    else:
        groups_by_labels[str(n_labels)].append(value)

    return groups_by_labels


rows = []

kc_by_labels = {}
acc_by_labels = {}


def unify_col_names(s):
    s = str.replace(s, 'D.ss_', '')
    s = str.replace(s, 'D.', '')
    s = str.replace(s, '.', '')
    s = str.replace(s, 'cls_loss', 'loss_cls')

    return s


for saved_dir in list_saved_dir:
    for i, config_name in enumerate(os.listdir(saved_dir)):
        matches = re.match(
            r'^.+_\d+-\d+-\d+_([^_]+)_([^_]+)_([^_]*)_dlr_(.+)_(.+)_ndf(\d+)_data_(\d+)_(\d+)_(.+)_(.+)_fold(\d+)$',
            config_name)
        if matches is None:
            matches = re.match(r'^.+_\d+-\d+-\d+_(.+)_([^_]+)_data_([^_]+)_fold(\d+)', config_name)
            if matches is None:
                print('Cannot parse {}!'.format(config_name))
                continue
            else:
                conf = dict()
                conf['device'] = matches.group(1)
                mt = re.match(r'.+bullx(\w+)$', conf['device'])
                if mt:
                    print('Yes')
                    conf['method_name'] = mt.group(1)
                else:
                    conf['method_name'] = ""
                conf['comment'] = matches.group(2)
                conf['n_labels'] = int(matches.group(3))
                conf['n_unlabels'] = int(matches.group(3))
                conf['fold_id'] = int(matches.group(4))

                model_matches = re.match(r'(.+)\..+\.(.+)', conf['comment'])
                if model_matches is not None:
                    conf["d_model"] = model_matches.group(1) + model_matches.group(2)

        else:
            conf = parse_config_name(matches)

        root_path = os.path.join(saved_dir, config_name)
        conf["root_path"] = root_path

        for _r, _d, _f in os.walk(root_path):
            for f in _f:
                if f.startswith('events.out'):
                    file_fullname = os.path.join(_r, f)
                    conf['tensorboard'] = file_fullname
                elif f.endswith('.pth'):
                    if 'acc' in f and 'kappa' in f:
                        m = re.match(r'model_(\d+)_(\d+_\d+)_eval_?(.+kappa)_([\.\d]+)_eval_?(.+acc)_([\.\d]+)\.pth$',
                                     f)
                        meter = m.group(3) + '_' + m.group(5)
                        meter = unify_col_names(meter)
                        conf[meter + '_epoch'] = int(m.group(1))
                        conf[meter + '_date'] = m.group(2)
                        conf[meter + '_val'] = '{}:{}'.format(m.group(4), m.group(6))
                        conf['val_kappa'] = float(m.group(4))
                        conf['val_acc'] = float(m.group(6))
                        conf[meter + '_filename'] = f
                        kc_by_labels = append_if_needed(kc_by_labels, conf["n_labels"], conf['val_kappa'])
                        acc_by_labels = append_if_needed(acc_by_labels, conf["n_labels"], conf['val_acc'])

                    else:
                        m = re.match(r'model_(\d+)_(\d+_\d+)_eval.([^\d]+)_([\.\d]+).+\.pth$', f)
                        meter = unify_col_names(m.group(3))
                        conf[meter + '_epoch'] = int(m.group(1))
                        conf[meter + '_date'] = m.group(2)
                        conf[meter + '_val'] = float(m.group(4))
                        conf[meter + '_filename'] = f
        rows.append(conf)

df = pd.DataFrame(rows)
headers = ['fold_id', 'device', 'method_name', 'd_model', 'n_labels', 'n_unlabels', 'val_acc', 'val_kappa', 'root_path',
           'kappa_acc_filename']
df.to_csv(meta_results_fullname, sep='|', index=None, columns=headers)

# Print stats
for k in kc_by_labels:
    print(f"KC of {k}\t{np.mean(np.array(kc_by_labels[k]))}\t {np.std(np.array(kc_by_labels[k]))}")

for k in acc_by_labels:
    print(f"BA of {k}\t{np.mean(np.array(acc_by_labels[k]))}\t {np.std(np.array(acc_by_labels[k]))}'")

print(meta_results_fullname)
