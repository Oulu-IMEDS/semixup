import os
import re
from tqdm import tqdm
import pandas as pd
import numpy as np
import argparse
import pickle

from tensorboardX import SummaryWriter
from collagen.data import ItemLoader
from collagen.metrics import BalancedAccuracyMeter, KappaMeter, MSEMeter
from collagen.core.utils import auto_detect_device, to_cpu
from sklearn.metrics import roc_curve, auc, average_precision_score
from collagen.callbacks.visualizer import ConfusionMatrixVisualizer
from collagen.metrics import plot_confusion_matrix
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from common.networks import *
from semixup.utils import parse_item, init_transform_wo_aug
from common.networks import make_model
from ssgan.utils import load_oai_most_datasets

device = auto_detect_device()

# Disable CuDNN
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic=True

def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default="./data/MOST_OAI_00_0_2_cropped", help='Root directory of images')
    parser.add_argument('--root_db', type=str, default="./data/", help='Root directory of meta data')
    parser.add_argument('--save_meta_dir', type=str, default="./Metadata/", help='Directory to save meta data')
    parser.add_argument('--meta_file', type=str, default="oai_most_img_patches.csv", help='Saved csv meta filename')
    parser.add_argument('--config_dir', type=str, default="./configs/", help='Configuration directory')
    parser.add_argument('--reload_data', action='store_true', help='Whether reload data')
    parser.add_argument('--num_threads', type=int, default=0, help='Number of threads for data loader')
    parser.add_argument('--bs', type=int, default=40, help='Batch size')
    parser.add_argument('--d_model', type=str, default="attn1", help='Discriminator model name')
    parser.add_argument('--log_dir', type=str, default=None, help='Log directory')
    parser.add_argument('--comment', type=str, default="semixup", help='Comment')
    parser.add_argument('--seed', type=int, default=12345, help='Random seed')
    parser.add_argument('--most_names_file', type=str, default="./data/most_meta/MOST_names.csv", help='Path of file of MOST names')
    parser.add_argument('--exp_meta_file', type=str, default='./results/exp_amounts_labels_unlabels.csv',
                        help='Path of experimental meta file')
    parser.add_argument('--out_file', type=str, default='./results/exp_amounts_labels_unlabels_most.csv',
                        help='Path of output experimental file')
    parser.add_argument('--model_col_name', type=str, default='D.kappa_filename', help='Column name of model path')
    parser.add_argument('--out_dir', type=str, default='results/cm', help='Output directory')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    return args


def filter_most_by_pa(ds, df_most_ex, pas=['PA05', 'PA10', 'PA15']):

    std_rows = []
    for i, row in df_most_ex.iterrows():
        std_row = dict()
        std_row['ID'] = row['ID_ex'].split('_')[0]
        std_row['visit_id'] = int(row['visit'][1:])
        std_row['PA'] = row['PA']
        std_rows.append(std_row)
    df_most_pa = pd.DataFrame(std_rows)

    ds_most_filtered = pd.merge(ds, df_most_pa, on=['ID', 'visit_id'])
    if isinstance(pas, str):
        ds_most_filtered = ds_most_filtered[ds_most_filtered['PA'] == pas]
    return ds_most_filtered

def parse_target(target):
    return target

def parse_output(output):
    return output

def parse_class(y):
    if y is None:
        return None
    elif len(y.shape) == 2:
        y_cls_cpu = np.argmax(y, axis=1)
    elif len(y.shape) == 1:
        y_cls_cpu = y
    else:
        raise ValueError("Only support dims 1 or 2, but got {}".format(len(y.shape)))

    y_cls_cpu = y_cls_cpu.astype(int)

    return y_cls_cpu


def bootstrap_df_loc(df, target_key, replace=True):
    targets = df[target_key].unique()
    sampled_idx = []
    for t in targets:
        idx_by_cls = np.flatnonzero(df[target_key] == t).tolist()
        sampled_idx += np.random.choice(idx_by_cls, len(idx_by_cls), replace=replace).tolist()
    return df.iloc[sampled_idx], sampled_idx

def calc_ci(stats, alpha=0.95):
    p = ((1.0 - alpha) / 2.0) * 100
    lower = max(0.0, np.percentile(stats, p))
    p = (alpha + ((1.0 - alpha) / 2.0)) * 100
    upper = min(1.0, np.percentile(stats, p))
    return lower, upper, np.array(stats).mean()


if __name__ == "__main__":
    args = init_args()
    save_servere_wrong_predictions = True
    if save_servere_wrong_predictions:
        from scipy.special import softmax
        import cv2

    model_name_map = {'alekseiAP': 'aleksei_attn'}
    input_meta_results_fullname = args.exp_meta_file
    output_meta_results_fullname = input_meta_results_fullname[:-4] + "_result.csv"
    print('Save to {}'.format(output_meta_results_fullname))
    # list_list_pas = ['PA05', 'PA10', 'PA15', ['PA05', 'PA10', 'PA15']]
    list_list_pas = ['PA10']

    method_name = ''

    inp_df = pd.read_csv(input_meta_results_fullname, sep='|')

    meta_fullname = os.path.join(args.save_meta_dir, args.meta_file)
    if not os.path.exists(meta_fullname):
        ds = load_oai_most_datasets(root=args.root_db, img_dir=args.root, save_meta_dir=args.save_meta_dir,
                                    saved_patch_dir=args.root_db, force_reload=args.reload_data,
                                    output_filename=args.meta_file, force_rewrite=args.reload_data)

        ds["KL"] = ds["KL"].astype(int)
        ds_most = ds[ds["dataset"] == "most"]
    else:
        print('Loading pkl file {}'.format(meta_fullname))
        ds = pd.read_csv(meta_fullname, sep='|')
        ds_most = ds[ds["dataset"] == "most"]


    df_most_ex = pd.read_csv(args.most_names_file, sep='/', header=None, names=["ID_ex", "visit", 'ex1', 'PA', 'ex2'])

    save_detail_preds = False

    print(f'Evaluating method {method_name} with model based on {args.model_col_name}...')
    eval_rows = []

    all_preds_cls = []
    pa = 'PA10'
    n_bootstrap = 1000

    for i, conf in inp_df.iterrows():
        conf["method_name"] = method_name

        m = re.match(r'.+(_[^_]+_dlr.+)', os.path.basename(conf['root_path']))
        if m is None:
            comment = "eval_MOST_" + os.path.basename(conf['root_path'])
        else:
            comment = "eval_MOST_" + m.group(1)
        writer = SummaryWriter(log_dir=args.log_dir, comment=comment)

        if conf['d_model'] in model_name_map:
            d_model_name = model_name_map[conf['d_model']]
        else:
            d_model_name = conf['d_model']
        print('Create model: .{}.'.format(d_model_name))
        # d_model = DisCustomVGGAux(nc=1, ndf=32, n_cls=5).to(device)
        n_cls = 5

        d_model = make_model(model_name=d_model_name, nc=1, ndf=32, n_cls=n_cls).to(device)

        d_model.eval()

        model_map = dict()
        model_map['kappa_acc'] = os.path.join(conf['root_path'], 'saved_models', conf[args.model_col_name])

        for md in model_map:

            d_model.load_state_dict(torch.load(model_map[md]), strict=False)
            print("Processing {} with {} by best {}...".format(pa, comment, md))

            ds_most_filtered = filter_most_by_pa(ds_most, df_most_ex, pa)

            loader = ItemLoader(root=args.root,
                                meta_data=ds_most_filtered,
                                transform=init_transform_wo_aug(),
                                parse_item_cb=parse_item,
                                batch_size=args.bs, num_workers=args.num_threads,
                                shuffle=False, drop_last=False)

            if save_detail_preds:
                bi_preds_probs_all = []
                bi_targets_all = []

            acc_meter = BalancedAccuracyMeter(prefix="d", name="accuracy", parse_output=parse_output,
                                              parse_target=parse_target)
            kappa_meter = KappaMeter(prefix="d", name="kappa", parse_target=parse_class, parse_output=parse_class)
            mse_meter = MSEMeter(prefix="d", name="mse", parse_target=parse_class, parse_output=parse_class)

            pred_cls = []
            targets_cls = []

            bi_preds_probs_all = []
            bi_targets_all = []

            progress_bar = tqdm(range(len(loader)), total=len(loader), desc=f"Init eval ::")

            acc_meter.on_epoch_begin(0)
            kappa_meter.on_epoch_begin(0)
            mse_meter.on_epoch_begin(0)

            for i in progress_bar:
                sample = loader.sample(1)[0]
                output = d_model(sample['data'].to(next(d_model.parameters()).device))
                sample['target'] = sample['target'].type(torch.int32)

                preds_logits = to_cpu(output)
                # preds = np.argmax(preds_logits, axis=-1)

                targets_cpu = to_cpu(sample['target'])
                targets_cls += targets_cpu.tolist()
                pred_cls.append(preds_logits)

                preds_probs = softmax(preds_logits, axis=-1)  # sigmoid(fold_output)
                bi_probs_cpu = preds_probs[:, 2] + preds_probs[:, 3] + preds_probs[:, 4]
                bi_target_cpu = np.zeros_like(targets_cpu).tolist()
                for i in range(targets_cpu.shape[0]):
                    if targets_cpu[i] >= 2:
                        bi_target_cpu[i] = 1

                bi_preds_probs_all += bi_probs_cpu.tolist()
                bi_targets_all += bi_target_cpu

                kappa_meter.on_minibatch_end(output=preds_logits, target=targets_cpu)
                acc_meter.on_minibatch_end(output=preds_logits, target=targets_cpu)
                mse_meter.on_minibatch_end(output=preds_logits, target=targets_cpu)

                postfix_progress = dict()
                postfix_progress['kappa'] = f'{kappa_meter.current():.03f}'
                postfix_progress['acc'] = f'{acc_meter.current():.03f}'
                postfix_progress['mse'] = f'{mse_meter.current():.03f}'
                progress_bar.set_postfix(ordered_dict=postfix_progress, refresh=True)

            acc_test = acc_meter.current()
            kappa_test = kappa_meter.current()
            mse_test = mse_meter.current()

            bi_targets_all = np.array(bi_targets_all)
            bi_preds_probs_all = np.array(bi_preds_probs_all)
            fpr, tpr, _ = roc_curve(y_true=bi_targets_all, y_score=bi_preds_probs_all)
            auc_test = auc(fpr, tpr)
            ap_test = average_precision_score(y_true=bi_targets_all, y_score=bi_preds_probs_all)

            pred_cls = np.concatenate(pred_cls, axis=0)

            bt_progress_bar = tqdm(range(n_bootstrap), total=n_bootstrap, desc=f"Bootstrap ::")
            all_accs = []
            all_kappas = []
            all_mse = []
            all_aucs = []
            all_aps = []

            pred_cls = np.array(pred_cls)
            targets_cls = np.array(targets_cls)

            for _ in bt_progress_bar:
                acc_meter.on_epoch_begin(0)
                kappa_meter.on_epoch_begin(0)
                mse_meter.on_epoch_begin(0)
                bt_bi_preds_probs_all = []
                bt_bi_targets_all = []

                _, selected_idx = bootstrap_df_loc(ds_most_filtered, target_key='KL', replace=True)
                selected_idx = np.array(selected_idx)
                acc_meter.on_minibatch_end(output=pred_cls[selected_idx,:], target=targets_cls[selected_idx])
                kappa_meter.on_minibatch_end(output=pred_cls[selected_idx,:], target=targets_cls[selected_idx])
                mse_meter.on_minibatch_end(output=pred_cls[selected_idx, :], target=targets_cls[selected_idx])

                bt_bi_preds_probs_all.extend(bi_preds_probs_all[selected_idx])
                bt_bi_targets_all.extend(bi_targets_all[selected_idx])
                fpr, tpr, _ = roc_curve(bt_bi_targets_all, bt_bi_preds_probs_all)
                auc_val = auc(x=fpr, y=tpr)
                ap_val = average_precision_score(y_true=bt_bi_targets_all, y_score=bt_bi_preds_probs_all)

                postfix_progress = {}
                postfix_progress['kappa'] = f'{kappa_meter.current():.03f}'
                postfix_progress['acc'] = f'{acc_meter.current():.03f}'
                postfix_progress['mse'] = f'{mse_meter.current():.03f}'

                bt_progress_bar.set_postfix(ordered_dict=postfix_progress, refresh=True)

                all_accs.append(acc_meter.current())
                all_kappas.append(kappa_meter.current())
                all_mse.append(mse_meter.current())
                all_aucs.append(auc_val)
                all_aps.append(ap_val)

            acc_ci_l, acc_ci_u, acc_ci_m = calc_ci(all_accs)
            kappa_ci_l, kappa_ci_u, kappa_ci_m = calc_ci(all_kappas)
            mse_ci_l, mse_ci_u, mse_ci_m = calc_ci(all_mse)
            auc_ci_l, auc_ci_u, auc_ci_m = calc_ci(all_aucs)
            ap_ci_l, ap_ci_u, ap_ci_m = calc_ci(all_aps)

            print(f'{comment}\nAcc: {acc_ci_l}, {acc_ci_u}, {acc_ci_m}')
            print(f'Kappa: {kappa_ci_l}, {kappa_ci_u}, {kappa_ci_m}')
            print(f'MSE: {mse_ci_l}, {mse_ci_u}, {mse_ci_m}')
            print(f'AUC: {auc_ci_l}, {auc_ci_u}, {auc_ci_m}')
            print(f'AP: {ap_ci_l}, {ap_ci_u}, {ap_ci_m}')

            conf['acc_ci_lb'] = acc_ci_l
            conf['acc_ci_ub'] = acc_ci_u
            conf['acc_test'] = acc_test

            conf['kappa_ci_lb'] = kappa_ci_l
            conf['kappa_ci_ub'] = kappa_ci_u
            conf['kappa_test'] = kappa_test

            conf['mse_ci_lb'] = mse_ci_l
            conf['mse_ci_ub'] = mse_ci_u
            conf['mse_test'] = mse_test

            conf['auc_ci_lb'] = auc_ci_l
            conf['auc_ci_ub'] = auc_ci_u
            conf['auc_test'] = auc_test

            conf['ap_ci_lb'] = ap_ci_l
            conf['ap_ci_ub'] = ap_ci_u
            conf['ap_test'] = ap_test

        # all_preds_cls.append(pred_cls)
        eval_rows.append(conf)
    # all_preds_cls.stack(all_preds_cls, axis=1)

    output_pickle_meta_results_fullname = output_meta_results_fullname[:-4] + ".pkl"
    with open(output_pickle_meta_results_fullname, "bw") as f:
        pickle.dump(eval_rows, f, protocol=pickle.HIGHEST_PROTOCOL)
    df_eval = pd.DataFrame(eval_rows)
    df_eval.to_csv(output_meta_results_fullname, sep='|', index=None)
