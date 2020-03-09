import os
import re
from tqdm import tqdm
import pandas as pd
import numpy as np
import argparse
import pickle
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from tensorboardX import SummaryWriter

from collagen.data import ItemLoader
from collagen.metrics import BalancedAccuracyMeter, KappaMeter, MSEMeter
from collagen.core.utils import auto_detect_device, to_cpu
from collagen.callbacks.visualizer import ConfusionMatrixVisualizer
from collagen.metrics import plot_confusion_matrix

from semixup.utils import parse_item, init_transform_wo_aug

from common.networks import *
from common.oai_most import load_oai_most_datasets

device = auto_detect_device()

# Disable CUDNN
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default="./data/MOST_OAI_00_0_2_cropped", help='Root directory of images')
    parser.add_argument('--root_db', type=str, default="./data/", help='Root directory of meta data')
    parser.add_argument('--save_meta_dir', type=str, default="./Metadata/", help='Directory to save meta data')
    parser.add_argument('--meta_file', type=str, default="oai_most_img_patches.csv", help='Saved csv meta filename')
    parser.add_argument('--config_dir', type=str, default="./configs/", help='Configuration directory')
    parser.add_argument('--reload_data', action='store_true', help='Whether reload data')
    parser.add_argument('--num_threads', type=int, default=0, help='Number of threads for data loader')
    parser.add_argument('--bs', type=int, default=16, help='Batch size')
    parser.add_argument('--d_model', type=str, default="attn1", help='Discriminator model name')
    parser.add_argument('--log_dir', type=str, default=None, help='Log directory')
    parser.add_argument('--comment', type=str, default="semixup", help='Comment')
    parser.add_argument('--seed', type=int, default=12345, help='Random seed')
    parser.add_argument('--most_names_file', type=str, default="./data/most_meta/MOST_names.csv",
                        help='Path of file of MOST names')
    parser.add_argument('--exp_meta_file', type=str, default='./results/exp_amounts_labels_unlabels.csv',
                        help='Path of experimental meta file')
    parser.add_argument('--model_col_name', type=str, default='D.kappa_filename', help='Column name of model path')
    parser.add_argument('--out_dir', type=str, default='results/cm', help='Output directory')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    return args


def filter_most_by_pa(ds, df_most_ex, pas=['PA10']):
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
        y_cls_cpu = to_cpu(y.argmax(dim=1), use_numpy=True)
    elif len(y.shape) == 1:
        y_cls_cpu = to_cpu(y, use_numpy=True)
    else:
        raise ValueError("Only support dims 1 or 2, but got {}".format(len(y.shape)))

    y_cls_cpu = y_cls_cpu.astype(int)

    return y_cls_cpu


def calc_weight_std(y_pred, y_true, x_mean, n_classes):
    assert len(y_true) == len(y_pred)
    weights = [0] * n_classes
    N = len(y_pred)
    for y in y_true:
        weights[y] += 1

    w = []
    for i in range(N):
        w.append(1.0 / weights[y_true[i]])

    w = np.array(w)

    comparison = np.array(y_pred) == np.array(y_true)

    x = []
    for c in comparison:
        x.append(1.0 if c else 0.0)

    x = np.array(x)
    N1 = N

    w_var = N1 * np.sum(w * (x - x_mean) ** 2) / ((N1 - 1) * np.sum(w))

    w_var2 = np.average((x - x_mean) ** 2, weights=w)

    w_std = np.sqrt(w_var)

    std_error = w_std / np.sqrt(N)
    std_error2 = np.sqrt(w_var2) / np.sqrt(N)

    print(f'Std errors are different {std_error} vs {std_error2}')

    return w_std, std_error


if __name__ == "__main__":
    args = init_args()
    save_servere_wrong_predictions = True
    save_detail_preds = True
    if save_servere_wrong_predictions:
        from scipy.special import softmax
        import cv2

    input_meta_results_fullname = args.exp_meta_file
    list_list_pas = ['PA10']

    method_name = ''
    n_cls = 5

    inp_df = pd.read_csv(input_meta_results_fullname, sep='|')

    use_strict_side = False
    required_side = ""  # Empty means to use both knee sides
    if use_strict_side:
        if not required_side or required_side is None:
            raise ValueError(f'Must input `required_side` when `use_strict_side` is on.')
        else:
            print(f'[WARN] Required knee side: {required_side}')

    meta_fullname = os.path.join(args.save_meta_dir, args.meta_file)
    if not os.path.exists(meta_fullname):
        ds = load_oai_most_datasets(root=args.root_db, img_dir=args.root, save_meta_dir=args.save_meta_dir,
                                    saved_patch_dir=args.root_db, force_reload=args.reload_data,
                                    output_filename=args.meta_file, force_rewrite=args.reload_data)

        ds["KL"] = ds["KL"].astype(int)
        if use_strict_side:
            ds_most = ds[(ds["dataset"] == "most") & (ds["Side"] == required_side)]
        else:
            ds_most = ds[ds["dataset"] == "most"]
    else:
        print('Loading pkl file {}'.format(meta_fullname))
        ds = pd.read_csv(meta_fullname, sep='|')
        if use_strict_side:
            ds_most = ds[(ds["dataset"] == "most") & (ds["Side"] == required_side)]
        else:
            ds_most = ds[ds["dataset"] == "most"]

    df_most_ex = pd.read_csv(args.most_names_file, sep='/', header=None, names=["ID_ex", "visit", 'ex1', 'PA', 'ex2'])

    acc_meter = BalancedAccuracyMeter(prefix="d", name="accuracy", parse_output=parse_output, parse_target=parse_target,
                                      topk=1)
    kappa_meter = KappaMeter(prefix="d", name="kappa", parse_target=parse_class, parse_output=parse_class)
    mse_meter = MSEMeter(prefix="d", name="mse", parse_target=parse_class, parse_output=parse_class)

    print(f'Evaluating method {method_name} with model based on {args.model_col_name}...')
    eval_rows = []
    for i, conf in inp_df.iterrows():
        if "method_name" in conf:
            method_name = conf["method_name"]

        m = re.match(r'.+(_[^_]+_dlr.+)', os.path.basename(conf['root_path']))
        if m is None:
            comment = "eval_MOST_" + os.path.basename(conf['root_path'])
        else:
            comment = "eval_MOST_" + m.group(1)
        writer = SummaryWriter(log_dir=args.log_dir, comment=comment)

        d_model_name = conf['d_model']

        print('Create model: .{}.'.format(d_model_name))

        d_model = make_model(model_name=d_model_name, nc=1, ndf=32, n_cls=n_cls).to(device)

        d_model.eval()

        model_map = dict()
        if args.model_col_name in conf and conf[args.model_col_name] and isinstance(conf[args.model_col_name], str):
            model_map['acc_kappa'] = os.path.join(conf['root_path'], 'saved_models', conf[args.model_col_name])
        else:
            print(f"Invalid {args.model_col_name}. Skip!")
            continue

        for md in model_map:
            d_model.load_state_dict(torch.load(model_map[md]), strict=False)
            for list_pas in list_list_pas:
                if isinstance(list_pas, list):
                    pas_str = "_".join(list_pas)
                else:
                    pas_str = list_pas

                print("Processing {} with {} by best {}...".format(pas_str, comment, md))

                cm_norm_viz = ConfusionMatrixVisualizer(writer=writer, tag="CM_" + md + "_" + pas_str, normalize=True,
                                                        labels=["KL" + str(i) for i in range(5)],
                                                        parse_class=parse_class)
                cm_viz = ConfusionMatrixVisualizer(writer=writer, tag="CM_raw_" + md + "_" + pas_str, normalize=False,
                                                   labels=["KL" + str(i) for i in range(5)], parse_class=parse_class)

                ds_most_filtered = filter_most_by_pa(ds_most, df_most_ex, list_pas)

                loader = ItemLoader(root=args.root,
                                    meta_data=ds_most_filtered,
                                    transform=init_transform_wo_aug(),
                                    parse_item_cb=parse_item,
                                    batch_size=args.bs, num_workers=args.num_threads,
                                    shuffle=False, drop_last=False)

                kappa_meter.on_epoch_begin(0)
                acc_meter.on_epoch_begin(0)
                mse_meter.on_epoch_begin(0)

                cm_viz.on_epoch_begin(0)
                cm_norm_viz.on_epoch_begin(0)
                progress_bar = tqdm(range(len(loader)), total=len(loader), desc="Eval::")

                if save_detail_preds:
                    bi_preds_probs_all = []
                    bi_targets_all = []

                for i in progress_bar:
                    sample = loader.sample(1)[0]
                    _output = d_model(sample['data'].to(next(d_model.parameters()).device))
                    sample['target'] = sample['target'].type(torch.int32)

                    output = _output

                    if save_servere_wrong_predictions:
                        preds_logits = to_cpu(output)

                        targets_cpu = to_cpu(sample['target'])

                        preds_probs = softmax(preds_logits, axis=-1)
                        preds = np.argmax(preds_probs, axis=-1)
                        bi_probs_cpu = preds_probs[:, 2] + preds_probs[:, 3] + preds_probs[:, 4]
                        bi_target_cpu = np.zeros_like(targets_cpu).tolist()
                        for i in range(targets_cpu.shape[0]):
                            if targets_cpu[i] >= 2:
                                bi_target_cpu[i] = 1

                        if save_detail_preds:
                            bi_preds_probs_all += bi_probs_cpu.tolist()
                            bi_targets_all += bi_target_cpu

                        # Detect outliers
                        outliers_mask = np.abs(targets_cpu - preds) >= 3

                        for o_id, o_detected in enumerate(outliers_mask):
                            if o_detected:
                                outlier_img_fullname = os.path.join(args.out_dir, 'outliers',
                                                                    f'outlier_{i}_gt{targets_cpu[o_id]}_pred{preds[o_id]}.png')
                                o_img = to_cpu(sample['data'][o_id, :, :, :])
                                o_img = np.concatenate((o_img[0, :, :], o_img[1, :, :]), axis=-1)
                                o_img = 255 * (o_img + 1) / 2.0
                                cv2.imwrite(outlier_img_fullname, o_img)

                    mse_meter.on_minibatch_end(output=_output, target=sample['target'])
                    kappa_meter.on_minibatch_end(output=_output, target=sample['target'])
                    acc_meter.on_minibatch_end(output=_output, target=sample['target'], device=device)
                    cm_viz.on_forward_end(output=_output, target=sample['target'])
                    cm_norm_viz.on_forward_end(output=_output, target=sample['target'])

                    postfix_progress = dict()
                    postfix_progress['mse'] = f'{mse_meter.current():.03f}'
                    postfix_progress['kappa'] = f'{kappa_meter.current():.03f}'
                    postfix_progress['acc'] = f'{acc_meter.current():.03f}'
                    progress_bar.set_postfix(ordered_dict=postfix_progress, refresh=True)
                cm_viz.on_epoch_end()
                cm_norm_viz.on_epoch_end()

                # Confusion matrix
                _labels = ["KL" + str(i) for i in range(5)]
                _corrects = cm_norm_viz.targets
                _predicts = cm_norm_viz.predictions
                fig = plot_confusion_matrix(_corrects, _predicts, _labels)
                canvas = FigureCanvas(fig)
                use_detailed_filename = False
                if use_detailed_filename:
                    cm_fullname = os.path.join(args.out_dir,
                                               "CM_{}_{}_{}_acc{:.2f}_kappa{:.2f}.png".format(f'{method_name}',
                                                                                              conf['n_labels'],
                                                                                              conf['n_unlabels'],
                                                                                              acc_meter.current(),
                                                                                              kappa_meter.current()))
                else:
                    cm_fullname = os.path.join(args.out_dir,
                                               "CM_{}_{}_{}.png".format(f'{method_name}', conf['n_labels'],
                                                                        conf['n_unlabels']))
                canvas.print_png(cm_fullname)

                acc_mean = acc_meter.current()
                kappa_mean = kappa_meter.current()
                w_std, std_error = calc_weight_std(acc_meter.preds, acc_meter.corrects, acc_mean, n_classes=5)

                conf["best_" + md + "_most_" + pas_str + "__kappa"] = kappa_mean
                conf["best_" + md + "_most_" + pas_str + "__acc"] = acc_mean
                conf["best_" + md + "_most_" + pas_str + "__stderr"] = std_error
                if save_detail_preds:
                    conf["bi_preds"] = bi_preds_probs_all
                    conf["bi_targets"] = bi_targets_all
                print('{}: {}'.format("best_" + md + "_most_" + pas_str + "__kappa", kappa_mean))
                print('{}: {}'.format("best_" + md + "_most_" + pas_str + "__acc", acc_mean))
                print('{}: {}'.format("best_" + md + "_most_" + pas_str + "__stderr", std_error))

        eval_rows.append(conf)

    if use_strict_side:
        required_side = "_" + required_side
    else:
        required_side = ""

    output_meta_results_fullname = input_meta_results_fullname[:-4] + required_side + "_result.csv"
    print('Save to {}'.format(output_meta_results_fullname))
    output_pickle_meta_results_fullname = output_meta_results_fullname[:-4] + ".pkl"
    with open(output_pickle_meta_results_fullname, "bw") as f:
        pickle.dump(eval_rows, f, protocol=pickle.HIGHEST_PROTOCOL)
    df_eval = pd.DataFrame(eval_rows)
    df_eval.to_csv(output_meta_results_fullname, sep='|', index=None)
