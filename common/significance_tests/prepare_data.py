import os
import re
from tqdm import tqdm
import pandas as pd
import numpy as np
import argparse
import pickle
import torch

from tensorboardX import SummaryWriter
from collagen.data import ItemLoader
from collagen.metrics import BalancedAccuracyMeter, KappaMeter, MSEMeter
from collagen.core.utils import auto_detect_device, to_cpu
from collagen.callbacks.visualizer import ConfusionMatrixVisualizer
from collagen.metrics import plot_confusion_matrix
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from semixup.networks import *
from semixup.utils import parse_item, init_transform_wo_aug, make_model

from common.eval import parse_class, parse_output, parse_target, filter_most_by_pa
from common.oai_most import load_oai_most_datasets

device = auto_detect_device()

# Disable CuDNN
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
    parser.add_argument('--exp_meta_file', type=str, default='./results/exp_bestmodels_wilcoxon_test.csv',
                        help='Path of experimental meta file')
    parser.add_argument('--model_col_name', type=str, default='kappa_acc_filename', help='Column name of model path')
    parser.add_argument('--out_dir', type=str, default='results/cm', help='Output directory')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    return args


if __name__ == "__main__":
    args = init_args()
    save_servere_wrong_predictions = True
    if save_servere_wrong_predictions:
        from scipy.special import softmax
        import cv2

    n_folds = 20
    most_fullname = f"./Metadata/cv_split_{n_folds}folds_most.pkl"
    with open(most_fullname, 'rb') as f:
        most_folds = pickle.load(f)

    input_meta_results_fullname = args.exp_meta_file
    output_meta_results_fullname = input_meta_results_fullname[:-4] + "_result.csv"
    print('Save to {}'.format(output_meta_results_fullname))

    list_list_pas = ['PA10']

    method_name = ''
    n_cls = 5

    inp_df = pd.read_csv(input_meta_results_fullname, sep='|')

    meta_fullname = os.path.join(args.save_meta_dir, args.meta_file)

    save_detail_preds = False

    print(f'Evaluating method {method_name} with model based on {args.model_col_name}...')
    eval_rows = []

    all_preds_cls = []
    pa = 'PA10'
    n_bootstrap = 1000

    labels_list = [50, 100, 500, 1000]
    nus = [1, 2, 3, 4, 5, 6]

    method_list = inp_df['method_name'].unique().tolist()

    for ind, conf in inp_df.iterrows():
        n_labels = conf['n_labels']
        n_unlabels = conf['n_unlabels']
        d_model_name = conf['d_model']

        weights_fullname = os.path.join(conf['root_path'], 'saved_models', conf[args.model_col_name])
        m = re.match(r'.+(_[^_]+_dlr.+)', os.path.basename(conf['root_path']))

        if m is None:
            comment = "eval_MOST_" + os.path.basename(conf['root_path'])
        else:
            comment = "eval_MOST_" + m.group(1)
        print("Processing {} with {} by best {}...".format(pa, comment, args.model_col_name))

        conf["method_name"] = method_name

        m = re.match(r'.+(_[^_]+_dlr.+)', os.path.basename(conf['root_path']))

        print('Create model: .{}.'.format(d_model_name))
        d_model = make_model(model_name=d_model_name, nc=1, ndf=32, n_cls=n_cls).to(device)
        d_model.load_state_dict(torch.load(weights_fullname), strict=False)
        d_model.eval()

        for i_fold in range(n_folds):
            print(f'\nCase: n_labels = {n_labels}, n_unlabels = {n_unlabels}, i_fold = {i_fold}')
            # Get validation set
            out_conf = conf.copy(deep=False)
            ds_most = most_folds[i_fold][1]
            loader = ItemLoader(root=args.root,
                                meta_data=ds_most,
                                transform=init_transform_wo_aug(),
                                parse_item_cb=parse_item,
                                batch_size=args.bs, num_workers=args.num_threads,
                                shuffle=False, drop_last=False)

            if save_detail_preds:
                bi_preds_probs_all = []
                bi_targets_all = []

            acc_meter = BalancedAccuracyMeter(prefix="d", name="accuracy", parse_output=parse_output,
                                              parse_target=parse_target)
            kappa_meter = KappaMeter(prefix="d", name="kappa", parse_target=parse_class,
                                     parse_output=parse_class)
            mse_meter = MSEMeter(prefix="d", name="mse", parse_target=parse_class, parse_output=parse_class)

            pred_cls = []
            targets_cls = []

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

            out_conf['fold_test'] = i_fold
            out_conf['acc_test'] = acc_test
            out_conf['kappa_test'] = kappa_test
            out_conf['mse_test'] = mse_test

            eval_rows.append(out_conf)

    output_pickle_meta_results_fullname = output_meta_results_fullname[:-4] + ".pkl"
    with open(output_pickle_meta_results_fullname, "bw") as f:
        pickle.dump(eval_rows, f, protocol=pickle.HIGHEST_PROTOCOL)
    df_eval = pd.DataFrame(eval_rows)
    df_eval.to_csv(output_meta_results_fullname, sep='|', index=None)
