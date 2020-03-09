import statsmodels.api as sm
import statsmodels.formula.api as smf

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

from semixup.networks import *
from semixup.utils import init_transform_wo_aug, make_model

from common.eval import filter_most_by_pa
from common.oai_most import load_oai_most_datasets

device = auto_detect_device()

torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default="./data/MOST_OAI_00_0_2_cropped", help='Root directory of images')
    parser.add_argument('--root_db', type=str, default="./data/", help='Root directory of meta data')
    parser.add_argument('--save_meta_dir', type=str, default="./Metadata/", help='Directory to save meta data')
    parser.add_argument('--meta_file', type=str, default="./Metadata/most_img_patches_site.csv",
                        help='Saved csv meta filename')
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
    parser.add_argument('--exp_meta_file', type=str, default='./results/exp_sl_ssl_bestmodels_drop035_most_se.csv',
                        help='Path of experimental meta file')
    parser.add_argument('--model_col_name', type=str, default='kappa_acc_filename', help='Column name of model path')

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    return args


def parse_item(root, entry, trf, data_key, target_key):
    img1 = cv2.imread(os.path.join(root, entry["Patch1_name"]), cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(os.path.join(root, entry["Patch2_name"]), cv2.IMREAD_GRAYSCALE)
    img2 = img2[:, ::-1]

    img1 = np.expand_dims(img1, axis=-1)
    img2 = np.expand_dims(img2, axis=-1)

    img1, target = trf((img1, entry["KL"]))
    img2, _ = trf((img2, entry["KL"]))
    img = torch.cat((img1, img2), dim=0)

    return {'data': img, 'target': target, 'side': entry['Side'], 'site': entry['SITE'], 'id': entry['ID']}


if __name__ == "__main__":

    phase = "export" # prepare or export
    output_r_dir = "./results/r/"
    if not os.path.exists(output_r_dir):
        os.mkdir(output_r_dir)

    args = init_args()

    input_meta_results_fullname = args.exp_meta_file
    output_meta_results_fullname = input_meta_results_fullname[:-4] + "_result.csv"
    output_pickle_meta_results_fullname = output_meta_results_fullname[:-4] + ".pkl"
    print('Save to {}'.format(output_meta_results_fullname))

    pa = 'PA10'
    list_list_pas = [pa]

    n_cls = 5

    ds_most = pd.read_csv(args.meta_file, sep='|')
    df_most_ex = pd.read_csv(args.most_names_file, sep='/', header=None, names=["ID_ex", "visit", 'ex1', 'PA', 'ex2'])
    ds_most_filtered = filter_most_by_pa(ds_most, df_most_ex, list_list_pas)

    inp_df = pd.read_csv(input_meta_results_fullname, sep='|')

    weights = [None] * n_cls

    for cls in range(n_cls):
        weights[cls] = len(ds_most_filtered[ds_most_filtered['KL'] == cls].index)

    save_detail_preds = False
    eval_rows = []
    labels_list = [50, 100, 500, 1000]
    our_method_name = 'semixup'
    opponent_method_names = list(set(inp_df['method_name'].tolist()) - set(our_method_name))

    if phase == "prepare":
        for index, conf in inp_df.iterrows():
            d_model_name = conf['d_model']

            weights_fullname = os.path.join(conf['root_path'], 'saved_models', conf[args.model_col_name])
            m = re.match(r'.+(_[^_]+_dlr.+)', os.path.basename(conf['root_path']))

            if m is None:
                comment = "eval_MOST_" + os.path.basename(conf['root_path'])
            else:
                comment = "eval_MOST_" + m.group(1)
            print("Processing {} with {} by best {}...".format(pa, comment, args.model_col_name))

            if "method_name" in conf:
                method_name = conf["method_name"]
            else:
                raise ValueError('Cannot find method name column.')

            print(f'Evaluating method {method_name} with model based on {args.model_col_name}...')

            print('Create model: .{}.'.format(d_model_name))
            d_model = make_model(model_name=d_model_name, nc=1, ndf=32, n_cls=n_cls).to(device)
            d_model.load_state_dict(torch.load(weights_fullname), strict=False)
            d_model.eval()

            loader = ItemLoader(root=args.root,
                                meta_data=ds_most_filtered,
                                transform=init_transform_wo_aug(),
                                parse_item_cb=parse_item,
                                batch_size=args.bs, num_workers=args.num_threads,
                                shuffle=False, drop_last=False)

            pred_cls = []
            targets_cls = []

            progress_bar = tqdm(range(len(loader)), total=len(loader), desc=f"Init eval ::")
            knee_sides = []
            center_sites = []
            ids = []

            for i in progress_bar:
                sample = loader.sample(1)[0]
                output = d_model(sample['data'].to(next(d_model.parameters()).device))
                sample['target'] = sample['target'].type(torch.int32)
                knee_sides += sample['side']
                ids += sample['id']
                center_sites += sample['site'].tolist()

                preds_logits = to_cpu(output)
                preds = np.argmax(preds_logits, axis=-1)

                targets_cpu = to_cpu(sample['target'])
                targets_cls += targets_cpu.tolist()
                pred_cls += preds.tolist()

            assert len(targets_cls) == len(pred_cls)

            pred_results = []
            for j in range(len(targets_cls)):
                if targets_cls[j] == pred_cls[j]:
                    pred_results.append(1.0 / weights[targets_cls[j]])
                else:
                    pred_results.append(0.0)

            conf['knee_sides'] = knee_sides
            conf['center_sites'] = center_sites
            conf['pred_results'] = pred_results
            conf['id'] = ids
            if conf['method_name'] == our_method_name:
                conf['method_code'] = [1] * len(targets_cls)
            else:
                conf['method_code'] = [0] * len(targets_cls)

            eval_rows.append(conf)

        df_eval = pd.DataFrame(eval_rows)
        with open(output_pickle_meta_results_fullname, "bw") as f:
            pickle.dump(df_eval, f, protocol=pickle.HIGHEST_PROTOCOL)

        df_eval.to_csv(output_meta_results_fullname, sep='|', index=None)
    elif phase == "export":
        with open(output_pickle_meta_results_fullname, "br") as f:
            df_eval = pickle.load(f)

        significance_dict = {}
        for n_labels in labels_list:
            significance_dict[n_labels] = {'sb': [], 'not_sb': []}
            our_row = df_eval[(df_eval['method_name'] == our_method_name) & (df_eval['n_labels'] == n_labels)]

            if len(our_row) != 1:
                raise ValueError(f'Must find exactly one row of our method {our_method_name}, but found {len(our_row)}')
            else:
                our_row = our_row.iloc[0]

            our_df_x = pd.DataFrame(our_row['pred_results'], columns=['x'])
            our_df_side = pd.DataFrame(our_row['knee_sides'], columns=['side'])
            our_row['center_sites'] = ["C" + str(int(n)) for n in our_row['center_sites']]
            our_df_center = pd.DataFrame(our_row['center_sites'], columns=['center'])
            our_df_method = pd.DataFrame(our_row['method_code'], columns=['method'])
            our_df_id = pd.DataFrame(our_row['id'], columns=['id'])
            our_df = pd.concat([our_df_x, our_df_id, our_df_side, our_df_center, our_df_method], axis=1)
            print(f'Compare semixup using with {n_labels} labels to...')

            for index, row in df_eval.iterrows():
                if (row['n_labels'] != n_labels) or (row['method_name'] == our_method_name):
                    continue

                print(f'\tvs {row["method_name"]} with {n_labels} labels')
                op_df_x = pd.DataFrame(row['pred_results'], columns=['x'])
                op_df_side = pd.DataFrame(row['knee_sides'], columns=['side'])
                row['center_sites'] = ["C" + str(int(n)) for n in row['center_sites']]
                op_df_center = pd.DataFrame(row['center_sites'], columns=['center'])
                op_df_method = pd.DataFrame(row['method_code'], columns=['method'])
                op_df_id = pd.DataFrame(row['id'], columns=['id'])
                op_df = pd.concat([op_df_x, op_df_id, op_df_side, op_df_center, op_df_method], axis=1)

                df_data = pd.concat([our_df, op_df], axis=0)

                r_filename = f'data_{n_labels}_compare_{our_method_name}_{our_row["d_model"]}_to_{row["method_name"]}_{row["d_model"]}.csv'
                output_r_fullname = os.path.join(output_r_dir, r_filename)

                df_data.to_csv(output_r_fullname, sep=';', index=None)

        print(significance_dict)
