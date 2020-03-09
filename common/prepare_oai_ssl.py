import os
import argparse

from collagen.data import SSFoldSplit
from common.oai_most import load_oai_most_datasets


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_img', type=str, default="./data/MOST_OAI_FULL_0_2", help='Root directory of images')
    parser.add_argument('--root_db', type=str, default="./data", help='Root directory of original metadata')
    parser.add_argument('--save_meta_dir', default='./processed_data/Metadata', help='Where to save processed metadata')
    parser.add_argument('--save_img_dir', default='./processed_data/MOST_OAI_00_0_2_cropped',
                        help='Where to save processed images')
    parser.add_argument('--output_file', default='oai_most_img_patches.csv', help='File name of processed metadata')
    parser.add_argument('--reload_data', action='store_true', help='Whether to reload data')
    parser.add_argument('--seed', type=int, default=12345, help='Random seed')

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = init_args()

    if not os.path.exists(args.root_db):
        raise ValueError(f'Not found directory {args.root_db}')

    if not os.path.exists(args.root_img):
        raise ValueError(f'Not found directory {args.root_img}')

    if not os.path.exists(args.save_meta_dir):
        os.makedirs(args.save_meta_dir)

    if not os.path.exists(args.save_img_dir):
        os.makedirs(args.save_img_dir)

    list_labeled_data = [50, 100, 500, 1000]
    n_folds = 5
    u_ts = range(7)
    list_unlabeled_target_col = [None]

    is_first = True

    for u_target_col in list_unlabeled_target_col:
        for n_l in list_labeled_data:
            for t in u_ts:
                n_u = n_l * t
                print("Storing data of unlabeled_target_col: {}, n_labeled_data: {}, n_unlabeled_data: {}...".format(
                    u_target_col, n_l, n_u))
                ds = load_oai_most_datasets(root=args.root_db, img_dir=args.root_img, save_meta_dir=args.save_meta_dir,
                                            saved_patch_dir=args.save_img_dir, force_reload=args.reload_data and is_first,
                                            output_filename=args.output_file,
                                            force_rewrite=args.reload_data and is_first)
                is_first = False

                ds["KL"] = ds["KL"].astype(int)
                ds["ID"] = ds["ID"].astype(str)
                ds_oai = ds[ds["dataset"] == "oai"]

                # Split labeled and unlabeled parts
                # Split 5 folds
                splitter_train = SSFoldSplit(ds_oai, n_ss_folds=4, n_folds=n_folds, target_col="KL",
                                             random_state=args.seed,
                                             labeled_train_size_per_class=n_l,
                                             unlabeled_train_size=5 * n_u, group_col='ID',
                                             equal_target=True, equal_unlabeled_target=False,
                                             shuffle=True)
                file_name = f"cv_split_{n_folds}fold_l_{n_l}_u_{n_u}_False_col_None.pkl"
                out_fullname = os.path.join(args.save_meta_dir, file_name)
                print(f'Writing file {out_fullname}')
                splitter_train.dump(out_fullname)
