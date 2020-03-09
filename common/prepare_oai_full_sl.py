import os
import pandas as pd
import argparse

from collagen.data import FoldSplit


def init_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_meta_file', type=str, default="./processed_data/Metadata/oai_most_img_patches.csv", help='Input metadata file')
    parser.add_argument('--save_meta_dir', default='./processed_data/Metadata', help='Where to save processed metadata')
    parser.add_argument('--seed', type=int, default=12345, help='Random seed')

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = init_args()
    n_folds = 5

    if not os.path.isfile(args.input_meta_file):
        raise ValueError(f'Not found {args.input_meta_file}')

    ds = pd.read_csv(args.input_meta_file, sep='|')

    ds["KL"] = ds["KL"].astype(int)
    ds["ID"] = ds["ID"].astype(str)
    ds_oai = ds[ds["dataset"] == "oai"]

    ds_oai.set_index(['ID', 'Side'])

    splitter = FoldSplit(ds_oai, n_folds=n_folds, target_col='KL', group_col='ID', random_state=args.seed)

    file_name = f"cv_split_{n_folds}fold_oai.pkl"
    out_fullname = os.path.join(args.save_meta_dir, file_name)
    print(f'Writing output file to {out_fullname}')
    splitter.dump(out_fullname)
