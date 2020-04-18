import os
import argparse
import pandas as pd
import random
from sklearn import model_selection
from sklearn.utils import resample
import pickle


from collagen.data import Splitter

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
    parser.add_argument('--seed', type=int, default=-1, help='Random seed')

    args = parser.parse_args()

    return args


class SSFoldSplit(Splitter):
    def __init__(self, ds: pd.DataFrame, n_ss_folds: int = 3, n_folds: int = 5, target_col: str = 'target',
                 random_state: int or None = None, unlabeled_target_col: str = '5means_classes', test_ratio: int = 0.25,
                 labeled_train_size_per_class: int = None, unlabeled_train_size_per_class: int = None,
                 labeled_train_size: int = None, unlabeled_train_size: int = None, group_col: str or None = None,
                 equal_target: bool = True, equal_unlabeled_target: bool = True, shuffle: bool = True):
        super().__init__()

        self._test_ratio = test_ratio

        if equal_target and labeled_train_size_per_class is None:
            raise ValueError("labeled_train_size_per_class must be determined when \
            equal_target is True, but found None")

        if not equal_target and labeled_train_size is None:
            raise ValueError("labeled_train_size must be determined when \
            equal_target is False, but found None")

        # Master split into Label/Unlabel
        if group_col is None:
            master_splitter = model_selection.StratifiedKFold(n_splits=n_ss_folds, random_state=random_state)
            unlabeled_idx, labeled_idx = next(master_splitter.split(ds, ds[target_col]))
        else:
            master_splitter = model_selection.GroupKFold(n_splits=n_ss_folds)
            unlabeled_idx, labeled_idx = next(master_splitter.split(ds, ds[target_col], groups=ds[group_col]))

        print(f'After ss split, no of labels: {len(labeled_idx)}, no of unlabels: {len(unlabeled_idx)}')

        unlabeled_ds = ds.iloc[unlabeled_idx]
        # u_groups = ds[unlabeled_target_col].iloc[unlabeled_idx]
        labeled_ds = ds.iloc[labeled_idx]
        l_groups = ds[target_col].iloc[labeled_idx]

        if not equal_target and labeled_train_size is not None and labeled_train_size > len(labeled_idx):
            raise ValueError('Input labeled train size {} is larger than actual labeled train size {}'.format(
                labeled_train_size, len(labeled_idx)))

        if unlabeled_train_size is not None and unlabeled_train_size > len(unlabeled_idx):
            unlabeled_train_size = len(unlabeled_idx)
            # raise ValueError('Input unlabeled train size {} is larger than actual unlabeled train size {}'.format(unlabeled_train_size, len(unlabeled_idx)))

        # Split labeled data using GroupKFold
        # Split unlabeled data using GroupKFold
        self.__cv_folds_idx = []
        self.__ds_chunks = []

        if group_col is None:
            labeled_splitter = model_selection.StratifiedKFold(n_splits=n_folds, random_state=random_state+2)
            labeled_spl_iter = labeled_splitter.split(labeled_ds, labeled_ds[target_col])
        else:
            labeled_splitter = model_selection.GroupKFold(n_splits=n_folds)
            labeled_spl_iter = labeled_splitter.split(labeled_ds, labeled_ds[target_col], groups=labeled_ds[group_col])

        for i in range(n_folds):
            # u_train, u_test = next(unlabeled_spl_iter)
            u_train = [i for i in range(len(unlabeled_ds.index))]
            l_train, l_test = next(labeled_spl_iter)
            if equal_unlabeled_target:
                u_train_target = unlabeled_ds[unlabeled_target_col]
                # u_test_target = unlabeled_ds[unlabeled_target_col]
            l_train_target = labeled_ds.iloc[l_train][target_col]
            l_train_data = labeled_ds.iloc[l_train]

            l_test_target = labeled_ds.iloc[l_test][target_col]
            l_test_data = labeled_ds.iloc[l_test]

            # Sample labeled_train_size of labeled data
            if equal_target:
                filtered_l_train_idx, chosen_l_train = self._sample_labeled_data(l_train_data, l_train_target,
                                                                                 target_col,
                                                                                 labeled_train_size_per_class,
                                                                                 random_state)

                filtered_l_test_idx, chosen_l_test = self._sample_labeled_data(l_test_data, l_test_target,
                                                                                 target_col,
                                                                                 int(labeled_train_size_per_class*self._test_ratio),
                                                                                 random_state)
            else:
                if labeled_train_size is not None:
                    chosen_l_train, _ = model_selection.train_test_split(l_train, train_size=labeled_train_size,
                                                                         random_state=random_state, shuffle=shuffle,
                                                                         stratify=l_train_target)
                    chosen_l_test, _ = model_selection.train_test_split(l_test, train_size=int(labeled_train_size*self._test_ratio),
                                                                         random_state=random_state, shuffle=shuffle,
                                                                         stratify=l_train_target)
                else:
                    chosen_l_train = l_train
                    chosen_l_test = l_test
                filtered_l_train_idx = labeled_ds.iloc[chosen_l_train]
                filtered_l_test_idx = labeled_ds.iloc[chosen_l_test]

            # Sample unlabeled_train_size of labeled data
            if equal_unlabeled_target:
                filtered_u_train_idx, chosen_u_train = self._sample_unlabeled_data(unlabeled_ds, u_train, unlabeled_target_col,
                                                                                   u_train_target,
                                                                                   unlabeled_train_size_per_class,
                                                                                   random_state)

            else:
                if unlabeled_train_size is not None:
                    # chosen_u_train, _ = model_selection.train_test_split(u_train, train_size=unlabeled_train_size,
                    #                                                      random_state=random_state, shuffle=shuffle)
                    is_replace = unlabeled_train_size > len(u_train)
                    if is_replace:
                        print(f'>>> [WARN] Sampling unlabeled data with replacement as {unlabeled_train_size} > {len(u_train)}')
                    chosen_u_train = resample(u_train, n_samples=unlabeled_train_size, replace=is_replace, random_state=random_state)
                    unlabeled_test_size = int(unlabeled_train_size*self._test_ratio)
                    # is_replace = unlabeled_test_size > len(u_test)
                    # chosen_u_test = resample(u_test, n_samples=unlabeled_test_size, replace=is_replace, random_state=random_state)
                else:
                    chosen_u_train = u_train
                    # chosen_u_test = u_test

                filtered_u_train_idx = unlabeled_ds.iloc[chosen_u_train]
                filtered_u_test_idx = [] #unlabeled_ds.iloc[chosen_u_test]

            self.__cv_folds_idx.append((chosen_l_train, chosen_l_test, chosen_u_train, []))

            self.__ds_chunks.append((filtered_l_train_idx,   filtered_l_test_idx,
                                     filtered_u_train_idx, []))

        self.__folds_iter = iter(self.__ds_chunks)

    def _sample_labeled_data(self, data, targets, target_col, data_per_class, random_state):
        labeled_targets = list(set(targets.tolist()))
        chosen_data = []
        for lt in labeled_targets:
            filtered_rows = data[data[target_col] == lt]
            filtered_rows_idx = filtered_rows.index
            replace = data_per_class > len(filtered_rows_idx)
            chosen_idx_by_target = resample(filtered_rows_idx, n_samples=data_per_class,
                                                replace=replace, random_state=random_state)
            chosen_data += chosen_idx_by_target.tolist()
        filtered_idx = data.loc[chosen_data]
        return filtered_idx, chosen_data

    def _sample_unlabeled_data(self, unlabeled_ds, u_train, unlabeled_target_col, u_train_target, data_per_class, random_state, replace=False):
        u_train_target = unlabeled_ds.iloc[u_train][unlabeled_target_col]
        u_train_data = unlabeled_ds.iloc[u_train]
        ideal_labeled_targets = list(set(u_train_target.tolist()))
        chosen_u_train = []
        for lt in ideal_labeled_targets:
            filtered_rows = u_train_data[u_train_data[unlabeled_target_col] == lt]
            filtered_rows_idx = filtered_rows.index
            replace = data_per_class > len(filtered_rows_idx)
            chosen_u_train_by_target = resample(filtered_rows_idx, n_samples=data_per_class,
                                                replace=replace, random_state=random_state)
            chosen_u_train += chosen_u_train_by_target.tolist()
        filtered_u_train_idx = u_train_data.loc[chosen_u_train]
        return filtered_u_train_idx, chosen_u_train

    def _sampling(self, l_train_data, l_train_target, target_col, labeled_train_size_per_class, random_state):
        labeled_targets = list(set(l_train_target.tolist()))
        chosen_l_train = []
        for lt in labeled_targets:
            filtered_rows = l_train_data[l_train_data[target_col] == lt]
            filtered_rows_idx = filtered_rows.index
            chosen_l_train_by_target = resample(filtered_rows_idx, n_samples=labeled_train_size_per_class, replace=True,
                                                random_state=random_state)
            chosen_l_train += chosen_l_train_by_target.tolist()
        filtered_l_train_idx = l_train_data.loc[chosen_l_train]
        return chosen_l_train, filtered_l_train_idx

    def dump(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self.__ds_chunks, f, pickle.HIGHEST_PROTOCOL)

    def __next__(self):
        return next(self.__folds_iter)

    def __iter__(self):
        return self

    def fold(self, i):
        return self.__ds_chunks[i]

    def n_folds(self):
        return len(self.__cv_folds_idx)

    def fold_idx(self, i):
        return self.__cv_folds_idx[i]


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

    #seeds: 80122, 66371, 39333, 67462, 77665

    list_labeled_data = [50, 100, 500, 1000, 1500, 2000, 3000]

    n_folds = 5
    # u_ts = range(7)
    limit_ts = 6
    list_unlabeled_target_col = [None]

    is_first = True

    n_classes = 5

    if args.seed < 0:
        seed = random.randint(0, 100000)
    else:
        seed = args.seed

    max_ss_folds = 5

    ds = load_oai_most_datasets(root=args.root_db, img_dir=args.root_img, save_meta_dir=args.save_meta_dir,
                                saved_patch_dir=args.save_img_dir,
                                force_reload=args.reload_data and is_first,
                                output_filename=args.output_file,
                                force_rewrite=args.reload_data and is_first)
    ds["KL"] = ds["KL"].astype(int)
    ds["ID"] = ds["ID"].astype(str)
    ds = ds[ds["dataset"] == "oai"]

    total_samples = len(ds.index)

    for u_target_col in list_unlabeled_target_col:
        for n_l in list_labeled_data:
            n_ss_folds = None
            max_ts = None
            found_optimal = False
            for _max_ts in range(limit_ts, 0, -1):
                total_labels = n_l * n_classes
                total_max_unlabels = total_labels * (_max_ts - 1)
                print(
                    f'> Total sample: {total_samples}, NEED {total_labels} L, {total_max_unlabels} U, max ts: {_max_ts}')
                for _n_ss_folds in range(max_ss_folds, 1, -1):
                    n_labels_per_fold = (n_folds - 1) * total_samples / _n_ss_folds / n_folds
                    n_unlabels_per_fold = (_n_ss_folds - 1) * total_samples / _n_ss_folds
                    print(f'If ss-split {_n_ss_folds}, HAVE {n_labels_per_fold} L and {n_unlabels_per_fold} U')
                    if total_labels < n_labels_per_fold and total_max_unlabels < n_unlabels_per_fold:
                        n_ss_folds = _n_ss_folds
                        max_ts = max_ts
                        found_optimal = True
                        break

                if found_optimal:
                    break

            if not found_optimal:
                raise ValueError(
                    f'Not found an optimal n_ss_folds for {total_labels} labels and max of {total_max_unlabels}!')

            print(f'>>> Found n_ss_folds is {n_ss_folds}, max_ts is {_max_ts}')

            for t in range(_max_ts + 1):
                n_u = n_l * t
                print("Storing data of unlabeled_target_col: {}, n_labeled_data: {}, n_unlabeled_data: {}...".format(
                    u_target_col, n_l, n_u))

                is_first = False

                # Split labeled and unlabeled parts
                # Split 5 folds
                splitter_train = SSFoldSplit(ds, n_ss_folds=n_ss_folds, n_folds=n_folds, target_col="KL",
                                             random_state=seed,
                                             labeled_train_size_per_class=n_l,
                                             unlabeled_train_size=5 * n_u, group_col='ID',
                                             equal_target=True, equal_unlabeled_target=False,
                                             shuffle=True)
                file_name = f"cv_split_{n_folds}fold_l_{n_l}_u_{n_u}_False_col_None_{seed}.pkl"
                out_fullname = os.path.join(args.save_meta_dir, file_name)
                print(f'Writing file {out_fullname}')
                splitter_train.dump(out_fullname)
