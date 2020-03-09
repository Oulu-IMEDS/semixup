import argparse

import numpy as np
import torch


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default="./data/MOST_OAI_00_0_2_cropped", help='Root directory of images')
    parser.add_argument('--n_epochs', type=int, default=300, help='Number of epochs')
    parser.add_argument('--bs', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--wd', type=float, default=0, help='Weight decay')
    parser.add_argument('--beta1', type=float, default=0.9, help='Weight decay')
    parser.add_argument('--n_features', type=int, default=32, help='Number of features')
    parser.add_argument('--num_threads', type=int, default=4, help='Number of threads for data loader')
    parser.add_argument('--save_data', default='data', help='Where to save downloaded dataset')
    parser.add_argument('--seed', type=int, default=12345, help='Random seed')
    parser.add_argument('--n_classes', type=int, default=5, help='Num of classes')
    parser.add_argument('--device', type=str, default="cuda", help='Use `cuda` or `cpu`')
    parser.add_argument('--logdir', type=str, default=None, help='Log directory')
    parser.add_argument('--comment', type=str, default="pimodel", help='Comment')
    parser.add_argument('--removed_losses', type=str, default="", help='Loss indice to remove')
    parser.add_argument('--drop_rate', type=float, default=0.35, help='Dropout rate')
    parser.add_argument('--fold_index', type=int, default=1, help='Fold index')
    parser.add_argument('--kfold_split_file', type=str, default=None, help='K-fold split file')
    parser.add_argument('--model_name', type=str, default=None, help='Model name')
    parser.add_argument('--pretrained_model', type=str, default=None, help='Pretrained model path')
    parser.add_argument('--n_labels', type=int, default=1000, help='Num of labeled data')
    parser.add_argument('--n_unlabels', type=int, default=2000, help='Num of unlabeled data')
    parser.add_argument('--n_training_batches', type=int, default=-1,
                        help='Num of training batches, if -1, auto computed')
    parser.add_argument('--equal_unlabels', action='store_true', help='Whether equally sample unlabeled data')
    parser.add_argument('--unlabeled_target_column', type=str, default="5means_classes",
                        help='Unlabeled target column name')
    parser.add_argument('--model_dir', type=str, default="saved_models", help='Saved model directory')

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    return args