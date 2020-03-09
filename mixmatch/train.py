from torch import optim
import torch
from tensorboardX import SummaryWriter
import yaml
import pickle
import os
import numpy as np

from collagen.core.utils import auto_detect_device
from collagen.data import SSFoldSplit
from collagen.strategies import Strategy
from collagen.callbacks.visualizer import ConfusionMatrixVisualizer
from collagen.metrics import RunningAverageMeter, BalancedAccuracyMeter, KappaMeter
from collagen.logging import MeterLogging
from collagen.savers import ModelSaver

from common.oai_most import load_oai_most_datasets
from common.networks import make_model
from common import init_args

from mixmatch.utils import parse_item, init_transforms
from mixmatch.utils import cond_accuracy_meter, parse_target, parse_output, parse_target_cls, parse_output_cls
from mixmatch.losses import MixMatchModelLoss
from mixmatch.data_provider import mixmatch_data_provider

device = auto_detect_device()


def interpolation_coef():
    return np.random.beta(0.75, 0.75)


def data_rearrange(x, y):
    batch_size = x.shape[0]
    index = torch.randperm(batch_size).to(x.device)
    return x[index, :, :, :], y[index]


if __name__ == "__main__":
    args = init_args()
    logdir = args.logdir
    comment = args.comment

    n_channels = 1

    print(args)

    if not os.path.exists(args.kfold_split_file):
        ds = load_oai_most_datasets(root=args.root_db, img_dir=args.root, save_meta_dir=args.save_meta_dir,
                                    saved_patch_dir=args.root_db, force_reload=args.reload_data,
                                    output_filename=args.meta_file, force_rewrite=args.reload_data)

        ds["KL"] = ds["KL"].astype(int)
        ds_oai = ds[ds["dataset"] == "oai"]

        # Data provider
        n_folds = 5

        splitter_train = SSFoldSplit(ds_oai, n_ss_folds=3, n_folds=n_folds, target_col="KL", random_state=args.seed,
                                     labeled_train_size_per_class=args.n_labels,
                                     unlabeled_train_size_per_class=args.n_unlabels,
                                     equal_target=True, equal_unlabeled_target=args.equal_unlabels,
                                     unlabeled_target_col=args.unlabeled_target_column, shuffle=True)

        splitter_train.dump(os.path.join(args.save_meta_dir,
                                         f"cv_split_{n_folds}fold_l{args.n_labels}_u{args.n_unlabels}_{args.equal_unlabels}_col_{args.unlabeled_target_column}.pkl"))
    else:
        print('Loading pkl file {}'.format(args.kfold_split_file))
        with open(args.kfold_split_file, 'rb') as f:
            splitter_train = iter(pickle.load(f))

    print('Processing fold {}...\n'.format(args.fold_index))

    # Initializing Discriminator
    model = make_model(nc=1, ndf=args.n_features, drop_rate=args.drop_rate, model_name=args.model_name).to(device)
    crit = MixMatchModelLoss(w_u=10.0).to(device)

    for i in range(args.fold_index):
        train_labeled_data, val_labeled_data, train_unlabeled_data, val_unlabeled_data = next(splitter_train)

    if args.n_unlabels == 0:
        train_unlabeled_data = train_labeled_data.copy(deep=True)

    data_provider = mixmatch_data_provider(root=args.root, labeled_meta_data=train_labeled_data, parse_item=parse_item,
                                           unlabeled_meta_data=train_unlabeled_data, bs=args.bs, model=model,
                                           augmentation=init_transforms(nc=n_channels)[2], n_augmentations=2,
                                           num_threads=args.num_threads, val_labeled_data=val_labeled_data,
                                           transforms=init_transforms(nc=n_channels))

    optim = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd, betas=(args.beta1, 0.999))

    summary_writer = SummaryWriter(logdir=logdir, comment=comment)
    model_dir = os.path.join(summary_writer.logdir, args.model_dir)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    # Callbacks
    callbacks_train = (RunningAverageMeter(prefix='train', name='loss_x'),
                       RunningAverageMeter(prefix='train', name='loss_u'),
                       MeterLogging(writer=summary_writer),
                       BalancedAccuracyMeter(prefix="train", name="acc", parse_target=parse_target,
                                             parse_output=parse_output,
                                             cond=cond_accuracy_meter),
                       KappaMeter(prefix='train', name='kappa', parse_target=parse_target_cls,
                                  parse_output=parse_output_cls,
                                  cond=cond_accuracy_meter),
                       ConfusionMatrixVisualizer(writer=summary_writer, cond=cond_accuracy_meter,
                                                 parse_target=parse_target_cls, parse_output=parse_output_cls,
                                                 labels=[f'KL{i}' for i in range(5)],
                                                 tag="train/confusion_matrix"))

    callbacks_eval = (RunningAverageMeter(prefix='eval', name='loss_x'),
                      RunningAverageMeter(prefix='eval', name='loss_u'),
                      BalancedAccuracyMeter(prefix="eval", name="acc", parse_target=parse_target,
                                            parse_output=parse_output,
                                            cond=cond_accuracy_meter),
                      KappaMeter(prefix='eval', name='kappa', parse_target=parse_target_cls,
                                 parse_output=parse_output_cls,
                                 cond=cond_accuracy_meter),
                      MeterLogging(writer=summary_writer),
                      ModelSaver(metric_names="eval/loss_cls", conditions='min', model=model, save_dir=model_dir),
                      ModelSaver(metric_names=("eval/kappa", "eval/acc"), conditions=('max', 'max'),
                                 model=model, save_dir=model_dir, mode="avg"),
                      ModelSaver(metric_names="eval/kappa", conditions='max', model=model, save_dir=model_dir),
                      ModelSaver(metric_names="eval/acc", conditions='max', model=model, save_dir=model_dir),
                      ConfusionMatrixVisualizer(writer=summary_writer, cond=cond_accuracy_meter,
                                                parse_target=parse_target_cls, parse_output=parse_output_cls,
                                                labels=[f'KL{i}' for i in range(5)],
                                                tag="eval/confusion_matrix"))

    st_callbacks = MeterLogging(writer=summary_writer)

    with open("settings.yml", "r") as f:
        sampling_config = yaml.load(f)

    mixmatch = Strategy(data_provider=data_provider,
                        train_loader_names=tuple(sampling_config["train"]["data_provider"].keys()),
                        val_loader_names=tuple(sampling_config["eval"]["data_provider"].keys()),
                        data_sampling_config=sampling_config,
                        loss=crit,
                        model=model,
                        n_epochs=args.n_epochs,
                        optimizer=optim,
                        train_callbacks=callbacks_train,
                        val_callbacks=callbacks_eval,
                        n_training_batches=None if args.n_training_batches < 1 else args.n_training_batches,
                        device=device)

    mixmatch.run()
