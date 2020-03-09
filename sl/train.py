import torch
import numpy as np
import yaml
import os
from tensorboardX import SummaryWriter
import pickle

from collagen.data import ItemLoader, DataProvider
from collagen.core.utils import auto_detect_device
from collagen.strategies import Strategy
from collagen.metrics import RunningAverageMeter, BalancedAccuracyMeter, KappaMeter
from collagen.callbacks import ProgressbarVisualizer
from collagen.savers import ModelSaver
from collagen.logging import MeterLogging

from common.networks import make_model

from common import init_args

from sl.utils import parse_class
from sl.utils import init_transforms, parse_item

device = auto_detect_device()

kfold_metrics = dict()


def collect_metrics(cbs):
    for cb in cbs:
        if cb.ctype == "saver":
            for metric_name in cb.metric_names:
                best_metric = cb.get_metric_by_name(metric_name)
                if best_metric is not None:
                    if metric_name not in kfold_metrics:
                        kfold_metrics[metric_name] = []
                    kfold_metrics[metric_name].append(best_metric)


def print_metrics():
    for meter_name in kfold_metrics:
        kfold_val = np.asarray(kfold_metrics[meter_name]).mean()
        print("K-fold metric {}: {}".format(meter_name, kfold_val))


if __name__ == "__main__":
    args = init_args()

    print('Loading pkl file {}'.format(args.kfold_split_file))
    with open(args.kfold_split_file, 'rb') as f:
        splitter = iter(pickle.load(f))

    criterion = torch.nn.CrossEntropyLoss()

    # Tensorboard visualization
    logdir = args.logdir
    comment = args.comment

    kfold_train_losses = []
    kfold_val_losses = []
    kfold__val_accuracies = []

    with open("settings.yml", "r") as f:
        sampling_config = yaml.load(f)


    for fold_id, df in enumerate(splitter):
        df_train = df[0]
        df_val = df[1]

        print("Fold {} on {} labeled samples...".format(fold_id, len(df_train.index)))
        item_loaders = dict()

        # Data provider
        for stage, df in zip(['train', 'eval'], [df_train, df_val]):
            item_loaders[f'data_{stage}'] = ItemLoader(root=args.root,
                                                       meta_data=df,
                                                       transform=init_transforms()[stage],
                                                       parse_item_cb=parse_item,
                                                       batch_size=args.bs, num_workers=args.num_threads,
                                                       shuffle=True if stage == "train" else False)
        data_provider = DataProvider(item_loaders)

        # Visualizers
        summary_writer = SummaryWriter(logdir=logdir, comment=comment + "_fold" + str(fold_id + 1))
        model_dir = os.path.join(summary_writer.logdir, args.model_dir)
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)

        # Model
        model = make_model(model_name=args.model_name, nc=1, ndf=32, n_cls=5, drop_rate=args.drop_rate).to(device)

        # Callbacks
        train_cbs = (RunningAverageMeter(prefix="train", name="loss"),
                     ProgressbarVisualizer(update_freq=1),
                     BalancedAccuracyMeter(prefix="train", name="acc"),
                     KappaMeter(prefix="train", name="kappa", parse_target=parse_class, parse_output=parse_class))
        val_cbs = (RunningAverageMeter(prefix="eval", name="loss"),
                   BalancedAccuracyMeter(prefix="eval", name="acc"),
                   KappaMeter(prefix="eval", name="kappa", parse_target=parse_class, parse_output=parse_class),
                   ProgressbarVisualizer(update_freq=1),
                   MeterLogging(writer=summary_writer),
                   ModelSaver(metric_names="eval/loss", conditions='min', model=model, save_dir=model_dir),
                   ModelSaver(metric_names="eval/acc", conditions='max', model=model, save_dir=model_dir),
                   ModelSaver(metric_names="eval/kappa", conditions='max', model=model, save_dir=model_dir),
                   ModelSaver(metric_names=("eval/kappa", "eval/acc"), conditions=('max', 'max'),
                              model=model, save_dir=model_dir, mode="avg"))

        optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.wd)

        strategy = Strategy(data_provider=data_provider, n_epochs=args.n_epochs, optimizer=optimizer,
                            train_loader_names=tuple(sampling_config["train"]['data_provider'].keys()),
                            val_loader_names=tuple(sampling_config["eval"]['data_provider'].keys()),
                            data_sampling_config=sampling_config,
                            loss=criterion, model=model, device=device,
                            train_callbacks=train_cbs, val_callbacks=val_cbs)

        strategy.run()

        collect_metrics(val_cbs)

        if args.fold_index > 0 and args.fold_index == fold_id + 1:
            break

    print_metrics()
