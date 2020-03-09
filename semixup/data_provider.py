import torch.nn as nn
import torch
from torch.utils.data.dataloader import default_collate
import pandas as pd

from collagen.core.utils import to_cpu
from collagen.data import ItemLoader, MixUpSampler, DataProvider


class SemixupSampler(ItemLoader):
    def __init__(self, name: str, alpha: callable or float, model: nn.Module or None,
                 data_rearrange: callable or None = None, target_rearrage: callable or None = None,
                 data_key: str = "data", target_key: str = 'target', augmentation: callable or None = None,
                 parse_item_cb: callable or None = None, meta_data: pd.DataFrame or None = None,
                 min_lambda: float = 0.7,
                 root: str or None = None, batch_size: int = 1, num_workers: int = 0, shuffle: bool = False,
                 pin_memory: bool = False, collate_fn: callable = default_collate, transform: callable or None = None,
                 sampler: torch.utils.data.sampler.Sampler or None = None, batch_sampler=None,
                 drop_last: bool = False, timeout: int = 0):
        super().__init__(meta_data=meta_data, parse_item_cb=parse_item_cb, root=root, batch_size=batch_size,
                         num_workers=num_workers, shuffle=shuffle, pin_memory=pin_memory, collate_fn=collate_fn,
                         transform=transform, sampler=sampler, batch_sampler=batch_sampler, drop_last=drop_last,
                         timeout=timeout)

        self.__model = model
        self.__name = name
        # self.__n_classes = n_classes
        self.__data_key = data_key
        self.__target_key = target_key
        self.__alpha = alpha
        self.__data_rearrange = self._default_change_data_ordering if data_rearrange is None else data_rearrange

        self.__augmentation = augmentation
        self.__min_l = min_lambda

    @property
    def alpha(self):
        return self.__alpha

    @staticmethod
    def _default_change_data_ordering(x, y):
        return torch.flip(x, dims=[0]), torch.flip(y, dims=[0])

    def __len__(self):
        return super().__len__()

    def sample(self, k=1):
        sampled_rows = super().sample(k)
        device = next(self.__model.parameters()).device
        samples = []
        for i in range(k):
            imgs1 = sampled_rows[i][self.__data_key]
            target1 = sampled_rows[i][self.__target_key]

            imgs2, target2 = self.__data_rearrange(imgs1, target1)

            if callable(self.alpha):
                l = self.alpha()
            elif isinstance(self.alpha, float):
                l = self.alpha
            else:
                raise ValueError('Not support alpha of {}'.format(type(self.alpha)))

            if not isinstance(l, float) or l < 0 or l > 1:
                raise ValueError('Alpha {} is not float or out of range [0,1]'.format(l))
            elif l < 0.5:
                l = 1 - l

            l = max(l, self.__min_l)

            mixup_imgs = l * imgs1 + (1 - l) * imgs2

            logits1 = self.__model(imgs1.to(device))
            logits2 = self.__model(imgs2.to(device))
            # mixup_logits = self.__model(mixup_imgs.to(device))

            imgs1_cpu = to_cpu(imgs1.permute(0, 2, 3, 1), use_numpy=True)
            imgs1_aug = self.__augmentation(imgs1_cpu)
            logits1_aug = self.__model(imgs1_aug.to(device))

            logits_mixup = l * logits1 + (1 - l) * logits2
            samples.append({'name': self.__name, 'mixup_data': mixup_imgs,
                            'logits': logits1, 'logits_aug': logits1_aug,
                            'logits_mixup': logits_mixup, 'alpha': l})
        return samples


def semixup_data_provider(model, alpha, n_classes, train_labeled_data, train_unlabeled_data, val_labeled_data,
                          transforms, parse_item, bs, num_threads, item_loaders=dict(), root="", augmentation=None,
                          data_rearrange=None):
    """
    Default setting of data provider for Semixup
    """
    item_loaders["labeled_train"] = MixUpSampler(meta_data=train_labeled_data, name='l_mixup', alpha=alpha, model=model,
                                                 transform=transforms[0], parse_item_cb=parse_item, batch_size=bs,
                                                 data_rearrange=data_rearrange,
                                                 num_workers=num_threads, root=root, shuffle=True)

    item_loaders["unlabeled_train"] = SemixupSampler(meta_data=train_unlabeled_data, name='u_mixup', alpha=alpha,
                                                     model=model, min_lambda=0.55,
                                                     transform=transforms[0], parse_item_cb=parse_item, batch_size=bs,
                                                     data_rearrange=data_rearrange,
                                                     num_workers=num_threads, augmentation=augmentation, root=root,
                                                     shuffle=True)

    item_loaders["labeled_eval"] = ItemLoader(meta_data=val_labeled_data, name='l_norm', transform=transforms[1],
                                              parse_item_cb=parse_item, batch_size=bs, num_workers=num_threads,
                                              root=root, shuffle=False)

    return DataProvider(item_loaders)
