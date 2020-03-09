import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import default_collate
import pandas as pd
import numpy as np

from collagen.data import ItemLoader, DataProvider
from collagen.core.utils import to_cpu


class MixMatchSampler(object):
    def __init__(self, model: nn.Module, name: str, augmentation, labeled_meta_data: pd.DataFrame,
                 unlabeled_meta_data: pd.DataFrame,
                 n_augmentations=1, output_type='logits', data_key: str = "data", target_key: str = 'target',
                 parse_item_cb: callable or None = None,
                 root: str or None = None, batch_size: int = 1, num_workers: int = 0, shuffle: bool = False,
                 pin_memory: bool = False, collate_fn: callable = default_collate, transform: callable or None = None,
                 sampler: torch.utils.data.sampler.Sampler or None = None, batch_sampler=None,
                 drop_last: bool = False, timeout: int = 0):
        self._label_sampler = ItemLoader(meta_data=labeled_meta_data, parse_item_cb=parse_item_cb, root=root,
                                         batch_size=batch_size,
                                         num_workers=num_workers, shuffle=shuffle, pin_memory=pin_memory,
                                         collate_fn=collate_fn,
                                         transform=transform, sampler=sampler, batch_sampler=batch_sampler,
                                         drop_last=drop_last, timeout=timeout)

        self._unlabel_sampler = ItemLoader(meta_data=unlabeled_meta_data,
                                           parse_item_cb=parse_item_cb, root=root,
                                           batch_size=batch_size,
                                           num_workers=num_workers, shuffle=shuffle,
                                           pin_memory=pin_memory, collate_fn=collate_fn,
                                           transform=transform, sampler=sampler,
                                           batch_sampler=batch_sampler, drop_last=drop_last,
                                           timeout=timeout)

        self._name = name
        self._model: nn.Module = model
        self._n_augmentations = n_augmentations
        self._augmentation = augmentation
        self._data_key = data_key
        self._target_key = target_key
        self._output_type = output_type

        self._len = max(len(self._label_sampler), len(self._unlabel_sampler))

    def __len__(self):
        return self._len

    def _crop_if_needed(self, df1, df2):
        assert len(df1) == len(df2)
        for i in range(len(df1)):
            if len(df1[i]['data']) != len(df2[i]['data']):
                min_len = min(len(df1[i]['data']), len(df2[i]['data']))
                df1[i][self._data_key] = df1[i][self._data_key][:min_len, :]
                df2[i][self._data_key] = df2[i][self._data_key][:min_len, :]
                df1[i][self._target_key] = df1[i][self._target_key][:min_len]
                df2[i][self._target_key] = df2[i][self._target_key][:min_len]
        return df1, df2

    def sharpen(self, x, T=0.5):
        assert len(x.shape) == 2

        _x = torch.pow(x, 1 / T)
        s = torch.sum(_x, dim=-1, keepdim=True)
        _x = _x / s
        return _x

    def _create_union_data(self, r1, r2):
        assert len(r1) == len(r2)
        r = []

        for i in range(len(r1)):
            union_rows = dict()
            union_rows[self._data_key] = torch.cat([r1[i][self._data_key], r2[i][self._data_key]], dim=0)
            union_rows["probs"] = torch.cat([r1[i]["probs"], r2[i]["probs"]], dim=0)
            union_rows['name'] = r1[i]['name']
            r.append(union_rows)
        return r

    def _mixup(self, x1, y1, x2, y2, alpha=0.75):
        l = np.random.beta(alpha, alpha)
        l = max(l, 1 - l)
        x = l * x1 + (1 - l) * x2
        y = l * y1 + (1 - l) * y2
        return x, y

    def sample(self, k=1):
        samples = []
        labeled_sampled_rows = self._label_sampler.sample(k)
        unlabeled_sampled_rows = self._unlabel_sampler.sample(k)

        labeled_sampled_rows, unlabeled_sampled_rows = self._crop_if_needed(labeled_sampled_rows,
                                                                            unlabeled_sampled_rows)

        for i in range(k):
            # Unlabeled data
            unlabeled_sampled_rows[i][self._data_key] = unlabeled_sampled_rows[i][self._data_key].to(
                next(self._model.parameters()).device)

            u_imgs = unlabeled_sampled_rows[i][self._data_key]

            list_imgs = []
            for b in range(u_imgs.shape[0]):
                for j in range(self._n_augmentations):
                    img = u_imgs[b, :, :, :]
                    if img.shape[0] == 1:
                        img = img[0, :, :]
                    else:
                        img = img.permute(1, 2, 0)

                    img_cpu = to_cpu(img)
                    aug_img = self._augmentation(img_cpu)
                    list_imgs.append(aug_img)

            batch_imgs = torch.cat(list_imgs, dim=0)
            batch_imgs = batch_imgs.to(next(self._model.parameters()).device)
            if self._output_type == 'logits':
                out = self._model(batch_imgs)
            elif self._output_type == 'features':
                out = self._model.get_features(batch_imgs)

            preds = F.softmax(out, dim=1)
            preds = preds.view(u_imgs.shape[0], -1, preds.shape[-1])

            mean_preds = torch.mean(preds, dim=1)
            guessing_labels = self.sharpen(mean_preds).detach()

            unlabeled_sampled_rows[i]["probs"] = guessing_labels

            # Labeled data
            labeled_sampled_rows[i][self._data_key] = labeled_sampled_rows[i][self._data_key].to(
                next(self._model.parameters()).device)
            target_l = labeled_sampled_rows[i][self._target_key]
            onehot_l = torch.zeros(guessing_labels.shape)
            onehot_l.scatter_(1, target_l.type(torch.int64).unsqueeze(-1), 1.0)
            labeled_sampled_rows[i]["probs"] = onehot_l.to(next(self._model.parameters()).device)

        union_rows = self._create_union_data(labeled_sampled_rows, unlabeled_sampled_rows)

        for i in range(k):
            ridx = np.random.permutation(union_rows[i][self._data_key].shape[0])
            u = unlabeled_sampled_rows[i]
            x = labeled_sampled_rows[i]

            x_mix, target_mix = self._mixup(x[self._data_key], x["probs"],
                                            union_rows[i][self._data_key][ridx[i]], union_rows[i]["probs"][ridx[i]])
            u_mix, pred_mix = self._mixup(u[self._data_key], u["probs"], union_rows[i][self._data_key][ridx[k + i]],
                                          union_rows[i]["probs"][ridx[k + i]])

            samples.append({'name': self._name, 'x_mix': x_mix, 'target_mix_x': target_mix, 'u_mix': u_mix,
                            'target_mix_u': pred_mix, 'target_x': x[self._target_key]})
        return samples


def mixmatch_data_provider(model, augmentation, labeled_meta_data, unlabeled_meta_data, val_labeled_data,
                           n_augmentations, parse_item, bs, transforms, root="", num_threads=4):
    itemloader_dict = {}
    itemloader_dict['all_train'] = MixMatchSampler(model=model, name="train_mixmatch", augmentation=augmentation,
                                                   labeled_meta_data=labeled_meta_data, root=root,
                                                   unlabeled_meta_data=unlabeled_meta_data,
                                                   n_augmentations=n_augmentations,
                                                   data_key='data', target_key='target', parse_item_cb=parse_item,
                                                   batch_size=bs, transform=transforms[0],
                                                   num_workers=num_threads, shuffle=True)

    itemloader_dict['labeled_eval'] = ItemLoader(root=root, meta_data=val_labeled_data, name='l_eval',
                                                 transform=transforms[1],
                                                 parse_item_cb=parse_item,
                                                 batch_size=bs, num_workers=num_threads,
                                                 shuffle=False)
    return DataProvider(itemloader_dict)
