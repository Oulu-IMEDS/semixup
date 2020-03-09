import torch
from torch import Tensor
import numpy as np
import cv2
import os

import solt.data as sld
from collagen.core.utils import to_cpu
from collagen.data.utils import ApplyTransform, Normalize, Compose

import solt.core as slc
import solt.transforms as slt

STD_SZ = (128, 128)


def cond_accuracy_meter(target, output):
    return True


def parse_target(target):
    if target['name'] == 'l_eval':
        return target['target']
    elif target['name'] == 'train_mixmatch':
        return target['target_x']
    else:
        return None


def parse_output(output):
    if isinstance(output, Tensor):
        return output
    elif isinstance(output, dict):
        return output['x_mix']
    else:
        raise ValueError('Not support output type {}'.format(type(output)))


def parse_output_cls(y):
    if isinstance(y, dict):
        y = y['x_mix']
    elif isinstance(y, Tensor):
        pass
    else:
        return None

    if y is None:
        return None
    elif len(y.shape) == 2:
        output_cpu = to_cpu(y.argmax(dim=1), use_numpy=True)
    elif len(y.shape) == 1:
        output_cpu = to_cpu(y, use_numpy=True)
    else:
        raise ValueError("Only support dims 1 or 2, but got {}".format(len(y.shape)))
    output_cpu = output_cpu.astype(int)
    return output_cpu


def parse_target_cls(y):
    if y['name'] == 'train_mixmatch':
        y = y['target_x']
    elif y['name'] == 'l_eval':
        y = y['target']
    else:
        return None

    if y is None:
        return None
    elif len(y.shape) == 2:
        output_cpu = to_cpu(y.argmax(dim=1), use_numpy=True)
    elif len(y.shape) == 1:
        output_cpu = to_cpu(y, use_numpy=True)
    else:
        raise ValueError("Only support dims 1 or 2, but got {}".format(len(y.shape)))
    output_cpu = output_cpu.astype(int)
    return output_cpu


def parse_target_accuracy_meter(target):
    if target['name'] == 'l':
        return target['target']
    else:
        return None


def wrap2solt(inp):
    img, label = inp
    if img.shape != STD_SZ:
        img = np.expand_dims(cv2.resize(img[:, :, 0], dsize=STD_SZ, interpolation=cv2.INTER_AREA), axis=-1)
    return sld.DataContainer((img, label), 'IL')


def unpack_solt(dc: sld.DataContainer):
    img, target = dc.data
    if len(img.shape) == 3 and img.shape[2] == 1:
        img = np.transpose(img, (2, 0, 1))
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=0)
    img, target = torch.from_numpy(img).float(), target
    return img / 255.0, np.float32(target)


def custom_augment(img):
    if len(img.shape) == 3:
        imgs = np.expand_dims(img, axis=0)
    elif len(img.shape) == 4:
        imgs = img
    else:
        raise ValueError('Expect num of dims 3 or 4, but got {}'.format(len(img.shape)))

    out_imgs = []
    for b in range(imgs.shape[0]):
        img1 = imgs[b, :, :, 0:1].astype(np.uint8)
        img2 = imgs[b, :, :, 1:2].astype(np.uint8)
        tr = Compose([
            wrap2solt,
            slc.Stream([
                slt.ImageAdditiveGaussianNoise(p=0.5, gain_range=0.3),
                slt.RandomRotate(p=1, rotation_range=(-10, 10)),
                slt.PadTransform(pad_to=int(STD_SZ[0] * 1.05)),
                slt.CropTransform(crop_size=STD_SZ[0], crop_mode='r'),
                slt.ImageGammaCorrection(p=0.5, gamma_range=(0.5, 1.5)),
            ]),
            unpack_solt,
            ApplyTransform(Normalize((0.5,), (0.5,)))
        ])

        img1, _ = tr((img1, 0))
        img2, _ = tr((img2, 0))

        out_img = torch.cat((img1, img2), dim=0)
        out_imgs.append(out_img)
    out_imgs = torch.stack(out_imgs, dim=0)
    return out_imgs


def init_transforms(nc=1):
    if nc == 1:
        norm_mean_std = Normalize((0.5,), (0.5,))
    elif nc == 3:
        norm_mean_std = Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    else:
        raise ValueError("Not support channels of {}".format(nc))

    train_trf = Compose([
        wrap2solt,
        slc.Stream([
            slt.ImageAdditiveGaussianNoise(p=0.5, gain_range=0.3),
            slt.RandomRotate(p=1, rotation_range=(-10, 10)),
            slt.PadTransform(pad_to=int(STD_SZ[0] * 1.05)),
            slt.CropTransform(crop_size=STD_SZ[0], crop_mode='r'),
            slt.ImageGammaCorrection(p=0.5, gamma_range=(0.5, 1.5)),
        ]),
        unpack_solt,
        ApplyTransform(norm_mean_std)
    ])

    test_trf = Compose([
        wrap2solt,
        unpack_solt,
        ApplyTransform(norm_mean_std)
    ])

    return train_trf, test_trf, custom_augment


def init_transform_wo_aug(nc=1):
    if nc == 1:
        norm_mean_std = Normalize((0.5,), (0.5,))
    elif nc == 3:
        norm_mean_std = Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    else:
        raise ValueError("Not support channels of {}".format(nc))

    trf = Compose([
        wrap2solt,
        unpack_solt,
        ApplyTransform(norm_mean_std)
    ])
    return trf


def parse_item(root, entry, trf, data_key, target_key):
    img1 = cv2.imread(os.path.join(root, entry["Patch1_name"]), cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(os.path.join(root, entry["Patch2_name"]), cv2.IMREAD_GRAYSCALE)
    img2 = img2[:, ::-1]

    img1 = np.expand_dims(img1, axis=-1)
    img2 = np.expand_dims(img2, axis=-1)

    img1, target = trf((img1, entry["KL"]))
    img2, _ = trf((img2, entry["KL"]))
    img = torch.cat((img1, img2), dim=0)

    return {'data': img, 'target': target}
