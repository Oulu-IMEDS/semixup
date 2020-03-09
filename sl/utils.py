import torch
import numpy as np
import cv2
import os

import solt.data as sld
import solt.core as slc
import solt.transforms as slt

from collagen.core.utils import to_cpu
from collagen.data.utils import ApplyTransform, Normalize, Compose

STD_SZ = (128, 128)


def parse_class(x):
    if len(x.shape) == 1:
        return to_cpu(x, use_numpy=True)
    elif len(x.shape) == 2:
        return to_cpu(x.argmax(dim=1), use_numpy=True)
    else:
        raise ValueError('Invalid tensor shape {}'.format(x.shape))


def parse_item(root, entry, trf, data_key, target_key):
    img1 = cv2.imread(os.path.join(root, entry["Patch1_name"]), cv2.IMREAD_COLOR)
    img2 = cv2.imread(os.path.join(root, entry["Patch2_name"]), cv2.IMREAD_COLOR)
    img1, target = trf((img1, entry["KL"]))
    img2, _ = trf((img2, entry["KL"]))
    img = np.concatenate((img1, img2), axis=0)
    return {'data': img, 'target': int(target)}


def wrap2solt(inp):
    img, label = inp
    img = cv2.resize(img, dsize=STD_SZ, interpolation=cv2.INTER_AREA)
    return sld.DataContainer((img, label), 'IL')


def unpack_solt(dc: sld.DataContainer):
    img, target = dc.data

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.expand_dims(img, axis=0)
    img, target = torch.from_numpy(img).float(), target
    return img / 255.0, np.float32(target)


def init_transforms():
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
        ApplyTransform(Normalize((0.5,), (0.5,)))
    ])

    test_trf = Compose([
        wrap2solt,
        unpack_solt,
        ApplyTransform(Normalize((0.5,), (0.5,)))
    ])

    return {"train": train_trf, "eval": test_trf}
