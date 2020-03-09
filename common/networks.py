import torch
import torch.nn as nn
from torch.tensor import OrderedDict
from collagen.core import Module


def make_model(model_name, nc, ndf, drop_rate=0.5, n_cls=5):
    print('Use model {}'.format(model_name))
    if model_name == "cvgg2vh":
        model = DisCustomVGG2VH(nc=nc, ndf=ndf, drop_rate=drop_rate, n_cls=n_cls)
    elif model_name == "cvgg2hv":
        model = DisCustomVGG2HV(nc=nc, ndf=ndf, drop_rate=drop_rate, n_cls=n_cls)
    elif model_name == "cvgg2gap":
        model = DisCustomVGG2GAP(nc=nc, ndf=ndf, drop_rate=drop_rate, n_cls=n_cls)

    elif model_name == 'alekseiori':
        model = DisAleksei(nc=nc, ndf=ndf, drop_rate=drop_rate, n_cls=n_cls, use_bn=True)
    elif model_name == 'alekseivh':
        model = DisAlekseiVH(nc=nc, ndf=ndf, drop_rate=drop_rate, n_cls=n_cls, use_bn=True)
    elif model_name == 'alekseihv':
        model = DisAlekseiHV(nc=nc, ndf=ndf, drop_rate=drop_rate, n_cls=n_cls, use_bn=True)
    else:
        raise ValueError('Not support model {}'.format(model_name))
    return model


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def aleksei_weights_init(m):
    """
    Initializes the weights using kaiming method.
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_uniform(m.weight.data)
        # m.bias.data.fill_(0)

    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform(m.weight.data)
        m.bias.data.fill_(0)


class DisCustomVGG2GAP(Module):
    def __init__(self, nc=1, ndf=32, n_cls=5, drop_rate=0.35, use_bn=False):
        super().__init__()
        # input is (nc) x 32 x 32
        self.__n_cls = n_cls
        self.__required_output_channels = self.__n_cls
        self.__drop_rate = drop_rate
        self.__use_bn = use_bn

        self.__drop = nn.Dropout(p=drop_rate)

        self._pool1 = nn.MaxPool2d(2, 2)
        self._pool2 = nn.MaxPool2d(2, 2)
        self._pool3 = nn.MaxPool2d(2, 2)
        self._pool4 = nn.MaxPool2d(2, 2)
        self._pool5 = nn.MaxPool2d(2, 2)

        # Group 1
        # input is (nc) x 128 x 128
        self._layer11 = nn.Sequential(nn.Conv2d(nc, ndf, 3, 1, 1, bias=False),
                                      nn.BatchNorm2d(ndf) if self.__use_bn else nn.InstanceNorm2d(ndf),
                                      nn.LeakyReLU(0.2, inplace=True))  # state size 128 x 128
        self._layer12 = nn.Sequential(nn.Conv2d(ndf, ndf, 3, 1, 1, bias=False),
                                      nn.BatchNorm2d(ndf) if self.__use_bn else nn.InstanceNorm2d(ndf),
                                      nn.LeakyReLU(0.2, inplace=True))  # state size 128 x 128
        self._layer13 = nn.Sequential(nn.Conv2d(ndf, ndf, 3, 1, 1, bias=False),
                                      nn.BatchNorm2d(ndf) if self.__use_bn else nn.InstanceNorm2d(ndf),
                                      nn.LeakyReLU(0.2, inplace=True))  # state size 128 x 128

        # Group 2
        self._layer21 = nn.Sequential(nn.Conv2d(ndf, ndf * 2, 3, 2, 1, bias=False),
                                      nn.BatchNorm2d(ndf * 2) if self.__use_bn else nn.InstanceNorm2d(ndf * 2),
                                      nn.LeakyReLU(0.2, inplace=True))  # state size 64 x 64

        self._layer22 = nn.Sequential(nn.Conv2d(ndf * 2, ndf * 2, 3, 1, 1, bias=False),
                                      nn.BatchNorm2d(ndf * 2) if self.__use_bn else nn.InstanceNorm2d(ndf * 2),
                                      nn.LeakyReLU(0.2, inplace=True))  # state size 64 x 64

        # Group 3
        self._layer31 = nn.Sequential(nn.Conv2d(ndf * 2, ndf * 4, 3, 2, 1, bias=False),
                                      nn.BatchNorm2d(ndf * 4) if self.__use_bn else nn.InstanceNorm2d(ndf * 4),
                                      nn.LeakyReLU(0.2, inplace=True))  # state size 32 x 32
        self._layer32 = nn.Sequential(nn.Conv2d(ndf * 4, ndf * 4, 3, 1, 1, bias=False),
                                      nn.BatchNorm2d(ndf * 4) if self.__use_bn else nn.InstanceNorm2d(ndf * 4),
                                      nn.LeakyReLU(0.2, inplace=True))  # state size 32 x 32

        # Group 4
        self._layer41 = nn.Sequential(nn.Conv2d(ndf * 4, ndf * 8, 3, 2, 1, bias=False),
                                      nn.BatchNorm2d(ndf * 8) if self.__use_bn else nn.InstanceNorm2d(ndf * 8),
                                      nn.LeakyReLU(0.2, inplace=True))  # state size 16 x 16
        self._layer42 = nn.Sequential(nn.Conv2d(ndf * 8, ndf * 8, 3, 1, 1, bias=False),
                                      nn.BatchNorm2d(ndf * 8) if self.__use_bn else nn.InstanceNorm2d(ndf * 8),
                                      nn.LeakyReLU(0.2, inplace=True))  # state size 16 x 16

        self._gap = nn.AvgPool2d(16, 16)

        # Input 128 x 128
        self.main_flow = nn.Sequential(self._layer11,
                                       self._layer12,
                                       self._layer13,  # 128 x 128
                                       self.__drop,

                                       self._layer21,
                                       self._layer22,  # 64 x 64
                                       self.__drop,

                                       self._layer31,
                                       self._layer32,  # 32 x 32
                                       self.__drop,

                                       self._layer41,
                                       self._layer42,  # 16 x 16

                                       self._gap)  # 1 x 1

        self.classify = nn.Sequential(nn.Conv2d(ndf * 16, n_cls, 1, 1, 0, bias=False))  # state size. (n_cls)x1x1

        self.apply(weights_init)

    def get_features(self, x):
        self.validate_input(x)
        t1 = int(x.shape[1] / 2)
        t2 = x.shape[1]
        x0 = x[:, 0:t1, :, :]
        x1 = x[:, t1:t2, :, :]
        x0 = self.main_flow(x0)
        x1 = self.main_flow(x1)
        output = torch.cat((x0, x1), dim=1).squeeze(-1).squeeze(-1)
        return output

    def forward(self, x):
        self.validate_input(x)
        t1 = int(x.shape[1] / 2)
        t2 = x.shape[1]

        x0 = x[:, 0:t1, :, :]
        x1 = x[:, t1:t2, :, :]
        x0 = self.main_flow(x0)
        x1 = self.main_flow(x1)
        x01 = torch.cat((x0, x1), dim=1)

        classifier = self.__drop(x01)

        output = self.classify(classifier).squeeze(-1).squeeze(-1)
        self.validate_output(output)
        return output


class DisCustomVGG2VH(Module):
    def __init__(self, nc=1, ndf=32, n_cls=5, drop_rate=0.35, use_bn=False):
        super().__init__()
        # input is (nc) x 32 x 32
        self.__n_cls = n_cls
        self.__required_output_channels = self.__n_cls
        self.__drop_rate = drop_rate
        self.__use_bn = use_bn

        self.__drop = nn.Dropout(p=drop_rate)

        self._pool1 = nn.MaxPool2d(2, 2)
        self._pool2 = nn.MaxPool2d(2, 2)
        self._pool3 = nn.MaxPool2d(2, 2)
        self._pool4 = nn.MaxPool2d(2, 2)
        self._pool5 = nn.MaxPool2d(2, 2)

        # Group 1
        # input is (nc) x 128 x 128
        self._layer11 = nn.Sequential(nn.Conv2d(nc, ndf, 3, 1, 1, bias=False),
                                      nn.BatchNorm2d(ndf) if self.__use_bn else nn.InstanceNorm2d(ndf),
                                      nn.LeakyReLU(0.2, inplace=True))  # state size 128 x 128
        self._layer12 = nn.Sequential(nn.Conv2d(ndf, ndf, 3, 1, 1, bias=False),
                                      nn.BatchNorm2d(ndf) if self.__use_bn else nn.InstanceNorm2d(ndf),
                                      nn.LeakyReLU(0.2, inplace=True))  # state size 128 x 128
        self._layer13 = nn.Sequential(nn.Conv2d(ndf, ndf, 3, 1, 1, bias=False),
                                      nn.BatchNorm2d(ndf) if self.__use_bn else nn.InstanceNorm2d(ndf),
                                      nn.LeakyReLU(0.2, inplace=True))  # state size 128 x 128

        # Group 2
        self._layer21 = nn.Sequential(nn.Conv2d(ndf, ndf * 2, 3, 2, 1, bias=False),
                                      nn.BatchNorm2d(ndf * 2) if self.__use_bn else nn.InstanceNorm2d(ndf * 2),
                                      nn.LeakyReLU(0.2, inplace=True))  # state size 64 x 64

        self._layer22 = nn.Sequential(nn.Conv2d(ndf * 2, ndf * 2, 3, 1, 1, bias=False),
                                      nn.BatchNorm2d(ndf * 2) if self.__use_bn else nn.InstanceNorm2d(ndf * 2),
                                      nn.LeakyReLU(0.2, inplace=True))  # state size 64 x 64

        # Group 3
        self._layer31 = nn.Sequential(nn.Conv2d(ndf * 2, ndf * 4, 3, 2, 1, bias=False),
                                      nn.BatchNorm2d(ndf * 4) if self.__use_bn else nn.InstanceNorm2d(ndf * 4),
                                      nn.LeakyReLU(0.2, inplace=True))  # state size 32 x 32
        self._layer32 = nn.Sequential(nn.Conv2d(ndf * 4, ndf * 4, 3, 1, 1, bias=False),
                                      nn.BatchNorm2d(ndf * 4) if self.__use_bn else nn.InstanceNorm2d(ndf * 4),
                                      nn.LeakyReLU(0.2, inplace=True))  # state size 32 x 32

        # Group 4
        self._layer41 = nn.Sequential(nn.Conv2d(ndf * 4, ndf * 8, 3, 2, 1, bias=False),
                                      nn.BatchNorm2d(ndf * 8) if self.__use_bn else nn.InstanceNorm2d(ndf * 8),
                                      nn.LeakyReLU(0.2, inplace=True))  # state size 16 x 16
        self._layer42 = nn.Sequential(nn.Conv2d(ndf * 8, ndf * 8, 3, 1, 1, bias=False),
                                      nn.BatchNorm2d(ndf * 8) if self.__use_bn else nn.InstanceNorm2d(ndf * 8),
                                      nn.LeakyReLU(0.2, inplace=True))  # state size 16 x 16

        # Pool VH
        self._pool_vh = nn.Sequential(nn.MaxPool2d((16, 1), (16, 1), 0),
                                      nn.Conv2d(ndf * 8, ndf * 8, 1, 1, 0, bias=False),
                                      nn.BatchNorm2d(ndf * 8) if self.__use_bn else nn.InstanceNorm2d(ndf * 8),
                                      nn.LeakyReLU(0.2, inplace=True),
                                      nn.MaxPool2d((1, 16), 16, 0))  # state size ndf * 4 x 1 x 1

        # Input 128 x 128
        self.main_flow = nn.Sequential(self._layer11,
                                       self._layer12,
                                       self._layer13,  # 128 x 128
                                       self.__drop,

                                       self._layer21,
                                       self._layer22,  # 64 x 64
                                       self.__drop,

                                       self._layer31,
                                       self._layer32,  # 32 x 32
                                       self.__drop,

                                       self._layer41,
                                       self._layer42,  # 16 x 16

                                       self._pool_vh)  # 1 x 1

        self.classify = nn.Sequential(nn.Conv2d(ndf * 16, n_cls, 1, 1, 0, bias=False))  # state size. (n_cls)x1x1

        self.apply(weights_init)

    def get_features(self, x):
        self.validate_input(x)
        t1 = int(x.shape[1] / 2)
        t2 = x.shape[1]
        x0 = x[:, 0:t1, :, :]
        x1 = x[:, t1:t2, :, :]
        x0 = self.main_flow(x0)
        x1 = self.main_flow(x1)
        output = torch.cat((x0, x1), dim=1).squeeze(-1).squeeze(-1)
        return output

    def forward(self, x):
        self.validate_input(x)
        t1 = int(x.shape[1] / 2)
        t2 = x.shape[1]

        x0 = x[:, 0:t1, :, :]
        x1 = x[:, t1:t2, :, :]
        x0 = self.main_flow(x0)
        x1 = self.main_flow(x1)
        x01 = torch.cat((x0, x1), dim=1)

        classifier = self.__drop(x01)

        output = self.classify(classifier).squeeze(-1).squeeze(-1)
        self.validate_output(output)
        return output


class DisCustomVGG2HV(Module):
    def __init__(self, nc=1, ndf=32, n_cls=5, drop_rate=0.35, use_bn=False):
        super().__init__()
        # input is (nc) x 32 x 32
        self.__n_cls = n_cls
        self.__required_output_channels = self.__n_cls
        self.__drop_rate = drop_rate
        self.__use_bn = use_bn

        self.__drop = nn.Dropout(p=drop_rate)

        self._pool1 = nn.MaxPool2d(2, 2)
        self._pool2 = nn.MaxPool2d(2, 2)
        self._pool3 = nn.MaxPool2d(2, 2)
        self._pool4 = nn.MaxPool2d(2, 2)
        self._pool5 = nn.MaxPool2d(2, 2)

        # Group 1
        # input is (nc) x 128 x 128
        self._layer11 = nn.Sequential(nn.Conv2d(nc, ndf, 3, 1, 1, bias=False),
                                      nn.BatchNorm2d(ndf) if self.__use_bn else nn.InstanceNorm2d(ndf),
                                      nn.LeakyReLU(0.2, inplace=True))  # state size 128 x 128
        self._layer12 = nn.Sequential(nn.Conv2d(ndf, ndf, 3, 1, 1, bias=False),
                                      nn.BatchNorm2d(ndf) if self.__use_bn else nn.InstanceNorm2d(ndf),
                                      nn.LeakyReLU(0.2, inplace=True))  # state size 128 x 128
        self._layer13 = nn.Sequential(nn.Conv2d(ndf, ndf, 3, 1, 1, bias=False),
                                      nn.BatchNorm2d(ndf) if self.__use_bn else nn.InstanceNorm2d(ndf),
                                      nn.LeakyReLU(0.2, inplace=True))  # state size 128 x 128

        # Group 2
        self._layer21 = nn.Sequential(nn.Conv2d(ndf, ndf * 2, 3, 2, 1, bias=False),
                                      nn.BatchNorm2d(ndf * 2) if self.__use_bn else nn.InstanceNorm2d(ndf * 2),
                                      nn.LeakyReLU(0.2, inplace=True))  # state size 64 x 64

        self._layer22 = nn.Sequential(nn.Conv2d(ndf * 2, ndf * 2, 3, 1, 1, bias=False),
                                      nn.BatchNorm2d(ndf * 2) if self.__use_bn else nn.InstanceNorm2d(ndf * 2),
                                      nn.LeakyReLU(0.2, inplace=True))  # state size 64 x 64

        # Group 3
        self._layer31 = nn.Sequential(nn.Conv2d(ndf * 2, ndf * 4, 3, 2, 1, bias=False),
                                      nn.BatchNorm2d(ndf * 4) if self.__use_bn else nn.InstanceNorm2d(ndf * 4),
                                      nn.LeakyReLU(0.2, inplace=True))  # state size 32 x 32
        self._layer32 = nn.Sequential(nn.Conv2d(ndf * 4, ndf * 4, 3, 1, 1, bias=False),
                                      nn.BatchNorm2d(ndf * 4) if self.__use_bn else nn.InstanceNorm2d(ndf * 4),
                                      nn.LeakyReLU(0.2, inplace=True))  # state size 32 x 32

        # Group 4
        self._layer41 = nn.Sequential(nn.Conv2d(ndf * 4, ndf * 8, 3, 2, 1, bias=False),
                                      nn.BatchNorm2d(ndf * 8) if self.__use_bn else nn.InstanceNorm2d(ndf * 8),
                                      nn.LeakyReLU(0.2, inplace=True))  # state size 16 x 16
        self._layer42 = nn.Sequential(nn.Conv2d(ndf * 8, ndf * 8, 3, 1, 1, bias=False),
                                      nn.BatchNorm2d(ndf * 8) if self.__use_bn else nn.InstanceNorm2d(ndf * 8),
                                      nn.LeakyReLU(0.2, inplace=True))  # state size 16 x 16

        # Pool HV
        self._pool_hv = nn.Sequential(nn.MaxPool2d((1, 16), (1, 16), 0),
                                      nn.Conv2d(ndf * 8, ndf * 8, 1, 1, 0, bias=False),
                                      nn.BatchNorm2d(ndf * 8) if self.__use_bn else nn.InstanceNorm2d(ndf * 8),
                                      nn.LeakyReLU(0.2, inplace=True),
                                      nn.MaxPool2d((16, 1), 16, 0))  # state size ndf * 4 x 1 x 1

        # Input 128 x 128
        self.main_flow = nn.Sequential(self._layer11,
                                       self._layer12,
                                       self._layer13,  # 128 x 128
                                       self.__drop,

                                       self._layer21,
                                       self._layer22,  # 64 x 64
                                       self.__drop,

                                       self._layer31,
                                       self._layer32,  # 32 x 32
                                       self.__drop,

                                       self._layer41,
                                       self._layer42,  # 16 x 16

                                       self._pool_hv)  # 1 x 1

        self.classify = nn.Sequential(nn.Conv2d(ndf * 16, n_cls, 1, 1, 0, bias=False))  # state size. (n_cls)x1x1

        self.apply(weights_init)

    def get_features(self, x):
        self.validate_input(x)
        t1 = int(x.shape[1] / 2)
        t2 = x.shape[1]
        x0 = x[:, 0:t1, :, :]
        x1 = x[:, t1:t2, :, :]
        x0 = self.main_flow(x0)
        x1 = self.main_flow(x1)
        output = torch.cat((x0, x1), dim=1).squeeze(-1).squeeze(-1)
        return output

    def forward(self, x):
        self.validate_input(x)
        t1 = int(x.shape[1] / 2)
        t2 = x.shape[1]

        x0 = x[:, 0:t1, :, :]
        x1 = x[:, t1:t2, :, :]
        x0 = self.main_flow(x0)
        x1 = self.main_flow(x1)
        x01 = torch.cat((x0, x1), dim=1)

        classifier = self.__drop(x01)

        output = self.classify(classifier).squeeze(-1).squeeze(-1)
        self.validate_output(output)
        return output


### Aleksei model versions

class DisAleksei(Module):
    def __init__(self, nc=1, ndf=64, n_cls=5, drop_rate=0.35, use_bn=False):
        super().__init__()
        # input is (nc) x 32 x 32
        self.__drop_rate = drop_rate
        self.__use_bn = use_bn
        self.__required_output_channels = n_cls

        self.__drop = nn.Dropout(p=self.__drop_rate)

        self._pool1 = nn.MaxPool2d(2, 2)
        self._pool2 = nn.MaxPool2d(2, 2)

        # Group 1
        # input is (nc) x 256 x 256
        self._layer11 = nn.Sequential(nn.Conv2d(nc, ndf, 3, 2, 0, bias=False),
                                      nn.BatchNorm2d(ndf, eps=1e-3) if self.__use_bn else nn.InstanceNorm2d(ndf),
                                      nn.ReLU(inplace=True))  # state size 128 x 128
        self._layer12 = nn.Sequential(nn.Conv2d(ndf, ndf, 3, 1, 0, bias=False),
                                      nn.BatchNorm2d(ndf, eps=1e-3) if self.__use_bn else nn.InstanceNorm2d(ndf),
                                      nn.ReLU(inplace=True))  # state size 128 x 128

        self._layer13 = nn.Sequential(nn.Conv2d(ndf, ndf, 3, 1, 0, bias=False),
                                      nn.BatchNorm2d(ndf, eps=1e-3) if self.__use_bn else nn.InstanceNorm2d(ndf),
                                      nn.ReLU(inplace=True))  # state size 128 x 128

        # Group 2

        self._layer21 = nn.Sequential(nn.Conv2d(ndf, ndf * 2, 3, 1, 0, bias=False),
                                      nn.BatchNorm2d(ndf * 2, eps=1e-3) if self.__use_bn else nn.InstanceNorm2d(
                                          ndf * 2),
                                      nn.ReLU(inplace=True))  # state size 64 x 64

        self._layer22 = nn.Sequential(nn.Conv2d(ndf * 2, ndf * 2, 3, 1, 0, bias=False),
                                      nn.BatchNorm2d(ndf * 2, eps=1e-3) if self.__use_bn else nn.InstanceNorm2d(
                                          ndf * 2),
                                      nn.ReLU(inplace=True))  # state size 64 x 64

        # Group 3
        self._layer31 = nn.Sequential(nn.Conv2d(ndf * 2, ndf * 4, 3, 1, 0, bias=False),
                                      nn.BatchNorm2d(ndf * 4, eps=1e-3) if self.__use_bn else nn.InstanceNorm2d(
                                          ndf * 4),
                                      nn.ReLU(inplace=True))  # state size 32 x 32

        self._global_avg_pool = nn.AvgPool2d(10, 1, 0)

        self.main_flow = nn.Sequential(OrderedDict([("conv_block1", self._layer11),
                                                    ("conv_block2", self._layer12),
                                                    ("conv_block3", self._layer13),
                                                    ("maxpool1", self._pool1),
                                                    ("conv_block4", self._layer21),
                                                    ("conv_block5", self._layer22),
                                                    ("maxpool2", self._pool2),
                                                    ("conv_block6", self._layer31),
                                                    ("global_avg_pool", self._global_avg_pool)
                                                    ]))

        self.classify = nn.Linear(ndf * 8, n_cls)  # state size. n_clsx1x1

        self.apply(aleksei_weights_init)

    def forward(self, x):
        if len(x.shape) != 4 or x.shape[1] != 2:
            raise ValueError('Input tensor ({}) must have 4 dims and 2 channels, but got'.format(x.shape, x.shape))

        x0 = x[:, 0:1, :, :]
        x1 = x[:, 1:2, :, :]
        x0 = self.main_flow(x0).view(x0.shape[0], -1)
        x1 = self.main_flow(x1).view(x1.shape[0], -1)
        x01 = torch.cat((x0, x1), dim=1)

        x01 = self.__drop(x01)

        output = self.classify(x01)

        if len(output.shape) < 2 or output.shape[1] != self.__required_output_channels:
            raise ValueError(
                'Require output channels {}, but got {}'.format(self.__required_output_channels, output.shape[1]))
        return output


class DisAlekseiVH(Module):
    def __init__(self, nc=1, ndf=64, n_cls=5, drop_rate=0.35, use_bn=False):
        super().__init__()
        # input is (nc) x 32 x 32
        self.__drop_rate = drop_rate
        self.__use_bn = use_bn
        self.__required_output_channels = n_cls

        self.__drop = nn.Dropout(p=self.__drop_rate)

        self._pool1 = nn.MaxPool2d(2, 2)
        self._pool2 = nn.MaxPool2d(2, 2)

        # Group 1
        # input is (nc) x 256 x 256
        self._layer11 = nn.Sequential(nn.Conv2d(nc, ndf, 3, 2, 0, bias=False),
                                      nn.BatchNorm2d(ndf, eps=1e-3) if self.__use_bn else nn.InstanceNorm2d(ndf),
                                      nn.ReLU(inplace=True))  # state size 128 x 128
        self._layer12 = nn.Sequential(nn.Conv2d(ndf, ndf, 3, 1, 0, bias=False),
                                      nn.BatchNorm2d(ndf, eps=1e-3) if self.__use_bn else nn.InstanceNorm2d(ndf),
                                      nn.ReLU(inplace=True))  # state size 128 x 128

        self._layer13 = nn.Sequential(nn.Conv2d(ndf, ndf, 3, 1, 0, bias=False),
                                      nn.BatchNorm2d(ndf, eps=1e-3) if self.__use_bn else nn.InstanceNorm2d(ndf),
                                      nn.ReLU(inplace=True))  # state size 128 x 128

        # Group 2

        self._layer21 = nn.Sequential(nn.Conv2d(ndf, ndf * 2, 3, 1, 0, bias=False),
                                      nn.BatchNorm2d(ndf * 2, eps=1e-3) if self.__use_bn else nn.InstanceNorm2d(
                                          ndf * 2),
                                      nn.ReLU(inplace=True))  # state size 64 x 64

        self._layer22 = nn.Sequential(nn.Conv2d(ndf * 2, ndf * 2, 3, 1, 0, bias=False),
                                      nn.BatchNorm2d(ndf * 2, eps=1e-3) if self.__use_bn else nn.InstanceNorm2d(
                                          ndf * 2),
                                      nn.ReLU(inplace=True))  # state size 64 x 64

        # Group 3
        self._layer31 = nn.Sequential(nn.Conv2d(ndf * 2, ndf * 4, 3, 1, 0, bias=False),
                                      nn.BatchNorm2d(ndf * 4, eps=1e-3) if self.__use_bn else nn.InstanceNorm2d(
                                          ndf * 4),
                                      nn.ReLU(inplace=True))  # state size 32 x 32

        # Pool VH
        self._pool = nn.Sequential(nn.MaxPool2d((10, 1), (10, 1), 0),
                                   nn.Conv2d(ndf * 4, ndf * 4, 1, 1, 0, bias=False),
                                   nn.BatchNorm2d(ndf * 4) if self.__use_bn else nn.InstanceNorm2d(ndf * 8),
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.MaxPool2d((1, 10), 10, 0))  # state size ndf * 4 x 1 x 1

        self.main_flow = nn.Sequential(OrderedDict([("conv_block1", self._layer11),
                                                    ("conv_block2", self._layer12),
                                                    ("conv_block3", self._layer13),
                                                    ("maxpool1", self._pool1),
                                                    ("conv_block4", self._layer21),
                                                    ("conv_block5", self._layer22),
                                                    ("maxpool2", self._pool2),
                                                    ("conv_block6", self._layer31),
                                                    ("sam", self._pool)
                                                    ]))

        self.classify = nn.Linear(ndf * 8, n_cls)  # state size. n_clsx1x1

        self.apply(aleksei_weights_init)

    def forward(self, x):
        if len(x.shape) != 4 or x.shape[1] != 2:
            raise ValueError('Input tensor ({}) must have 4 dims and 2 channels, but got'.format(x.shape, x.shape))

        x0 = x[:, 0:1, :, :]
        x1 = x[:, 1:2, :, :]
        x0 = self.main_flow(x0).view(x0.shape[0], -1)
        x1 = self.main_flow(x1).view(x1.shape[0], -1)
        x01 = torch.cat((x0, x1), dim=1)

        x01 = self.__drop(x01)

        output = self.classify(x01)

        if len(output.shape) < 2 or output.shape[1] != self.__required_output_channels:
            raise ValueError(
                'Require output channels {}, but got {}'.format(self.__required_output_channels, output.shape[1]))
        return output


class DisAlekseiHV(Module):
    def __init__(self, nc=1, ndf=64, n_cls=5, drop_rate=0.35, use_bn=False):
        super().__init__()
        # input is (nc) x 32 x 32
        self.__drop_rate = drop_rate
        self.__use_bn = use_bn
        self.__required_output_channels = n_cls

        self.__drop = nn.Dropout(p=self.__drop_rate)

        self._pool1 = nn.MaxPool2d(2, 2)
        self._pool2 = nn.MaxPool2d(2, 2)

        # Group 1
        # input is (nc) x 256 x 256
        self._layer11 = nn.Sequential(nn.Conv2d(nc, ndf, 3, 2, 0, bias=False),
                                      nn.BatchNorm2d(ndf, eps=1e-3) if self.__use_bn else nn.InstanceNorm2d(ndf),
                                      nn.ReLU(inplace=True))  # state size 128 x 128
        self._layer12 = nn.Sequential(nn.Conv2d(ndf, ndf, 3, 1, 0, bias=False),
                                      nn.BatchNorm2d(ndf, eps=1e-3) if self.__use_bn else nn.InstanceNorm2d(ndf),
                                      nn.ReLU(inplace=True))  # state size 128 x 128

        self._layer13 = nn.Sequential(nn.Conv2d(ndf, ndf, 3, 1, 0, bias=False),
                                      nn.BatchNorm2d(ndf, eps=1e-3) if self.__use_bn else nn.InstanceNorm2d(ndf),
                                      nn.ReLU(inplace=True))  # state size 128 x 128

        # Group 2

        self._layer21 = nn.Sequential(nn.Conv2d(ndf, ndf * 2, 3, 1, 0, bias=False),
                                      nn.BatchNorm2d(ndf * 2, eps=1e-3) if self.__use_bn else nn.InstanceNorm2d(
                                          ndf * 2),
                                      nn.ReLU(inplace=True))  # state size 64 x 64

        self._layer22 = nn.Sequential(nn.Conv2d(ndf * 2, ndf * 2, 3, 1, 0, bias=False),
                                      nn.BatchNorm2d(ndf * 2, eps=1e-3) if self.__use_bn else nn.InstanceNorm2d(
                                          ndf * 2),
                                      nn.ReLU(inplace=True))  # state size 64 x 64

        # Group 3
        self._layer31 = nn.Sequential(nn.Conv2d(ndf * 2, ndf * 4, 3, 1, 0, bias=False),
                                      nn.BatchNorm2d(ndf * 4, eps=1e-3) if self.__use_bn else nn.InstanceNorm2d(
                                          ndf * 4),
                                      nn.ReLU(inplace=True))  # state size 32 x 32

        # Pool HV
        self._pool = nn.Sequential(nn.MaxPool2d((1, 10), (1, 10), 0),
                                   nn.Conv2d(ndf * 4, ndf * 4, 1, 1, 0, bias=False),
                                   nn.BatchNorm2d(ndf * 4) if self.__use_bn else nn.InstanceNorm2d(ndf * 8),
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.MaxPool2d((10, 1), 10, 0))  # state size ndf * 4 x 1 x 1

        self.main_flow = nn.Sequential(OrderedDict([("conv_block1", self._layer11),
                                                    ("conv_block2", self._layer12),
                                                    ("conv_block3", self._layer13),
                                                    ("maxpool1", self._pool1),
                                                    ("conv_block4", self._layer21),
                                                    ("conv_block5", self._layer22),
                                                    ("maxpool2", self._pool2),
                                                    ("conv_block6", self._layer31),
                                                    ("sam", self._pool)
                                                    ]))

        self.classify = nn.Linear(ndf * 8, n_cls)  # state size. n_clsx1x1

        self.apply(aleksei_weights_init)

    def forward(self, x):
        if len(x.shape) != 4 or x.shape[1] != 2:
            raise ValueError('Input tensor ({}) must have 4 dims and 2 channels, but got'.format(x.shape, x.shape))

        x0 = x[:, 0:1, :, :]
        x1 = x[:, 1:2, :, :]
        x0 = self.main_flow(x0).view(x0.shape[0], -1)
        x1 = self.main_flow(x1).view(x1.shape[0], -1)
        x01 = torch.cat((x0, x1), dim=1)

        x01 = self.__drop(x01)

        output = self.classify(x01)
        if len(output.shape) < 2 or output.shape[1] != self.__required_output_channels:
            raise ValueError(
                'Require output channels {}, but got {}'.format(self.__required_output_channels, output.shape[1]))
        return output
