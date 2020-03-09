import torch
from torch.nn import Module, CrossEntropyLoss
from torch.nn import functional as F
from torch import Tensor
import numpy as np

from collagen.core.utils import to_cpu


class Loss(Module):
    def __init__(self, cons_coef=2.0, cons_mode='mse', mixup_coef=2.0, cons_mixup_coef=4.0, use_cons=False, elim_loss=""):
        super().__init__()
        self.__cons_mode = cons_mode
        if cons_mode == 'mse':
            self.__loss_cons = self.softmax_mse_loss
        elif cons_mode == 'kl':
            self.__loss_cons = self.softmax_kl_loss

        self.__loss_cls = CrossEntropyLoss(reduction='sum')

        self.__cons_coef = cons_coef
        self.__losses = {'loss_cls': None, 'loss_cons': None}

        self.__use_cons = use_cons
        self.__n_minibatches = 1.0
        self.__mixup_coef = mixup_coef
        self.__cons_mixup_coef = cons_mixup_coef

        self._elim_loss = elim_loss

        if "1" in self._elim_loss or "2" in self._elim_loss or "3" in self._elim_loss:
            print(f'[WARN] Semixup without regularizer {self._elim_loss}.')
        else:
            print('[INFO] Semixup uses all regularizers.')

        if use_cons and mixup_coef is None:
            raise ValueError('Must input coefficient of consistency')

        self.__n_aug = 1

        if self.__n_aug < 1:
            raise ValueError('Not support {} augmentations'.format(self.__n_aug))

    @staticmethod
    def softmax_kl_loss(input_logits, target_logits):
        """Takes softmax on both sides and returns KL divergence

        Note:
        - Returns the sum over all examples. Divide by the batch size afterwards
          if you want the mean.
        - Sends gradients to inputs but not the targets.
        """
        assert input_logits.size() == target_logits.size()
        input_log_softmax = F.log_softmax(input_logits, dim=1)
        target_softmax = F.softmax(target_logits, dim=1)
        n_classes = target_logits.shape[1]
        return F.kl_div(input_log_softmax, target_softmax, reduction='sum') / n_classes

    @staticmethod
    def softmax_mse_loss(input_logits, target_logits):
        """Takes softmax on both sides and returns MSE loss

        Note:
        - Returns the sum over all examples. Divide by the batch size afterwards
          if you want the mean.
        - Sends gradients to inputs but not the targets.
        """
        assert input_logits.size() == target_logits.size()
        input_softmax = F.softmax(input_logits, dim=1)
        target_softmax = F.softmax(target_logits, dim=1)
        n_classes = input_logits.size()[1]
        return F.mse_loss(input_softmax, target_softmax, reduction='sum') / n_classes

    @staticmethod
    def mse_loss(input_logits, target_logits):
        """Returns MSE loss

        Note:
        - Returns the sum over all examples. Divide by the batch size afterwards
          if you want the mean.
        - Sends gradients to inputs but not the targets.
        """
        assert input_logits.size() == target_logits.size()
        input_logits = input_logits.view(input_logits.shape[0], -1)
        target_logits = target_logits.view(target_logits.shape[0], -1)
        n_features = input_logits.size()[1]
        return F.mse_loss(input_logits, target_logits, reduction='sum') / n_features

    def forward(self, pred: Tensor, target: Tensor or dict):
        n_minibatch_size = pred.shape[0]
        if target['name'] == 'u_mixup':
            mixup_logits = pred

            if len(target['logits_mixup'].shape) == 2:
                loss_mixup = self.__loss_cons(mixup_logits, target['logits_mixup'])
            else:
                raise ValueError('Not support augmented logits with shape {}'.format(target['logits_mixup'].shape))

            loss_cons_aug_mixup = self.__loss_cons(mixup_logits, target['logits_aug'])

            loss_cons_aug = self.__loss_cons(target['logits_aug'], target['logits'])

            loss_cons_mixup = self.__loss_cons(target['logits'], mixup_logits)

            device = loss_mixup.device
            self.__losses['loss'] = torch.tensor([0], dtype=torch.float32).to(device)
            if "1" in self._elim_loss:
                self.__losses['loss_cons'] = None
            else:
                self.__losses['loss_cons'] = self.__cons_coef * loss_cons_aug / (
                        self.__n_minibatches * n_minibatch_size)
                self.__losses['loss'] += self.__losses['loss_cons']

            if "2" in self._elim_loss:
                self.__losses['loss_mixup'] = None
            else:
                self.__losses['loss_mixup'] = self.__mixup_coef * loss_mixup / (self.__n_minibatches * n_minibatch_size)
                self.__losses['loss'] += self.__losses['loss_mixup']

            if "3" in self._elim_loss:
                self.__losses['loss_cons_mixup'] = None
                self.__losses['loss_cons_aug_mixup'] = None
            else:
                self.__losses['loss_cons_mixup'] = self.__cons_mixup_coef * loss_cons_mixup / (
                        self.__n_minibatches * n_minibatch_size)
                self.__losses['loss_cons_aug_mixup'] = self.__cons_mixup_coef * loss_cons_aug_mixup / (
                        self.__n_minibatches * n_minibatch_size)
                self.__losses['loss'] += self.__losses['loss_cons_mixup'] + self.__losses['loss_cons_aug_mixup']

            self.__losses['loss_cls'] = None
        elif target['name'] == 'l_mixup':
            target_cls1 = target['target'].type(torch.int64)
            target_cls2 = target['target_bg'].type(torch.int64)
            alpha = target['alpha']

            loss_cls = alpha * self.__loss_cls(pred, target_cls1) + (1 - alpha) * self.__loss_cls(pred, target_cls2)

            self.__losses['loss_cons_aug_mixup'] = None
            self.__losses['loss_cons'] = None
            self.__losses['loss_mixup'] = None
            self.__losses['loss_cons_mixup'] = None
            self.__losses['loss_cls'] = loss_cls / (self.__n_minibatches * n_minibatch_size)
            self.__losses['loss'] = self.__losses['loss_cls']
        elif target['name'] == 'l_norm':
            target_cls = target['target'].type(torch.int64)
            loss_cls = self.__loss_cls(pred, target_cls)

            self.__losses['loss_cons_aug_mixup'] = None
            self.__losses['loss_cons'] = None
            self.__losses['loss_mixup'] = None
            self.__losses['loss_cons_mixup'] = None
            self.__losses['loss_cls'] = loss_cls / (self.__n_minibatches * n_minibatch_size)
            self.__losses['loss'] = self.__losses['loss_cls']
        else:
            raise ValueError("Not support target name {}".format(target['name']))

        _loss = self.__losses['loss']
        return _loss

    def get_loss_by_name(self, name):
        if name in self.__losses:
            return self.__losses[name]
        else:
            return None
