import torch
from collagen.core import Module
from torch import Tensor
from torch.nn import CrossEntropyLoss, functional as F


class PiModelLoss(Module):
    def __init__(self, alpha=0.5, cons_mode='mse'):
        super().__init__()
        self.__cons_mode = cons_mode
        if cons_mode == 'mse':
            self.__loss_cons = self.softmax_mse_loss
        elif cons_mode == 'kl':
            self.__loss_cons = self.softmax_kl_loss
        self.__loss_cls = CrossEntropyLoss(reduction='sum')
        self.__alpha = alpha
        self.__losses = {'loss_cls': None, 'loss_cons': None}

        self.__n_minibatches = 2.0

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

    def forward(self, pred: Tensor, target: Tensor):
        n_minibatch_size = pred.shape[0]
        if target['name'] == 'u':
            aug_logit = target['logits']

            if len(aug_logit.shape) == 2:
                loss_cons = self.__loss_cons(aug_logit, pred)
            elif len(aug_logit.shape) == 3:
                loss_cons = self.__loss_cons(aug_logit[0, :, :], aug_logit[1, :, :])
            else:
                raise ValueError('Not support augmented logits with shape {}'.format(aug_logit.shape))

            self.__losses['loss_cons'] = self.__alpha * loss_cons / n_minibatch_size
            self.__losses['loss_cls'] = None
            self.__losses['loss'] = self.__losses['loss_cons']
            _loss = self.__losses['loss']

        elif target['name'] == 'l':
            target_cls = target['target'].type(torch.int64)

            loss_cls = self.__loss_cls(pred, target_cls)

            self.__losses['loss_cons'] = None
            self.__losses['loss_cls'] = loss_cls / n_minibatch_size
            self.__losses['loss'] = self.__losses['loss_cls']
            _loss = self.__losses['loss']
        else:
            raise ValueError("Not support target name {}".format(target['name']))

        return _loss

    def get_loss_by_name(self, name):
        if name in self.__losses:
            return self.__losses[name]
        else:
            return None
