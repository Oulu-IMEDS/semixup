import torch
from torch.nn import Module, CrossEntropyLoss
from torch.nn import functional as F
from torch import Tensor

from collagen.lrscheduler.utils import ramps

class Loss(Module):
    def __init__(self, w_max=1.0, rampup_len=80, cons_mode='mse'):
        super().__init__()
        self.__cons_mode = cons_mode
        if cons_mode == 'mse':
            self.__loss_cons = self.softmax_mse_loss
        elif cons_mode == 'kl':
            self.__loss_cons = self.softmax_kl_loss
        self.__loss_cls = CrossEntropyLoss(reduction='sum')
        self.__w_max = w_max
        self.__losses = {'loss_cls': None, 'loss_cons': None}

        self.__n_minibatches = 2.0

        self.__n_aug = 1

        self.__step = 0
        self.__rampup_len = rampup_len

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
            ict_logit = target['logits_mixup']

            if len(ict_logit.shape) == 2:
                loss_cons = self.__loss_cons(ict_logit, pred)
            else:
                raise ValueError('Not support augmented logits with shape {}'.format(ict_logit.shape))

            self.__losses['loss_cons'] = self.__w_max * ramps.sigmoid_rampup(int(self.__step), self.__rampup_len) * loss_cons / (self.__n_minibatches*n_minibatch_size)
            self.__losses['loss_cls'] = None
            self.__losses['loss'] = self.__losses['loss_cons']
        elif target['name'] == 'l_mixup':
            ict_logit = target['logits_mixup']
            target_cls1 = target['target'].type(torch.int64)
            target_cls2 = target['target_bg'].type(torch.int64)
            alpha = target['alpha']

            loss_cls = alpha*self.__loss_cls(pred, target_cls1) + (1-alpha)*self.__loss_cls(pred, target_cls2)
            if len(ict_logit.shape) == 2:
                loss_cons = self.__loss_cons(ict_logit, pred)
            else:
                raise ValueError('Not support augmented logits with shape {}'.format(ict_logit.shape))

            self.__losses['loss_cons'] = self.__w_max * ramps.sigmoid_rampup(int(self.__step), self.__rampup_len) * loss_cons / (self.__n_minibatches*n_minibatch_size)
            self.__losses['loss_cls'] = loss_cls / (self.__n_minibatches*n_minibatch_size)
            self.__losses['loss'] = self.__losses['loss_cls'] + self.__losses['loss_cons']
        elif target['name'] == 'l_norm':
            target_cls = target['target'].type(torch.int64)

            loss_cls = self.__loss_cls(pred, target_cls)

            self.__losses['loss_cls'] = loss_cls / (self.__n_minibatches * n_minibatch_size)
            self.__losses['loss_cons'] = None
            self.__losses['loss'] = self.__losses['loss_cls']
        else:
            raise ValueError("Not support target name {}".format(target['name']))

        _loss = self.__losses['loss']
        self.__step += 1.0/self.__n_minibatches
        return _loss

    def get_loss_by_name(self, name):
        if name in self.__losses:
            return self.__losses[name]
        else:
            return None
