import torch
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

from collagen.core import Module


class MixMatchModelLoss(Module):
    def __init__(self, w_u=0.5):
        super().__init__()
        self.__loss_cons = self.softmax_mse_loss
        self.__loss_cls = CrossEntropyLoss()
        self.__w_u = w_u
        self.__losses = {'loss_x': None, 'loss_u': None}

        self.__n_minibatches = 2.0

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

    def forward(self, pred, target):
        if target['name'] == 'train_mixmatch':
            n_minibatch_size = pred['x_mix'].shape[0]

            target_mix_x = target['target_mix_x']
            target_mix_u = target['target_mix_u']

            preds_l = torch.log_softmax(pred['x_mix'], dim=1)
            logits_u = pred['u_mix']

            loss_x = -(preds_l * target_mix_x).sum(dim=1).mean()
            loss_u = self.__loss_cons(logits_u, target_mix_u)

            self.__losses['loss_u'] = self.__w_u * loss_u / (self.__n_minibatches * n_minibatch_size)
            self.__losses['loss_x'] = loss_x / self.__n_minibatches
            self.__losses['loss'] = self.__losses['loss_x'] + self.__losses['loss_u']
        elif target['name'] == 'l_eval':
            self.__losses['loss_x'] = None
            self.__losses['loss_u'] = None
            self.__losses['loss'] = self.__loss_cls(pred, target['target'].type(torch.int64))

        return self.__losses['loss']

    def get_loss_by_name(self, name):
        if name in self.__losses:
            return self.__losses[name]
        else:
            return None
