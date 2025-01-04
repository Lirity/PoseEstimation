import torch.nn as nn


class Loss(nn.Module):
    def __init__(self, cfg):
        super(Loss, self).__init__()
        self.cfg = cfg
        self.criterion = nn.L1Loss()

    def forward(self, pred, gt):
        loss_t = self.criterion(pred['translation'], gt['translation_label'])
        loss_s = self.criterion(pred['size'], gt['size_label'])

        loss = self.cfg.t_weight * loss_t + self.cfg.s_weight * loss_s
        return {
            'loss': loss,
            't': loss_t,
            's': loss_s,
        }
