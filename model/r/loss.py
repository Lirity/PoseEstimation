import torch
import torch.nn as nn
import torch.nn.functional as F

from model.base.loss import SigmoidFocalLoss
from utils.rotation_utils import angle_of_rotation

class Loss(nn.Module):
    def __init__(self, cfg):
        super(Loss, self).__init__()
        self.cfg = cfg
        self.l1loss = nn.L1Loss()
        self.smoothl1loss = SmoothL1Dis
        self.sfloss = SigmoidFocalLoss()

    def forward(self, pred, gt):
        rho_prob = pred['rho_prob']
        rho_label = F.one_hot(gt['rho_label'].squeeze(1), num_classes=rho_prob.size(1)).float()
        rho_loss = self.sfloss(rho_prob, rho_label).mean()
        pred_rho = torch.max(torch.sigmoid(rho_prob), 1)[1]
        rho_acc = (pred_rho.long() == gt['rho_label'].squeeze(1).long()).float().mean() * 100.0

        phi_prob = pred['phi_prob']
        phi_label = F.one_hot(gt['phi_label'].squeeze(1), num_classes=phi_prob.size(1)).float()
        phi_loss = self.sfloss(phi_prob, phi_label).mean()
        pred_phi = torch.max(torch.sigmoid(phi_prob), 1)[1]
        phi_acc = (pred_phi.long() == gt['phi_label'].squeeze(1).long()).float().mean() * 100.0

        vp_loss = rho_loss + phi_loss
        ip_loss = self.l1loss(pred['pred_rotation'], gt['rotation_label'])

        residual_angle = angle_of_rotation(pred['pred_rotation'].transpose(1, 2) @ gt['rotation_label'])

        loss = self.cfg.vp_weight * vp_loss + ip_loss
        return {
            'loss': loss,
            'vp_loss': vp_loss,
            'ip_loss': ip_loss,
            'rho_acc': rho_acc,
            'phi_acc': phi_acc,
            'residual_angle_loss': residual_angle.mean(),

            '5d_loss': (residual_angle < 5).sum() / residual_angle.shape[0],

        }
    
def SmoothL1Dis(p1, p2, threshold=0.1):
    '''
    p1: b*n*3
    p2: b*n*3
    '''
    diff = torch.abs(p1 - p2)
    less = torch.pow(diff, 2) / (2.0 * threshold)
    higher = diff - threshold / 2.0
    dis = torch.where(diff > threshold, higher, less)
    dis = torch.mean(torch.sum(dis, dim=2))
    return dis