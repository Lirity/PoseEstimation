import torch
import torch.nn as nn

from model.base.module import PointNet2MSG


class Net(nn.Module):
    def __init__(self, n_cls=6):
        super(Net, self).__init__()
        self.n_cls = n_cls
        self.pn2msg = PointNet2MSG(radii_list=[[0.01, 0.02], [0.02,0.04], [0.04,0.08], [0.08,0.16]])

        self.t_mlp = nn.Sequential(
            nn.Conv1d(256, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(128, 3, 1),
        )
        self.t_mlp[-1].bias.data.zero_()
        self.s_mlp = nn.Sequential(
            nn.Conv1d(256, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(128, 3*self.n_cls, 1),
        )

    def forward(self, inputs):
        rgb = inputs['rgb']
        pts = inputs['pts']
        cls = inputs['category_label'].long()

        x = torch.cat([pts, pts, rgb], dim=2)
        x = self.pn2msg(x)

        t = self.t_mlp(x) + pts.transpose(1,2)
        t = torch.mean(t, dim=2)

        cls = cls.reshape(pts.size(0),1,1).expand(pts.size(0),1,3).contiguous()
        s = self.s_mlp(x).reshape(pts.size(0),self.n_cls,3,pts.size(1)).mean(3)
        s = torch.gather(s,1,cls).squeeze(1)

        end_points = {}
        end_points['translation'] = t
        end_points['size'] = s
        return end_points
