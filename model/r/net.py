import torch
import torch.nn as nn
import torchvision.transforms as transforms

from lib.sphericalmap_utils.smap_utils import Feat2Smap
from model.base.extractor_dino import ViTExtractor
from model.base.module import V_Branch, I_Branch, SphericalFPN, PointNet2MSG


class Net(nn.Module):
    def __init__(self, resolution=64, ds_rate=2, num_patches=15):
        super(Net, self).__init__()
        self.res = resolution
        self.ds_rate = ds_rate
        self.ds_res = resolution // ds_rate
        self.n_cls = 6
        self.match_num = 100

        # data processing
        self.feat2smap = Feat2Smap(self.res)

        self.spherical_fpn = SphericalFPN(
            ds_rate=self.ds_rate, dim_in1=1, dim_in2=3 + 384)
        self.v_branch = V_Branch(resolution=self.ds_res, in_dim=256)
        self.i_branch = I_Branch(resolution=self.ds_res, in_dim=256)

        # dinov2 TODO 优化成预处理
        extractor = ViTExtractor('dinov2_vits14', 14, device='cuda')
        self.extractor = torch.hub.load(
            'facebookresearch/dinov2',
            'dinov2_vits14').cuda()
        self.extractor_preprocess = transforms.Normalize(mean=extractor.mean, std=extractor.std)
        self.extractor_layer = 11
        self.extractor_facet = 'token'
        self.num_patches = num_patches

        self.pn2msg = PointNet2MSG(
            radii_list=[[0.01, 0.02],
                        [0.02, 0.04],
                        [0.04, 0.08],
                        [0.08, 0.16]])

        self.pred_pts = nn.Sequential(
            nn.Conv1d(256, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(128, 3 * self.n_cls, 1),
        )

    def extract_feature(self, rgb_raw):
        rgb_raw = rgb_raw.permute(0, 3, 1, 2)   # (b, 3, h, w)
        rgb_raw = self.extractor_preprocess(rgb_raw)    # (b, 3, h, w)
        with torch.no_grad():
            dino_feature = self.extractor.forward_features(rgb_raw)["x_prenorm"][:, 1:]  # (b, 255, 384)
        dino_feature = dino_feature.reshape(dino_feature.shape[0], (self.num_patches) ** 2, -1)  # (b, 225, 384)
        return dino_feature.contiguous()

    def forward(self, inputs):
        b = inputs['rgb'].shape[0]
        rgb = inputs['rgb']  # (b, 2048, 3)
        pts = inputs['pts']  # (b, 2048, 3)
        rgb_raw = inputs['rgb_raw']  # (b, 210, 210, 3)
        pts_raw = inputs['pts_raw']  # (b, 15, 15, 3)
        choose = inputs['choose'][:, :self.match_num]
        cls = inputs['category_label'].reshape(-1)
        index = cls + torch.arange(b, dtype=torch.long).cuda() * 6

        dis_map, rgb_map = self.feat2smap(pts, rgb)  # (b, 1, 64, 64) (b, 3, 64, 64)

        dino_feature = self.extract_feature(rgb_raw)  # (b, 225, 384)
        
        ptsf = pts_raw.reshape(b, (self.num_patches) ** 2, -1)[torch.arange(b)[:, None], choose, :]  # (b, 100, 3)
        dino_feature = dino_feature[torch.arange(b)[:, None], choose, :]  # (b, 100, 384)
        # ptsf = pts_raw.reshape(b, (self.num_patches) ** 2, -1)  # (b, 225, 3)

        _, ref_map = self.feat2smap(ptsf, dino_feature)  # (b, 384, 64, 64)

        x = self.spherical_fpn(dis_map, torch.cat([rgb_map, ref_map], dim=1))  # (b, 256, 32, 32)

        # if self.training:
        y = torch.cat([pts, pts, rgb], dim=2)
        feat = self.pn2msg(y)    # (b, 256, 2048)
        pred_pts = self.pred_pts(feat)  # (b, 3*6, 2048)
        pred_pts = pred_pts.view(-1, 3, 2048).contiguous()
        pred_pts = torch.index_select(pred_pts, 0, index)
        pred_pts = pred_pts.permute(0, 2, 1).contiguous()

        # viewpoint rotation
        vp_rot, rho_prob, phi_prob = self.v_branch(x, inputs)

        # in-plane rotation
        ip_rot = self.i_branch(x, vp_rot)

        outputs = {
            'pred_rotation': vp_rot @ ip_rot,
            'rho_prob': rho_prob,
            'phi_prob': phi_prob,
            'pred_pts': pred_pts + pts,
        }

        return outputs
