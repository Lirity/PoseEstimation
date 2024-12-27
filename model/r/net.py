import math

import open3d
import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms

from lib.sphericalmap_utils.smap_utils import Feat2Smap
from model.base.extractor_dino import ViTExtractor
from model.base.module import V_Branch, I_Branch, SphericalFPN

class Net(nn.Module):
    def __init__(self, resolution=64, ds_rate=2, num_patches=15):
        super(Net, self).__init__()
        self.res = resolution
        self.ds_rate = ds_rate
        self.ds_res = resolution // ds_rate
        extractor = ViTExtractor('dinov2_vits14', 14, device='cuda')
        self.extractor = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').cuda()

        self.extractor_preprocess = transforms.Normalize(mean=extractor.mean, std=extractor.std)
        self.extractor_layer = 11
        self.extractor_facet = 'token'

        self.num_patches = num_patches
        # data processing
        self.feat2smap = Feat2Smap(self.res)
        self.spherical_fpn = SphericalFPN(ds_rate=self.ds_rate, dim_in1=1, dim_in2=3 + 384)

        self.v_branch = V_Branch(resolution=self.ds_res, in_dim=256)
        self.i_branch = I_Branch(resolution=self.ds_res, in_dim=256)

    def extract_feature(self, rgb_raw):
        rgb_raw = rgb_raw.permute(0, 3, 1, 2)
        rgb_raw = self.extractor_preprocess(rgb_raw)

        with torch.no_grad():
            dino_feature = self.extractor.forward_features(rgb_raw)["x_prenorm"][:, 1:]

        dino_feature = dino_feature.reshape(dino_feature.shape[0], self.num_patches, self.num_patches, -1)
        return dino_feature.contiguous()  # b x c x h x w

    def rotate_pts_batch(self, pts, rotation):
        pts_shape = pts.shape
        b = pts_shape[0]

        return (rotation[:, None, :, :] @ pts.reshape(b, -1, 3)[:, :, :, None]).squeeze().reshape(pts_shape)

    def forward(self, inputs):
        rgb = inputs['rgb']
        pts = inputs['pts']
        b = inputs['rgb_raw'].shape[0]

        rgb_raw = inputs['rgb_raw'] # (b, 210, 210, 3)
        pts_raw = inputs['pts_raw'] # (b, 15, 15, 3)


        # 大模型特征
        feature = self.extract_feature(rgb_raw).reshape(b, (self.num_patches) ** 2, -1) # (b, 225, 384)
        
        match_num = 100
        choose = inputs['choose'][:, :match_num]    # (b, 2048) -> (b, 100)

        # 筛选100个点和对应的大模型特征
        ptsf = pts_raw.reshape(b, (self.num_patches) ** 2, -1)[torch.arange(b)[:, None], choose, :]  # (b, 100, 3)
        feature = feature[torch.arange(b)[:, None], choose, :]  # (b, 100, 384)

        dis_map, rgb_map = self.feat2smap(pts, rgb)
        _, ref_map = self.feat2smap(ptsf, feature)

        # backbone
        x = self.spherical_fpn(dis_map, torch.cat([rgb_map, ref_map], dim=1))

        # viewpoint rotation
        vp_rot, rho_prob, phi_prob = self.v_branch(x, inputs)
        
        # in-plane rotation
        ip_rot = self.i_branch(x, vp_rot)

        outputs = {
            'pred_rotation': vp_rot @ ip_rot,
            'rho_prob': rho_prob,
            'phi_prob': phi_prob,

        }
        return outputs