import math

import open3d
import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms

from lib.sphericalmap_utils.smap_utils import Feat2Smap
from model.base.extractor_dino import ViTExtractor
from model.base.module import PointNet2MSG, V_Branch, I_Branch, SphericalFPN3

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
        self.pn2msg = PointNet2MSG(radii_list=[[0.01, 0.02], [0.02, 0.04], [0.04, 0.08], [0.08, 0.16]], dim_in=384 + 3)
        self.pts_extractor = PointNet2MSG(radii_list=[[0.01, 0.02], [0.02, 0.04], [0.04, 0.08], [0.08, 0.16]], dim_in=3)
        self.num_patches = num_patches

        self.cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
        self.soft_max = nn.Softmax(dim=-1)
        self.ppf_nn = [10, 20, 40, 80, 160, 300]
        # data processing
        self.feat2smap = Feat2Smap(self.res)
        self.feat2smap_drift = Feat2Smap(self.res // self.ds_rate)
        self.spherical_fpn = SphericalFPN3(ds_rate=self.ds_rate, dim_in1=1, dim_in2=3 + 384,
                                           dim_in3=4 * len(self.ppf_nn))  # 387)
        self.v_branch = V_Branch(resolution=self.ds_res, in_dim=256)
        self.i_branch = I_Branch(resolution=self.ds_res, in_dim=256)
        self.match_threshould = nn.Parameter(torch.tensor(-1.0, requires_grad=True))

    def cal_normal(self, points, rand_rotaton, translation, size):
        b, n, _ = points.shape
        translation = translation / size.norm(dim=-1, keepdim=True)
        points = self.rotate_pts_batch(points, rand_rotaton) + translation[:, None, :].repeat(1, n, 1)

        rets = []
        for point in points:
            pcd = open3d.geometry.PointCloud()
            pcd.points = open3d.utility.Vector3dVector(point.cpu().numpy())
            pcd.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamHybrid(
                radius=4, max_nn=10))

            pcd.orient_normals_towards_camera_location()
            ret = torch.FloatTensor(np.asarray(pcd.normals)).cuda()

            rets.append(ret)
        rets = torch.stack(rets, dim=0)
        rets = self.rotate_pts_batch(rets, rand_rotaton.transpose(1, 2))
        return rets

    def calc_ppf(self, pts, normal):
        b, n, _ = pts.shape
        k_list = self.ppf_nn
        knn_index_list = hierachical_knn_query_list(pts, pts, k_list)
        patches_list = []
        patches_normals_list = []
        for k, knn_index in zip(k_list, knn_index_list):
            # if k>1:
            #     knn_index = knn_index[:,:,1:]
            patches_list.append(pts[torch.arange(b)[:, None, None], knn_index])
            patches_normals_list.append(normal[torch.arange(b)[:, None, None], knn_index])
        ppfs = []
        for patches, patch_normals in zip(patches_list, patches_normals_list):
            ppfs.append(calc_ppf_gpu(pts, normal, patches, patch_normals).reshape(b, n, -1))

        return torch.cat(ppfs, dim=-1)

    def extract_feature(self, rgb_raw):

        rgb_raw = rgb_raw.permute(0, 3, 1, 2)

        rgb_raw = self.extractor_preprocess(rgb_raw)
        # import pdb;pdb.set_trace()

        with torch.no_grad():
            dino_feature = self.extractor.forward_features(rgb_raw)["x_prenorm"][:, 1:]

        dino_feature = dino_feature.reshape(dino_feature.shape[0], self.num_patches, self.num_patches, -1)
        return dino_feature.contiguous()  # b x c x h x w

    def rotate_pts_batch(self, pts, rotation):
        pts_shape = pts.shape
        b = pts_shape[0]

        return (rotation[:, None, :, :] @ pts.reshape(b, -1, 3)[:, :, :, None]).squeeze().reshape(pts_shape)

    def inference(self, inputs):
        pts = inputs['pts']
        rgb = inputs['rgb']
        b, rgb_h, rgb_w, _ = inputs['rgb_raw'].shape

        rgb_raw = inputs['rgb_raw']
        feature = self.extract_feature(rgb_raw).reshape(b, (self.num_patches) ** 2, -1)

        match_num = 100
        choose = inputs['choose'][:, :match_num]
        feature = feature[torch.arange(b)[:, None],
                  choose.reshape(b, match_num), :]
        pts_raw = inputs['pts_raw']
        ptsf = pts_raw.reshape(b, (self.num_patches) ** 2, -1)[torch.arange(b)[:, None], choose, :]

        ptsg = pts[:, :300, :]
        with torch.no_grad():
            normals = self.cal_normal(ptsg, torch.eye(3)[None, :, :].repeat(ptsg.shape[0], 1, 1).cuda(),
                                      inputs['center'] + inputs['translation'], inputs['size'])
            ppf_feature = self.calc_ppf(ptsg, normals)

        dis_map, rgb_map = self.feat2smap(pts, rgb)
        _, ref_map = self.feat2smap(ptsf, feature)
        _, ppf_map = self.feat2smap(ptsg, ppf_feature)

        x = self.spherical_fpn(dis_map, torch.cat([rgb_map, ref_map], dim=1), ppf_map)
        vp_rot, rho_prob, phi_prob = self.v_branch(x, inputs)
        pred_vp_rot = self.v_branch._get_vp_rotation(rho_prob, phi_prob, {})

        ip_rot = self.i_branch(x, vp_rot)
        outputs = {
            'pred_rotation': vp_rot @ ip_rot,
        }
        return outputs

    def forward(self, inputs):
        rgb = inputs['rgb']
        pts = inputs['pts']
        b, rgb_h, rgb_w, _ = inputs['rgb_raw'].shape

        rgb_raw = inputs['rgb_raw'] # (b, 210, 210, 3)

        feature = self.extract_feature(rgb_raw).reshape(b, (self.num_patches) ** 2, -1) # (b, 225, 384)

        match_num = 100
        choose = inputs['choose'][:, :match_num]    # (b, 2048) -> (b, 100)


        pts_raw = inputs['pts_raw']
        pts_raw = pts_raw.reshape(b, (self.num_patches) ** 2, -1)[torch.arange(b)[:, None], choose, :]  # (b, 100, 3)

        rgb_raw = rgb_raw.reshape(b, (self.num_patches) ** 2, -1)[torch.arange(b)[:, None], choose, :]

        ptsf = pts_raw
        feature = feature[torch.arange(b)[:, None], choose, :]  # (b, 100, 384)

        ptsg = pts[:, :300, :]
        with torch.no_grad():
            normals = self.cal_normal(ptsg, inputs['rand_rotation'], inputs['translation_label'], inputs['size_label'])
            ppf_feature = self.calc_ppf(ptsg, normals)



        dis_map, rgb_map = self.feat2smap(pts, rgb)

        print(ptsf.shape)
        print(feature.shape)
        aaa
        _, ref_map = self.feat2smap(ptsf, feature)
        _, ppf_map = self.feat2smap(ptsg, ppf_feature)

        # backbone
        x = self.spherical_fpn(dis_map, torch.cat([rgb_map, ref_map], dim=1), ppf_map)

        # viewpoint rotation
        vp_rot, rho_prob, phi_prob = self.v_branch(x, inputs)
        pred_vp_rot = self.v_branch._get_vp_rotation(rho_prob, phi_prob, {})

        ip_rot = self.i_branch(x, vp_rot)

        outputs = {
            'pred_rotation': vp_rot @ ip_rot,
            'pred_vp_rotation': pred_vp_rot,
            'rho_prob': rho_prob,
            'phi_prob': phi_prob,

        }
        return outputs
    


def hierachical_knn_query_list(points, ref_points, k_list):
    b, p, _ = points.shape
    _, r, _ = ref_points.shape
    dist = torch.cdist(points, ref_points)
    k = k_list[0]
    ret = [torch.topk(-dist, k)[1]]
    for i in range(1, len(k_list)):
        k = k_list[i]
        last_k = k_list[i - 1]

        knn_index = torch.topk(-dist, k)[1][:, :, last_k:]
        ret.append(knn_index)
        # ret.append(ref_points[torch.arange(b)[:,None,None], knn_index])
    return ret

def calc_ppf_gpu(points, point_normals, patches, patch_normals):
    '''
    Calculate ppf gpu
    points: [b, n, 3]
    point_normals: [b, n, 3]
    patches: [b, n, nsamples, 3]
    patch_normals: [b, n, nsamples, 3]
    '''
    points = torch.unsqueeze(points, dim=2).expand(-1, -1, patches.shape[2], -1)
    point_normals = torch.unsqueeze(point_normals, dim=2).expand(-1, -1, patches.shape[2], -1)
    vec_d = patches - points  # [b, n, n_samples, 3]
    d = torch.sqrt(torch.sum(vec_d ** 2, dim=-1, keepdim=True))  # [b, n, n_samples, 1]
    # angle(n1, vec_d)
    y = torch.sum(point_normals * vec_d, dim=-1, keepdim=True)
    x = torch.cross(point_normals, vec_d, dim=-1)
    x = torch.sqrt(torch.sum(x ** 2, dim=-1, keepdim=True))
    angle1 = torch.atan2(x, y) / math.pi

    # angle(n2, vec_d)
    y = torch.sum(patch_normals * vec_d, dim=-1, keepdim=True)
    x = torch.cross(patch_normals, vec_d, dim=-1)
    x = torch.sqrt(torch.sum(x ** 2, dim=-1, keepdim=True))
    angle2 = torch.atan2(x, y) / math.pi

    # angle(n1, n2)
    y = torch.sum(point_normals * patch_normals, dim=-1, keepdim=True)
    x = torch.cross(point_normals, patch_normals, dim=-1)
    x = torch.sqrt(torch.sum(x ** 2, dim=-1, keepdim=True))
    angle3 = torch.atan2(x, y) / math.pi

    ppf = torch.cat([d, angle1, angle2, angle3], dim=-1)  # [b, n, samples, 4]
    return ppf.mean(dim=-2, keepdim=True)
    # return torch.cat([ppf.mean(dim = -2, keepdim = True), 
    #                   ppf.max(dim = -2,  keepdim = True)[0], 
    #                   ppf.min(dim = -2,  keepdim = True)[0]], dim = -2)