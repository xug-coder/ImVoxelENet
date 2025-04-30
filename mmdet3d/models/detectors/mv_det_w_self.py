import torch
from mmdet.models import DETECTORS, build_backbone, build_head, build_neck
from mmdet.models.detectors import BaseDetector

from mmdet3d.core import bbox3d2result

from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)

@DETECTORS.register_module()
class mvDet_w_self(BaseDetector):
    def __init__(self,
                 backbone,
                 neck,
                 neck_3d,
                 bbox_head,
                 n_voxels,
                 voxel_size,
                 head_2d=None,
                 train_cfg=None,
                 test_cfg=None,
                 ray_num=1000,
                 pretrained=None):
        super().__init__()
        self.backbone = build_backbone(backbone)
        self.neck = build_neck(neck)
        self.neck_3d = build_neck(neck_3d)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = build_head(bbox_head)
        self.bbox_head.voxel_size = voxel_size
        self.head_2d = build_head(head_2d) if head_2d is not None else None
        self.n_voxels = n_voxels
        self.voxel_size = voxel_size
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)
        self.ray_num=ray_num

        self.tr = Transformer(32, 4, 1, 64, 64, 0.10)

    def init_weights(self, pretrained=None):
        super().init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        self.neck.init_weights()
        self.neck_3d.init_weights()
        self.bbox_head.init_weights()
        if self.head_2d is not None:
            self.head_2d.init_weights()

    def extract_feat(self, img, img_metas, mode):
        batch_size = img.shape[0]
        img = img.reshape([-1] + list(img.shape)[2:])
        x = self.backbone(img)
        features_2d = self.head_2d.forward(x[-1], img_metas) if self.head_2d is not None else None
        x = self.neck(x)[0]
        x = x.reshape([batch_size, -1] + list(x.shape[1:])) # 18, 256, 120, 160

        stride = img.shape[-1] / x.shape[-1]
        assert stride == 4  # may be removed in the future
        stride = int(stride)

        volumes, valids = [], []
        for feature, img_meta in zip(x, img_metas): # img_metas 是什么？这个玩意运行了1次
            # use predicted pitch and roll for SUNRGBDTotal test
            angles = features_2d[0] if features_2d is not None and mode == 'test' else None
            projection = self._compute_projection(img_meta, stride, angles).to(x.device)
            points = get_points(
                n_voxels=torch.tensor(self.n_voxels),
                voxel_size=torch.tensor(self.voxel_size),
                origin=torch.tensor(img_meta['lidar2img']['origin'])
            ).to(x.device)
            height = img_meta['img_shape'][0] // stride
            width = img_meta['img_shape'][1] // stride
            volume, valid, pairs = backproject(feature[:, :, :height, :width], points, projection)
            volume = volume.sum(dim=0)
            valid = valid.sum(dim=0)
            volume = volume / valid
            valid = valid > 0
            volume[:, ~valid[0]] = .0
            volumes.append(volume)
            valids.append(valid)
        x = torch.stack(volumes)
        valids = torch.stack(valids)

        # new tr
        if pairs:
            pairs1, pairs2 = pairs
            length = self.ray_num if self.ray_num < pairs1.shape[0] else pairs1.shape[0]
            inds = torch.randperm(pairs1.shape[0])[:length]
            pairs1 = pairs1[inds]
            pairs2 = pairs2[inds]
            x_idx = pairs1[:, 0]
            y_idx = pairs1[:, 1]
            z_idx = pairs1[:, 2]
            x_idx2 = pairs2[:, 0]
            y_idx2 = pairs2[:, 1]
            z_idx2 = pairs2[:, 2]
            # 使用高级索引提取每个位置的特征
            # 这里相当于对每个通道从 [40, 40, 40] 空间中提取相应的值
            y = []
            yy = []
            # for i in range(x.shape[0]):
            features = x[0, :, x_idx, y_idx, z_idx]  # [256, n]
            # 转置为 [n, 256]
            features = features.permute(1, 0)  # [n, 256]
            # print(features.shape)
            f2 = x[0, :, x_idx2, y_idx2, z_idx2]
            f2 = f2.permute(1,0)
                # y.append(features)
                # yy.append(f2)
            # y = torch.stack(y) 
            # yy = torch.stack(yy)
            f = torch.cat([features,f2], dim=-1)#.reshape(-1,16,32) # 这个地方很难说是对的
            ff = self.tr(f).reshape(length, 256, 2)
            f1, f2 = ff[:,:,0], ff[:,:,1]

            for i in range(256):  # 更新每个通道的特征到体素
                x[0, i].index_put_((x_idx, y_idx, z_idx), f1[:, i]+x[0,i,x_idx,y_idx,z_idx], accumulate=True)
                x[0, i].index_put_((x_idx2, y_idx2, z_idx2), f2[:, i]+x[0,i,x_idx2,y_idx2,z_idx2], accumulate=True)

        x = self.neck_3d(x)
        return x, valids, features_2d

    def forward_train(self, img, img_metas, gt_bboxes_3d, gt_labels_3d, **kwargs):
        x, valids, features_2d = self.extract_feat(img, img_metas, 'train')
        losses = self.bbox_head.forward_train(x, valids.float(), img_metas, gt_bboxes_3d, gt_labels_3d)
        if self.head_2d is not None:
            losses.update(self.head_2d.loss(*features_2d, img_metas))
        return losses

    def forward_test(self, img, img_metas, **kwargs):
        # not supporting aug_test for now
        return self.simple_test(img, img_metas)

    def simple_test(self, img, img_metas):
        x, valids, features_2d = self.extract_feat(img, img_metas, 'test')
        x = self.bbox_head(x)
        bbox_list = self.bbox_head.get_bboxes(*x, valids.float(), img_metas)
        bbox_results = [
            bbox3d2result(det_bboxes, det_scores, det_labels)
            for det_bboxes, det_scores, det_labels in bbox_list
        ]
        if self.head_2d is not None:
            angles, layouts = self.head_2d.get_bboxes(*features_2d, img_metas)
            for i in range(len(img)):
                bbox_results[i]['angles'] = angles[i]
                bbox_results[i]['layout'] = layouts[i]
        return bbox_results

    def aug_test(self, imgs, img_metas):
        pass

    def show_results(self, *args, **kwargs):
        pass

    @staticmethod
    def _compute_projection(img_meta, stride, angles):
        projection = []
        intrinsic = torch.tensor(img_meta['lidar2img']['intrinsic'][:3, :3])
        ratio = img_meta['ori_shape'][0] / (img_meta['img_shape'][0] / stride)
        intrinsic[:2] /= ratio
        # use predicted pitch and roll for SUNRGBDTotal test
        if angles is not None:
            extrinsics = []
            for angle in angles:
                extrinsics.append(get_extrinsics(angle).to(intrinsic.device))
        else:
            extrinsics = map(torch.tensor, img_meta['lidar2img']['extrinsic'])
        for extrinsic in extrinsics:
            projection.append(intrinsic @ extrinsic[:3])
        return torch.stack(projection)


@torch.no_grad()
def get_points(n_voxels, voxel_size, origin):
    points = torch.stack(torch.meshgrid([
        torch.arange(n_voxels[0]),
        torch.arange(n_voxels[1]),
        torch.arange(n_voxels[2])
    ]))
    new_origin = origin - n_voxels / 2. * voxel_size
    points = points * voxel_size.view(3, 1, 1, 1) + new_origin.view(3, 1, 1, 1)
    return points

# modify from https://github.com/magicleap/Atlas/blob/master/atlas/model.py
def backproject_back(features, points, projection):
    n_images, n_channels, height, width = features.shape
    n_x_voxels, n_y_voxels, n_z_voxels = points.shape[-3:]
    points = points.view(1, 3, -1).expand(n_images, 3, -1)
    points = torch.cat((points, torch.ones_like(points[:, :1])), dim=1)
    points_2d_3 = torch.bmm(projection, points)
    x = (points_2d_3[:, 0] / points_2d_3[:, 2]).round().long()
    y = (points_2d_3[:, 1] / points_2d_3[:, 2]).round().long()
    z = points_2d_3[:, 2]
    valid = (x >= 0) & (y >= 0) & (x < width) & (y < height) & (z > 0)
    volume = torch.zeros((n_images, n_channels, points.shape[-1]), device=features.device)
    for i in range(n_images):
        volume[i, :, valid[i]] = features[i, :, y[i, valid[i]], x[i, valid[i]]] 
        # 这个投影过程好像没有用深度啊，所以能不能按照射线采样？？？
        # 改在上一句上，加个tr
    volume = volume.view(n_images, n_channels, n_x_voxels, n_y_voxels, n_z_voxels)
    valid = valid.view(n_images, 1, n_x_voxels, n_y_voxels, n_z_voxels)
    return volume, valid

def backproject(features, points, projection):
    n_images, n_channels, height, width = features.shape
    n_x_voxels, n_y_voxels, n_z_voxels = points.shape[-3:]

    voxel_coords = torch.stack(torch.meshgrid(
        torch.arange(n_x_voxels), 
        torch.arange(n_y_voxels), 
        torch.arange(n_z_voxels)
    ), dim=-1).view(-1, 3).float().to(features.device)  # Shape: [N_voxels, 3]
    voxel_coords = voxel_coords.unsqueeze(0).expand(n_images, -1, -1)  # Shape: [n_images, N_voxels, 3]
    
    points = points.view(1, 3, -1).expand(n_images, 3, -1)
    points = torch.cat((points, torch.ones_like(points[:, :1])), dim=1)
    points_2d_3 = torch.bmm(projection, points) # 这块，改成不是round()的结构呢？
    x = (points_2d_3[:, 0] / points_2d_3[:, 2]).round().long()
    y = (points_2d_3[:, 1] / points_2d_3[:, 2]).round().long()
    z = points_2d_3[:, 2]

    valid = (x >= 0) & (y >= 0) & (x < width) & (y < height) & (z > 0)
    volume = torch.zeros((n_images, n_channels, points.shape[-1]), device=features.device)

    # Step 2: Create a tensor to hold a flat representation of (x, y) coordinates and their depth
    flat_coords = torch.stack([x, y], dim=-1) 
    flat_z = z.view(n_images, -1)  # Flatten the z (depth) values
    voxel_pairs1 = []
    voxel_pairs2 = []

    for i in range(n_images):
        volume[i, :, valid[i]] = features[i, :, y[i, valid[i]], x[i, valid[i]]] 
        # Step 3: Detect duplicated (x, y) coordinates for each image
        flat_coords_i = flat_coords[i][valid[i]]
        unique_coords, inverse_indices, counts = torch.unique(flat_coords_i, return_inverse=True, return_counts=True, dim=0)
        voxel_indices = torch.arange(n_x_voxels * n_y_voxels * n_z_voxels, device=features.device)[valid[i]]


        # Step 4: Find indices of duplicated (x, y) coordinates
        duplicate_indices = torch.nonzero(counts > 1).squeeze()

        # Step 5: For each duplicated (x, y), compare the depth (z) values of colliding points
        # if len(duplicate_indices) > 0:
        #     for idx in duplicate_indices:
        #         # Get all 3D points that map to the same (x, y) pixel
        #         colliding_points = torch.nonzero(inverse_indices == idx).squeeze()
        #         colliding_depths = flat_z[i, colliding_points]
        #         print(f"Image {i}, pixel {unique_coords[idx].tolist()} has multiple points with depths: {colliding_depths.tolist()}")

        if len(duplicate_indices.shape)<1:
            continue
        for idx in duplicate_indices:
            # 获取重复的体素索引
            duplicate_voxel_indices = voxel_indices[inverse_indices == idx]

            if len(duplicate_voxel_indices) > 1:
                # 只保留前两个重复的体素对
                # voxel_pairs.append(duplicate_voxel_indices[:2].tolist())
                voxel1_idx, voxel2_idx = duplicate_voxel_indices[:2]
                # 获取体素的三维坐标
                voxel1_coord = voxel_coords[i, voxel1_idx].view(3).long()  # (x1, y1, z1)
                voxel2_coord = voxel_coords[i, voxel2_idx].view(3).long()  
                voxel_pairs1.append(voxel1_coord)
                voxel_pairs2.append(voxel2_coord)
    volume = volume.view(n_images, n_channels, n_x_voxels, n_y_voxels, n_z_voxels)
    valid = valid.view(n_images, 1, n_x_voxels, n_y_voxels, n_z_voxels)
    assert len(voxel_pairs1) == len(voxel_pairs2)
    if len(voxel_pairs1)<1:
        return volume, valid, None
    voxel_pairs1 = torch.stack(voxel_pairs1)
    voxel_pairs2 = torch.stack(voxel_pairs2)
    voxel_pairs = [voxel_pairs1, voxel_pairs2]
    
    # for i in range(n_images):
        # 这个投影过程好像没有用深度啊，所以能不能按照射线采样？？？
        # 改在上一句上，加个tr
    return volume, valid, voxel_pairs






# for SUNRGBDTotal test
def get_extrinsics(angles):
    yaw = angles.new_zeros(())
    pitch, roll = angles
    r = angles.new_zeros((3, 3))
    r[0, 0] = torch.cos(yaw) * torch.cos(pitch)
    r[0, 1] = torch.sin(yaw) * torch.sin(roll) - torch.cos(yaw) * torch.cos(roll) * torch.sin(pitch)
    r[0, 2] = torch.cos(roll) * torch.sin(yaw) + torch.cos(yaw) * torch.sin(pitch) * torch.sin(roll)
    r[1, 0] = torch.sin(pitch)
    r[1, 1] = torch.cos(pitch) * torch.cos(roll)
    r[1, 2] = -torch.cos(pitch) * torch.sin(roll)
    r[2, 0] = -torch.cos(pitch) * torch.sin(yaw)
    r[2, 1] = torch.cos(yaw) * torch.sin(roll) + torch.cos(roll) * torch.sin(yaw) * torch.sin(pitch)
    r[2, 2] = torch.cos(yaw) * torch.cos(roll) - torch.sin(yaw) * torch.sin(pitch) * torch.sin(roll)

    # follow Total3DUnderstanding
    t = angles.new_tensor([[0., 0., 1.], [0., -1., 0.], [-1., 0., 0.]])
    r = t @ r.T
    # follow DepthInstance3DBoxes
    r = r[:, [2, 0, 1]]
    r[2] *= -1
    extrinsic = angles.new_zeros((4, 4))
    extrinsic[:3, :3] = r
    extrinsic[3, 3] = 1.
    return extrinsic

