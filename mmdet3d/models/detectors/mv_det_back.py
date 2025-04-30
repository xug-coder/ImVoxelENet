import torch
from mmdet.models import DETECTORS, build_backbone, build_head, build_neck
from mmdet.models.detectors import BaseDetector

from mmdet3d.core import bbox3d2result


@DETECTORS.register_module()
class mvDet_back(BaseDetector):
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
            volume, valid = backproject(feature[:, :, :height, :width], points, projection)
            volume = volume.sum(dim=0)
            valid = valid.sum(dim=0)
            volume = volume / valid
            valid = valid > 0
            volume[:, ~valid[0]] = .0
            volumes.append(volume)
            valids.append(valid)
        x = torch.stack(volumes)
        valids = torch.stack(valids)
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

# modify from https://github.com/magicleap/Atlas/blob/master/atlas/model.py
def backproject(features, points, projection):
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
        volume[i, :, valid[i]] = volume[i, :, valid[i]] + features[i, :, y[i, valid[i]], x[i, valid[i]]] 
        # 这个投影过程好像没有用深度啊，所以能不能按照射线采样？？？
        # 改在上一句上，加个tr
    volume = volume.view(n_images, n_channels, n_x_voxels, n_y_voxels, n_z_voxels)
    valid = valid.view(n_images, 1, n_x_voxels, n_y_voxels, n_z_voxels)
    return volume, valid

def backproject_with_ray_tracing(features, points, projection, voxel_size=1.0):
    n_images, n_channels, height, width = features.shape
    n_x_voxels, n_y_voxels, n_z_voxels = points.shape[-3:]

    # Reshape points to (1, 3, num_voxels) and expand for each image
    points = points.view(1, 3, -1).expand(n_images, 3, -1)
    points = torch.cat((points, torch.ones_like(points[:, :1])), dim=1)

    # Project points to 2D image plane
    points_2d_3 = torch.bmm(projection, points)
    x = (points_2d_3[:, 0] / points_2d_3[:, 2]).round().long()
    y = (points_2d_3[:, 1] / points_2d_3[:, 2]).round().long()
    z = points_2d_3[:, 2]
    
    # Determine valid points within the image boundaries and in front of the camera
    valid = (x >= 0) & (y >= 0) & (x < width) & (y < height) & (z > 0)

    # Initialize a volume tensor for storing voxel features
    volume = torch.zeros((n_images, n_channels, points.shape[-1]), device=features.device)

    # For each image, trace rays and assign features to intersected voxels
    for i in range(n_images):
        # Get 2D feature map for image i
        img_features = features[i]

        # Find valid 2D pixel coordinates
        valid_x = x[i, valid[i]]
        valid_y = y[i, valid[i]]

        # Extract features from the feature map for valid pixels
        pixel_features = img_features[:, valid_y, valid_x]

        # Get the valid 3D voxel coordinates
        valid_points = points[:, :, valid[i]].view(3, -1)
        
        # Calculate ray direction for each valid voxel (from camera center to voxel)
        # Assuming camera is at origin for simplicity, else calculate camera position
        ray_directions = valid_points - torch.tensor([0, 0, 0], device=features.device).view(3, 1) # 初始位置要改
        ray_directions = ray_directions / ray_directions.norm(dim=0, keepdim=True)

        # Traverse along the ray and assign the same features to all intersected voxels
        for j in range(valid_points.shape[1]):
            voxel_center = valid_points[:, j]

            # Compute the number of steps (intersections with voxels along the ray)
            num_steps = int(voxel_center.norm().item() / voxel_size)

            # Assign the same pixel features along the ray
            for step in range(num_steps):
                intersected_voxel = voxel_center - ray_directions[:, j] * step * voxel_size
                voxel_idx = (intersected_voxel / voxel_size).round().long()

                # Only assign feature if voxel index is within bounds
                if all(0 <= voxel_idx[k] < [n_x_voxels, n_y_voxels, n_z_voxels][k] for k in range(3)):
                    volume[i, :, voxel_idx[0] * n_y_voxels * n_z_voxels + voxel_idx[1] * n_z_voxels + voxel_idx[2]] = pixel_features[:, j]

    # Reshape volume back to 3D grid
    volume = volume.view(n_images, n_channels, n_x_voxels, n_y_voxels, n_z_voxels)
    
    # Reshape valid mask
    valid = valid.view(n_images, 1, n_x_voxels, n_y_voxels, n_z_voxels)
    
    return volume, valid


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
