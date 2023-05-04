import os
import cv2
import glob
import json
import tqdm
import random
import numpy as np
from scipy.spatial.transform import Slerp, Rotation

import trimesh

import torch
from torch.utils.data import DataLoader

from .utils import get_rays, create_dodecahedron_cameras
from .colmap_utils import *

def rotmat(a, b):
	a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
	v = np.cross(a, b)
	c = np.dot(a, b)
	# handle exception for the opposite direction input
	if c < -1 + 1e-10:
		return rotmat(a + np.random.uniform(-1e-2, 1e-2, 3), b)
	s = np.linalg.norm(v)
	kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
	return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2 + 1e-10))


def center_poses(poses, pts3d=None, enable_cam_center=False):
    
    def normalize(v):
        return v / (np.linalg.norm(v) + 1e-10)

    if pts3d is None or enable_cam_center:
        center = poses[:, :3, 3].mean(0)
    else:
        center = pts3d.mean(0)
        
    
    up = normalize(poses[:, :3, 1].mean(0)) # (3)
    R = rotmat(up, [0, 0, 1])
    R = np.pad(R, [0, 1])
    R[-1, -1] = 1
    
    poses[:, :3, 3] -= center
    poses_centered = R @ poses # (N_images, 4, 4)

    if pts3d is not None:
        pts3d_centered = (pts3d - center) @ R[:3, :3].T
        # pts3d_centered = pts3d @ R[:3, :3].T - center
        return poses_centered, pts3d_centered

    return poses_centered


def visualize_poses(poses, size=0.05, bound=1, points=None):
    # poses: [B, 4, 4]

    axes = trimesh.creation.axis(axis_length=4)
    box = trimesh.primitives.Box(extents=[2*bound]*3).as_outline()
    box.colors = np.array([[128, 128, 128]] * len(box.entities))
    objects = [axes, box]

    if bound > 1:
        unit_box = trimesh.primitives.Box(extents=[2]*3).as_outline()
        unit_box.colors = np.array([[128, 128, 128]] * len(unit_box.entities))
        objects.append(unit_box)

    for pose in poses:
        # a camera is visualized with 8 line segments.
        pos = pose[:3, 3]
        a = pos + size * pose[:3, 0] + size * pose[:3, 1] - size * pose[:3, 2]
        b = pos - size * pose[:3, 0] + size * pose[:3, 1] - size * pose[:3, 2]
        c = pos - size * pose[:3, 0] - size * pose[:3, 1] - size * pose[:3, 2]
        d = pos + size * pose[:3, 0] - size * pose[:3, 1] - size * pose[:3, 2]

        dir = (a + b + c + d) / 4 - pos
        dir = dir / (np.linalg.norm(dir) + 1e-8)
        o = pos + dir * 3

        segs = np.array([[pos, a], [pos, b], [pos, c], [pos, d], [a, b], [b, c], [c, d], [d, a], [pos, o]])
        segs = trimesh.load_path(segs)
        objects.append(segs)

    if points is not None:
        print('[visualize points]', points.shape, points.dtype, points.min(0), points.max(0))
        colors = np.zeros((points.shape[0], 4), dtype=np.uint8)
        colors[:, 2] = 255 # blue
        colors[:, 3] = 30 # transparent
        objects.append(trimesh.PointCloud(points, colors))

    # tmp: verify mesh matches the points
    # mesh = trimesh.load('trial_garden_colmap/mesh_stage0/mesh.ply')
    # objects.append(mesh)

    scene = trimesh.Scene(objects)
    scene.set_camera(distance=bound, center=[0, 0, 0])
    scene.show()


class ColmapDataset:
    def __init__(self, opt, device, type='train', n_test=24):
        super().__init__()
        
        self.opt = opt
        self.device = device
        self.type = type # train, val, test
        self.downscale = opt.downscale
        self.preload = opt.preload # preload data into GPU
        self.scale = opt.scale # camera radius scale to make sure camera are inside the bounding box.
        # self.offset = opt.offset # camera offset
        self.fp16 = opt.fp16 # if preload, load into fp16.
        self.root_path = opt.path # contains "colmap_sparse"

        self.training = self.type in ['train', 'all', 'trainval']

        # locate colmap dir
        candidate_paths = [
            os.path.join(self.root_path, "colmap_sparse", "0"),
            os.path.join(self.root_path, "sparse", "0"),
            os.path.join(self.root_path, "colmap"),
        ]
        
        self.colmap_path = None
        for path in candidate_paths:
            if os.path.exists(path):
                self.colmap_path = path
                break

        if self.colmap_path is None:
            raise ValueError(f"Cannot find colmap sparse output under {self.root_path}, please run colmap first!")

        camdata = read_cameras_binary(os.path.join(self.colmap_path, 'cameras.bin'))

        # read image size (assume all images are of the same shape!)
        self.H = int(round(camdata[1].height / self.downscale))
        self.W = int(round(camdata[1].width / self.downscale))
        print(f'[INFO] ColmapDataset: image H = {self.H}, W = {self.W}')

        # read image paths
        imdata = read_images_binary(os.path.join(self.colmap_path, "images.bin"))
        imkeys = np.array(sorted(imdata.keys()))

        img_names = [os.path.basename(imdata[k].name) for k in imkeys]
        img_folder = os.path.join(self.root_path, f"images_{self.downscale}")
        if not os.path.exists(img_folder):
            img_folder = os.path.join(self.root_path, "images")
        img_paths = np.array([os.path.join(img_folder, name) for name in img_names])

        # only keep existing images
        exist_mask = np.array([os.path.exists(f) for f in img_paths])
        print(f'[INFO] {exist_mask.sum()} image exists in all {exist_mask.shape[0]} colmap entries.')
        imkeys = imkeys[exist_mask]
        img_paths = img_paths[exist_mask]

        # load masks
        mask_folder = os.path.join(self.root_path, 'mask')
        if os.path.exists(mask_folder):
            print(f'[INFO] use mask under folder {mask_folder}')
            mask_paths = np.array([os.path.join(self.root_path, 'mask', os.path.splitext(os.path.basename(p))[0] + '.png') for p in img_paths])
        else:
            mask_paths = None

        # read intrinsics
        intrinsics = []
        for k in imkeys:
            cam = camdata[imdata[k].camera_id]
            if cam.model in ['SIMPLE_RADIAL', 'SIMPLE_PINHOLE']:
                fl_x = fl_y = cam.params[0] / self.downscale
                cx = cam.params[1] / self.downscale
                cy = cam.params[2] / self.downscale
            elif cam.model in ['PINHOLE', 'OPENCV']:
                fl_x = cam.params[0] / self.downscale
                fl_y = cam.params[1] / self.downscale
                cx = cam.params[2] / self.downscale
                cy = cam.params[3] / self.downscale
            else:
                raise ValueError(f"Unsupported colmap camera model: {cam.model}")
            intrinsics.append(np.array([fl_x, fl_y, cx, cy], dtype=np.float32))
        
        self.intrinsics = torch.from_numpy(np.stack(intrinsics)) # [N, 4]

        # read poses
        poses = []
        for k in imkeys:
            P = np.eye(4, dtype=np.float64)
            P[:3, :3] = imdata[k].qvec2rotmat()
            P[:3, 3] = imdata[k].tvec
            poses.append(P)
        
        poses = np.linalg.inv(np.stack(poses, axis=0)) # [N, 4, 4]

        # read sparse points
        ptsdata = read_points3d_binary(os.path.join(self.colmap_path, "points3D.bin"))
        ptskeys = np.array(sorted(ptsdata.keys()))
        pts3d = np.array([ptsdata[k].xyz for k in ptskeys]) # [M, 3]
        self.ptserr = np.array([ptsdata[k].error for k in ptskeys]) # [M]
        self.mean_ptserr = np.mean(self.ptserr)

        # center pose
        self.poses, self.pts3d = center_poses(poses, pts3d, self.opt.enable_cam_center)
        print(f'[INFO] ColmapDataset: load poses {self.poses.shape}, points {self.pts3d.shape}')

        # rectify convention...
        self.poses[:, :3, 1:3] *= -1
        self.poses = self.poses[:, [1, 0, 2, 3], :]
        self.poses[:, 2] *= -1

        self.pts3d = self.pts3d[:, [1, 0, 2]]
        self.pts3d[:, 2] *= -1

        # auto-scale
        if self.scale == -1:
            self.scale = 1 / np.linalg.norm(self.poses[:, :3, 3], axis=-1).min()
            print(f'[INFO] ColmapDataset: auto-scale {self.scale:.4f}')

        self.poses[:, :3, 3] *= self.scale
        self.pts3d *= self.scale

        # use pts3d to estimate aabb
        # self.pts_aabb = np.concatenate([np.percentile(self.pts3d, 1, axis=0), np.percentile(self.pts3d, 99, axis=0)]) # [6]
        self.pts_aabb = np.concatenate([np.min(self.pts3d, axis=0), np.max(self.pts3d, axis=0)]) # [6]
        if np.abs(self.pts_aabb).max() > self.opt.bound:
            print(f'[WARN] ColmapDataset: estimated AABB {self.pts_aabb.tolist()} exceeds provided bound {self.opt.bound}! Consider improving --bound to make scene included in trainable region.')

        # process pts3d into sparse depth data.

        if self.type != 'test':
        
            self.cam_near_far = [] # always extract this infomation
            self.sparse_depth_info = [] if self.opt.enable_sparse_depth else None
            self.dense_depth_info = [] if self.opt.enable_dense_depth else None

            print(f'[INFO] extracting sparse depth info...')
            # map from colmap points3d dict key to dense array index
            pts_key_to_id = np.ones(ptskeys.max() + 1, dtype=np.int64) * len(ptskeys)
            pts_key_to_id[ptskeys] = np.arange(0, len(ptskeys))
            # loop imgs
            _mean_valid_sparse_depth = 0
            for i, k in enumerate(tqdm.tqdm(imkeys)):
                xys = imdata[k].xys
                xys = np.stack([xys[:, 1], xys[:, 0]], axis=-1) # invert x and y convention...
                pts = imdata[k].point3D_ids

                mask = (pts != -1) & (xys[:, 0] >= 0) & (xys[:, 0] < camdata[1].height) & (xys[:, 1] >= 0) & (xys[:, 1] < camdata[1].width)

                assert mask.any(), 'every image must contain sparse point'
                
                valid_ids = pts_key_to_id[pts[mask]]
                pts = self.pts3d[valid_ids] # points [M, 3]
                err = self.ptserr[valid_ids] # err [M]
                xys = xys[mask] # pixel coord [M, 2], float, original resolution!

                xys = np.round(xys / self.downscale).astype(np.int32) # downscale
                xys[:, 0] = xys[:, 0].clip(0, self.H - 1)
                xys[:, 1] = xys[:, 1].clip(0, self.W - 1)
                
                # calc the depth
                P = self.poses[i]
                depth = (P[:3, 3] - pts) @ P[:3, 2]

                # calc weight
                weight = 2 * np.exp(- (err / self.mean_ptserr) ** 2)

                _mean_valid_sparse_depth += depth.shape[0]

                # camera near far
                # self.cam_near_far.append([np.percentile(depth, 0.1), np.percentile(depth, 99.9)])
                self.cam_near_far.append([np.min(depth), np.max(depth)])

                # sparse depth info
                if self.opt.enable_sparse_depth:
                    self.sparse_depth_info.append([
                        torch.from_numpy(xys.astype(np.float32)),
                        torch.from_numpy(depth.astype(np.float32)),
                        torch.from_numpy(weight.astype(np.float32))
                    ])

                # dense depth info
                if self.opt.enable_dense_depth:

                    depth_path = os.path.join(self.root_path, 'depths', os.path.splitext(os.path.basename(imdata[k].name))[0] + '.npy')

                    if not os.path.exists(depth_path):
                        # call depth estimation automatically.
                        raise RuntimeError('[ERROR] depth estimation not found, please run `python depth_tools/extract_depth.py`')

                    dense_depth = np.load(depth_path) # [h, w]

                    # interpolate to current resolution
                    dense_depth = cv2.resize(dense_depth, (self.W, self.H), interpolation=cv2.INTER_LINEAR)

                    # NOTE: this can be very INACCURATE... in loss we do another LS and map dense to rendered depth.
                    # map dense to sparse depth by solving a weighted least square problem
                    from sklearn.linear_model import RANSACRegressor

                    X = dense_depth[tuple(xys.T)].reshape(-1, 1) # [M], dense
                    Y = depth.reshape(-1) # [M], sparse
                    W = weight.reshape(-1)

                    LR = RANSACRegressor().fit(X, Y, W)
                    scale = LR.estimator_.coef_[0]
                    bias = LR.estimator_.intercept_

                    score = np.mean((X * scale + bias - Y) ** 2)

                    # must be wrong... use the most confident two samples.
                    if scale < 0:
                        idx_by_conf = np.argsort(W)[::-1]
                        x0, y0 = X[idx_by_conf[0]][0], Y[idx_by_conf[0]]
                        x1, y1 = X[idx_by_conf[1]][0], Y[idx_by_conf[1]]
                        scale = (y0 - y1) / (x0 - x1)
                        bias = y0 - x0 * scale
                        score = np.mean((X * scale + bias - Y) ** 2)
                    
                        # if still wrong, use the most confident ONE sample...
                        if scale < 0:
                            scale = y0 / x0
                            bias = 0
                            score = np.mean((X * scale + bias - Y) ** 2)

                    print(f'[INFO] estimate dense depth scale by linear regression: MSE = {score:.4f}, scale = {scale:.4f}, bias = {bias:.4f}')

                    dense_depth = dense_depth * scale + bias

                    self.dense_depth_info.append(dense_depth)

            print(f'[INFO] extracted {_mean_valid_sparse_depth / len(imkeys):.2f} valid sparse depth on average per image')

            self.cam_near_far = torch.from_numpy(np.array(self.cam_near_far, dtype=np.float32)) # [N, 2]

            if self.opt.enable_sparse_depth:
                self.sparse_depth_info = np.array(self.sparse_depth_info, dtype=object) # just to support array-slicing

            elif self.opt.enable_dense_depth:
                self.dense_depth_info = torch.from_numpy(np.stack(self.dense_depth_info, axis=0))

        else: # test time: no depth info
            self.sparse_depth_info = None
            self.dense_depth_info = None
            self.cam_near_far = None


        # make split
        if self.type == 'test':
            
            poses = []

            if self.opt.camera_traj == 'circle':

                print(f'[INFO] use circular camera traj for testing.')
                
                # circle 360 pose
                # radius = np.linalg.norm(self.poses[:, :3, 3], axis=-1).mean(0)
                radius = 0.1
                theta = np.deg2rad(80)
                for i in range(100):
                    phi = np.deg2rad(i / 100 * 360)
                    center = np.array([
                        radius * np.sin(theta) * np.sin(phi),
                        radius * np.sin(theta) * np.cos(phi),
                        radius * np.cos(theta),
                    ])
                    # look at
                    def normalize(v):
                        return v / (np.linalg.norm(v) + 1e-10)
                    forward_v = normalize(center)
                    up_v = np.array([0, 0, 1])
                    right_v = normalize(np.cross(forward_v, up_v))
                    up_v = normalize(np.cross(right_v, forward_v))
                    # make pose
                    pose = np.eye(4)
                    pose[:3, :3] = np.stack((right_v, up_v, forward_v), axis=-1)
                    pose[:3, 3] = center
                    poses.append(pose)
                
                self.poses = np.stack(poses, axis=0)
            
            # choose some random poses, and interpolate between.
            else:

                fs = np.random.choice(len(self.poses), 5, replace=False)
                pose0 = self.poses[fs[0]]
                for i in range(1, len(fs)):
                    pose1 = self.poses[fs[i]]
                    rots = Rotation.from_matrix(np.stack([pose0[:3, :3], pose1[:3, :3]]))
                    slerp = Slerp([0, 1], rots)    
                    for i in range(n_test + 1):
                        ratio = np.sin(((i / n_test) - 0.5) * np.pi) * 0.5 + 0.5
                        pose = np.eye(4, dtype=np.float32)
                        pose[:3, :3] = slerp(ratio).as_matrix()
                        pose[:3, 3] = (1 - ratio) * pose0[:3, 3] + ratio * pose1[:3, 3]
                        poses.append(pose)
                    pose0 = pose1

                self.poses = np.stack(poses, axis=0)

            # fix intrinsics for test case
            self.intrinsics = self.intrinsics[[0]].repeat(self.poses.shape[0], 1)

            self.images = None
        
        else:
            
            all_ids = np.arange(len(img_paths))
            val_ids = all_ids[::8]
            # val_ids = all_ids[::50]

            if self.type == 'train':
                train_ids = np.array([i for i in all_ids if i not in val_ids])
                self.poses = self.poses[train_ids]
                self.intrinsics = self.intrinsics[train_ids]
                img_paths = img_paths[train_ids]
                if mask_paths is not None:
                    mask_paths = mask_paths[train_ids]
                if self.sparse_depth_info is not None:
                    self.sparse_depth_info = self.sparse_depth_info[train_ids]
                if self.dense_depth_info is not None:
                    self.dense_depth_info = self.dense_depth_info[train_ids]
                if self.cam_near_far is not None:
                    self.cam_near_far = self.cam_near_far[train_ids]
            elif self.type == 'val':
                self.poses = self.poses[val_ids]
                self.intrinsics = self.intrinsics[val_ids]
                img_paths = img_paths[val_ids]
                if mask_paths is not None:
                    mask_paths = mask_paths[val_ids]
                if self.sparse_depth_info is not None:
                    self.sparse_depth_info = self.sparse_depth_info[val_ids]
                if self.dense_depth_info is not None:
                    self.dense_depth_info = self.dense_depth_info[val_ids]
                if self.cam_near_far is not None:
                    self.cam_near_far = self.cam_near_far[val_ids]
            # else: trainval use all.
            
            # read images
            self.images = []

            for i, f in enumerate(tqdm.tqdm(img_paths, desc=f'Loading {self.type} data')):

                image = cv2.imread(f, cv2.IMREAD_UNCHANGED) # [H, W, 3] o [H, W, 4]

                # add support for the alpha channel as a mask.
                if image.shape[-1] == 3: 
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)

                # if mask is available, load as the alpha channel
                if mask_paths is not None:
                    m_path = mask_paths[i]
                    mask = cv2.imread(m_path, cv2.IMREAD_UNCHANGED) # [H, W]
                    if len(mask.shape) == 2: 
                        mask = mask[..., None]
                    image = np.concatenate([image, mask[..., :1]], axis=-1)

                if image.shape[0] != self.H or image.shape[1] != self.W:
                    image = cv2.resize(image, (self.W, self.H), interpolation=cv2.INTER_AREA)

                self.images.append(image)

            self.images = np.stack(self.images, axis=0)
       
        # view all poses.
        if self.opt.vis_pose:
            visualize_poses(self.poses, bound=self.opt.bound, points=self.pts3d)

        self.poses = torch.from_numpy(self.poses.astype(np.float32)) # [N, 4, 4]

        if self.images is not None:
            self.images = torch.from_numpy(np.stack(self.images, axis=0).astype(np.uint8)) # [N, H, W, C]
        
        # perspective projection matrix
        self.near = self.opt.min_near
        self.far = 1000 # infinite
        aspect = self.W / self.H

        projections = []
        for intrinsic in self.intrinsics:
            y = self.H / (2.0 * intrinsic[1].item()) # fl_y
            projections.append(np.array([[1/(y*aspect), 0, 0, 0], 
                                        [0, -1/y, 0, 0],
                                        [0, 0, -(self.far+self.near)/(self.far-self.near), -(2*self.far*self.near)/(self.far-self.near)],
                                        [0, 0, -1, 0]], dtype=np.float32))
        self.projections = torch.from_numpy(np.stack(projections)) # [N, 4, 4]
        self.mvps = self.projections @ torch.inverse(self.poses)
    
        # tmp: dodecahedron_cameras for mesh visibility test
        dodecahedron_poses = create_dodecahedron_cameras()
        # visualize_poses(dodecahedron_poses, bound=self.opt.bound, points=self.pts3d)
        self.dodecahedron_poses = torch.from_numpy(dodecahedron_poses.astype(np.float32)) # [N, 4, 4]
        self.dodecahedron_mvps = self.projections[[0]] @ torch.inverse(self.dodecahedron_poses) # assume the same intrinsic

        if self.preload:
            self.intrinsics = self.intrinsics.to(self.device)
            self.poses = self.poses.to(self.device)
            if self.images is not None:
                self.images = self.images.to(self.device)
            if self.cam_near_far is not None:
                self.cam_near_far = self.cam_near_far.to(self.device)
            self.mvps = self.mvps.to(self.device)


    def collate(self, index):

        results = {'H': self.H, 'W': self.W}

        # if use sparse depth supervision, randomly use sparse depth.
        if self.opt.enable_sparse_depth and self.training and self.opt.stage == 0 and \
            random.random() > 0.9:

            depth_coords, depth, depth_weight = self.sparse_depth_info[index[0]]

            depth_coords = depth_coords.long().to(self.device)
            depth = depth.unsqueeze(0).to(self.device)
            depth_weight = depth_weight.unsqueeze(0).to(self.device)

            num_rays = len(depth_coords)
            poses = self.poses[index].to(self.device) # [N, 4, 4]
            intrinsics = self.intrinsics[index].to(self.device) # [N, 4]
            rays = get_rays(poses, intrinsics, self.H, self.W, num_rays, coords=depth_coords)

        else:

            depth = None
            depth_weight = None

            if self.training and self.opt.stage == 0:
                # randomly sample over images too
                num_rays = self.opt.num_rays

                if self.opt.random_image_batch:
                    index = torch.randint(0, len(self.poses), size=(num_rays,), device=self.device)

            else:
                num_rays = -1

            poses = self.poses[index].to(self.device) # [1/N, 4, 4]
            intrinsics = self.intrinsics[index].to(self.device) # [1/N, 4]
            rays = get_rays(poses, intrinsics, self.H, self.W, num_rays, patch_size=self.opt.patch_size)

        if self.opt.stage > 0:
            mvp = self.mvps[index].to(self.device)
            results['mvp'] = mvp

        if self.images is not None:
            
            if self.training and self.opt.stage == 0:
                images = self.images[index, rays['j'], rays['i']].float().to(self.device) / 255 # [N, 3/4]

                if self.opt.enable_dense_depth:
                    depth = self.dense_depth_info[index, rays['j'], rays['i']].float().to(self.device) # [N]
            else:
                images = self.images[index].squeeze(0).float().to(self.device) / 255 # [H, W, 3/4]

            if self.training:
                C = self.images.shape[-1]
                images = images.view(-1, C)

            results['images'] = images
        
        if self.opt.enable_cam_near_far and self.cam_near_far is not None:
            cam_near_far = self.cam_near_far[index].to(self.device) # [1/N, 2]
            results['cam_near_far'] = cam_near_far
        
        results['rays_o'] = rays['rays_o']
        results['rays_d'] = rays['rays_d']
        results['index'] = index

        if depth is not None:
            results['depth'] = depth
        
        if depth_weight is not None:
            results['depth_weight'] = depth_weight
        
        return results

    def dataloader(self):
        size = len(self.poses)
        loader = DataLoader(list(range(size)), batch_size=1, collate_fn=self.collate, shuffle=self.training, num_workers=0)
        loader._data = self # an ugly fix... we need to access error_map & poses in trainer.
        loader.has_gt = self.images is not None
        return loader