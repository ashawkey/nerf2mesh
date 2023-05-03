import os
import cv2
import glob
import json
import tqdm
import numpy as np
from scipy.spatial.transform import Slerp, Rotation

import trimesh

import torch
from torch.utils.data import DataLoader

from .utils import get_rays, create_dodecahedron_cameras


# ref: https://github.com/NVlabs/instant-ngp/blob/b76004c8cf478880227401ae763be4c02f80b62f/include/neural-graphics-primitives/nerf_loader.h#L50
def nerf_matrix_to_ngp(pose, scale=0.33, offset=[0, 0, 0]):
    pose[:3, 3] = pose[:3, 3] * scale + np.array(offset)
    pose = pose.astype(np.float32)
    return pose

def visualize_poses(poses, size=0.1, bound=1):
    # poses: [B, 4, 4]

    axes = trimesh.creation.axis(axis_length=4)
    box = trimesh.primitives.Box(extents=[2*bound]*3).as_outline()
    box.colors = np.array([[128, 128, 128]] * len(box.entities))
    objects = [axes, box]

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

    trimesh.Scene(objects).show()

def load_K_Rt_from_P(P):
    
    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsic = np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]])

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsic, pose

class NeRFDataset:
    def __init__(self, opt, device, type='train', n_test=10):
        super().__init__()
        
        self.opt = opt
        self.device = device
        self.type = type # train, val, test
        self.downscale = opt.downscale
        self.root_path = opt.path
        self.preload = opt.preload # preload data into GPU
        self.scale = opt.scale # camera radius scale to make sure camera are inside the bounding box.
        self.offset = opt.offset # camera offset
        self.bound = opt.bound # bounding box half length, also used as the radius to random sample poses.
        self.fp16 = opt.fp16 # if preload, load into fp16.

        if self.scale == -1:
            print(f'[WARN] --data_format dtu cannot auto-choose --scale, use 1 as default.')
            self.scale = 1
            
        self.training = self.type in ['train', 'all', 'trainval']

        camera_dict = np.load(os.path.join(self.root_path, 'cameras_sphere.npz'))
        image_paths = sorted(glob.glob(os.path.join(self.root_path, 'image', '*.png')))
        mask_paths = sorted(glob.glob(os.path.join(self.root_path, 'mask', '*.png')))

        world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(len(image_paths))]
        scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(len(image_paths))]

        intrinsics = []
        poses = []

        for scale_mat, world_mat in zip(scale_mats, world_mats):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsic, pose = load_K_Rt_from_P(P)
            intrinsics.append(intrinsic)
            pose = nerf_matrix_to_ngp(pose, scale=self.scale, offset=self.offset)
            poses.append(pose)

        self.intrinsics = torch.from_numpy(np.stack(intrinsics)).float() # [N, 4]
        self.poses = np.stack(poses) # [N, 4, 4]

        self.poses[:, :3, 1:3] *= -1
        self.poses = self.poses[:, [1, 0, 2, 3], :]
        self.poses[:, 2] *= -1
        
        # we have to actually read an image to get H and W later.
        self.H = self.W = None
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
            self.H = self.W = 512

        # manually split a valid set (the first frame).
        else:
            if type == 'train':
                image_paths = image_paths[1:]
                mask_paths = mask_paths[1:]
                self.poses = self.poses[1:]
                self.intrinsics = self.intrinsics[1:]
            elif type == 'val':
                image_paths = image_paths[:1]
                mask_paths = mask_paths[:1]
                self.poses = self.poses[:1]
                self.intrinsics = self.intrinsics[:1]
            # else 'all' or 'trainval' : use all frames
        
            # read images
            self.images = []
            for i in tqdm.tqdm(range(len(image_paths)), desc=f'Loading {type} data'):

                f_path = image_paths[i]
                m_path = mask_paths[i]

                image = cv2.imread(f_path, cv2.IMREAD_UNCHANGED) # [H, W, 3]
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # if use mask, add as an alpha channel
                mask = cv2.imread(m_path, cv2.IMREAD_UNCHANGED) # [H, W, 3]
                image = np.concatenate([image, mask[..., :1]], axis=-1)
                
                if self.H is None or self.W is None:
                    self.H = image.shape[0] // self.downscale
                    self.W = image.shape[1] // self.downscale

                if image.shape[0] != self.H or image.shape[1] != self.W:
                    image = cv2.resize(image, (self.W, self.H), interpolation=cv2.INTER_AREA)

                self.images.append(image)
            
        if self.images is not None:
            self.images = torch.from_numpy(np.stack(self.images, axis=0).astype(np.uint8)) # [N, H, W, C]
        
        # [debug] uncomment to view all training poses.
        if self.opt.vis_pose:
            visualize_poses(self.poses, bound=self.opt.bound)

        self.poses = torch.from_numpy(self.poses.astype(np.float32)) # [N, 4, 4]
            
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
            self.mvps = self.mvps.to(self.device)


    def collate(self, index):

        B = len(index) # a list of length 1

        results = {'H': self.H, 'W': self.W}

        if self.training and self.opt.stage == 0:
            # randomly sample over images too
            num_rays = self.opt.num_rays

            if self.opt.random_image_batch:
                index = torch.randint(0, len(self.poses), size=(num_rays,), device=self.device)

        else:
            num_rays = -1

        poses = self.poses[index].to(self.device) # [N, 4, 4]
        intrinsics = self.intrinsics[index].to(self.device) # [1/N, 4]

        rays = get_rays(poses, intrinsics, self.H, self.W, num_rays, self.opt.patch_size)

        results['rays_o'] = rays['rays_o']
        results['rays_d'] = rays['rays_d']
        results['index'] = index

        if self.opt.stage > 0:
            mvp = self.mvps[index].to(self.device)
            results['mvp'] = mvp

        if self.images is not None:
            
            if self.training and self.opt.stage == 0:
                images = self.images[index, rays['j'], rays['i']].float().to(self.device) / 255 # [N, 3/4]
            else:
                images = self.images[index].squeeze(0).float().to(self.device) / 255 # [H, W, 3/4]

            if self.training:
                C = self.images.shape[-1]
                images = images.view(-1, C)

            results['images'] = images
            
        return results

    def dataloader(self):
        size = len(self.poses)
        loader = DataLoader(list(range(size)), batch_size=1, collate_fn=self.collate, shuffle=self.training, num_workers=0)
        loader._data = self # an ugly fix... we need to access error_map & poses in trainer.
        loader.has_gt = self.images is not None
        return loader