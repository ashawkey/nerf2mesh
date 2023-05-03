import os
import glob
import tqdm
from PIL import Image
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from dpt import DPTDepthModel

IMAGE_SIZE = 384

parser = argparse.ArgumentParser()
parser.add_argument('path', type=str)
parser.add_argument('--out_dir', type=str)
parser.add_argument('--ckpt', type=str, default='./depth_tools/omnidata_dpt_depth_v2.ckpt')

opt = parser.parse_args()

if opt.path[-1] == '/':
    opt.path = opt.path[:-1]

out_dir = os.path.join(os.path.dirname(opt.path), f'depths')

os.makedirs(out_dir, exist_ok=True)

map_location = (lambda storage, loc: storage.cuda()) if torch.cuda.is_available() else torch.device('cpu')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = DPTDepthModel(backbone='vitb_rn50_384') # DPT Hybrid

print(f'[INFO] loading checkpoint from {opt.ckpt}')
checkpoint = torch.load(opt.ckpt, map_location=map_location)

if 'state_dict' in checkpoint:
    state_dict = {}
    for k, v in checkpoint['state_dict'].items():
        state_dict[k[6:]] = v
else:
    state_dict = checkpoint

model.load_state_dict(state_dict)
model.to(device)
model.eval()

trans_totensor = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=0.5, std=0.5)
])


@torch.no_grad()
def run_image(img_path):
    # img: filepath
    img = Image.open(img_path)
    W, H = img.size
    img_input = trans_totensor(img).unsqueeze(0).to(device)

    depth = model(img_input)

    depth = F.interpolate(depth.unsqueeze(1), size=(H, W), mode='bicubic', align_corners=False)
    depth = depth.squeeze().cpu().numpy()

    out_path = os.path.join(out_dir, os.path.splitext(os.path.basename(img_path))[0]) + '.npy'

    # plt.matshow(depth)
    # plt.show()

    # plt.matshow(img_input.detach().cpu()[0].permute(1,2,0).numpy())
    # plt.show()
    # print(f'[INFO] {out_path} {depth.min()} {depth.max()} {depth.shape}')

    np.save(out_path, depth)


img_paths = glob.glob(os.path.join(opt.path, '*'))
for img_path in tqdm.tqdm(img_paths):
    run_image(img_path)