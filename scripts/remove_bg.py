import os
import tqdm
import cv2
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

class BackgroundRemoval():
    def __init__(self, device='cuda'):

        from carvekit.api.high import HiInterface
        self.interface = HiInterface(
            object_type="object",  # Can be "object" or "hairs-like".
            batch_size_seg=5,
            batch_size_matting=1,
            device=device,
            seg_mask_size=640,  # Use 640 for Tracer B7 and 320 for U2Net
            matting_mask_size=2048,
            trimap_prob_threshold=231,
            trimap_dilation=30,
            trimap_erosion_iters=5,
            fp16=True,
        )
    
    @torch.no_grad()
    def __call__(self, image):
        # image: PIL Image
        image = self.interface([image])[0]
        image = np.array(image)
        return image

parser = argparse.ArgumentParser()
parser.add_argument('path', type=str)

opt = parser.parse_args()

if opt.path[-1] == '/':
    opt.path = opt.path[:-1]

out_dir = os.path.join(os.path.dirname(opt.path), f'mask')
os.makedirs(out_dir, exist_ok=True)

print(f'[INFO] removing background: {opt.path} --> {out_dir}')

model = BackgroundRemoval()

def run_image(img_path):
    # img: filepath
    image = Image.open(img_path)
    carved_image = model(image) # [H, W, 4]
    mask = (carved_image[..., -1] > 0).astype(np.uint8) * 255 # [H, W]
    out_path = os.path.join(out_dir, os.path.splitext(os.path.basename(img_path))[0] + '.png')
    cv2.imwrite(out_path, mask)

img_paths = glob.glob(os.path.join(opt.path, '*'))
for img_path in tqdm.tqdm(img_paths):
    run_image(img_path)
