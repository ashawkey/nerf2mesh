import os
import glob
import tqdm
import argparse

from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('path', type=str)
parser.add_argument('--downscale', type=int, default=4)

opt = parser.parse_args()

in_dir = os.path.join(opt.path, f'images')
out_dir = os.path.join(opt.path, f'images_{opt.downscale}')

os.makedirs(out_dir, exist_ok=True)

def run_image(img_path):
    # img: filepath
    img = Image.open(img_path)
    W, H = img.size
    img = img.resize((W // opt.downscale, H // opt.downscale), Image.Resampling.BILINEAR)
    out_path = os.path.join(out_dir, os.path.basename(img_path))
    img.save(out_path)

img_paths = glob.glob(os.path.join(in_dir, '*'))
for img_path in tqdm.tqdm(img_paths):
    run_image(img_path)
