# outdoor
CUDA_VISIBLE_DEVICES=5 python main.py data/garden/ --workspace trial_360_garden   -O --data_format colmap --bound 16 --enable_cam_center --enable_cam_near_far --scale 0.3 --downscale 4 --stage 0 --lambda_entropy 1e-3 --clean_min_f 16 --clean_min_d 10 --visibility_mask_dilation 50
CUDA_VISIBLE_DEVICES=5 python main.py data/garden/ --workspace trial_360_garden   -O --data_format colmap --bound 16 --enable_cam_center --enable_cam_near_far --scale 0.3 --downscale 4 --stage 1 --iters 10000

CUDA_VISIBLE_DEVICES=5 python main.py data/stump/ --workspace trial_360_stump     -O --data_format colmap --bound 16 --enable_cam_center --enable_cam_near_far --scale 0.3 --downscale 4 --stage 0 --lambda_entropy 1e-3 --clean_min_f 16 --clean_min_d 10 --visibility_mask_dilation 50
CUDA_VISIBLE_DEVICES=5 python main.py data/stump/ --workspace trial_360_stump     -O --data_format colmap --bound 16 --enable_cam_center --enable_cam_near_far --scale 0.3 --downscale 4 --stage 1 --iters 10000

CUDA_VISIBLE_DEVICES=5 python main.py data/bicycle/ --workspace trial_360_bicycle -O --data_format colmap --bound 16 --enable_cam_center --enable_cam_near_far --scale 0.3 --downscale 4 --stage 0 --lambda_entropy 1e-3 --clean_min_f 16 --clean_min_d 10 --visibility_mask_dilation 50
CUDA_VISIBLE_DEVICES=5 python main.py data/bicycle/ --workspace trial_360_bicycle -O --data_format colmap --bound 16 --enable_cam_center --enable_cam_near_far --scale 0.3 --downscale 4 --stage 1 --iters 10000