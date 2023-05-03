# indoor
CUDA_VISIBLE_DEVICES=4 python main.py data/room/ --workspace trial_sdf_360_room -O --data_format colmap --bound 8 --enable_cam_center --enable_cam_near_far --scale 0.2 --downscale 4 --stage 0 --lambda_entropy 1e-3 --clean_min_f 16 --clean_min_d 10 --visibility_mask_dilation 50 --sdf
CUDA_VISIBLE_DEVICES=4 python main.py data/room/ --workspace trial_sdf_360_room -O --data_format colmap --bound 8 --enable_cam_center --enable_cam_near_far --scale 0.2 --downscale 4 --stage 1 --iters 10000 --lambda_lap 1e-3 --lambda_normal 1e-3 --sdf

CUDA_VISIBLE_DEVICES=4 python main.py data/bonsai/ --workspace trial_sdf_360_bonsai -O --data_format colmap --bound 8 --enable_cam_center --enable_cam_near_far --scale 0.2 --downscale 4 --stage 0 --lambda_entropy 1e-3 --clean_min_f 16 --clean_min_d 10 --visibility_mask_dilation 50 --sdf
CUDA_VISIBLE_DEVICES=4 python main.py data/bonsai/ --workspace trial_sdf_360_bonsai -O --data_format colmap --bound 8 --enable_cam_center --enable_cam_near_far --scale 0.2 --downscale 4 --stage 1 --iters 10000 --lambda_lap 1e-3 --lambda_normal 1e-3 --sdf

# CUDA_VISIBLE_DEVICES=4 python main.py data/kitchen/ --workspace trial_sdf_360_kitchen -O --data_format colmap --bound 8 --enable_cam_center --enable_cam_near_far --scale 0.2 --downscale 4 --stage 0 --lambda_entropy 1e-3 --clean_min_f 16 --clean_min_d 10 --visibility_mask_dilation 50 --sdf
# CUDA_VISIBLE_DEVICES=4 python main.py data/kitchen/ --workspace trial_sdf_360_kitchen -O --data_format colmap --bound 8 --enable_cam_center --enable_cam_near_far --scale 0.2 --downscale 4 --stage 1 --iters 10000 --lambda_lap 1e-3 --lambda_normal 1e-3 --sdf

# CUDA_VISIBLE_DEVICES=4 python main.py data/counter/ --workspace trial_sdf_360_counter -O --data_format colmap --bound 8 --enable_cam_center --enable_cam_near_far --scale 0.2 --downscale 4 --stage 0 --lambda_entropy 1e-3 --clean_min_f 16 --clean_min_d 10 --visibility_mask_dilation 50 --sdf
# CUDA_VISIBLE_DEVICES=4 python main.py data/counter/ --workspace trial_sdf_360_counter -O --data_format colmap --bound 8 --enable_cam_center --enable_cam_near_far --scale 0.2 --downscale 4 --stage 1 --iters 10000 --lambda_lap 1e-3 --lambda_normal 1e-3 --sdf