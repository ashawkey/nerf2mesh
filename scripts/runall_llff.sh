CUDA_VISIBLE_DEVICES=6 python main.py data/nerf_llff_data/fern --workspace trial_llff_fern -O --data_format colmap --bound 4 --downscale 4 --stage 0 --visibility_mask_dilation 50
CUDA_VISIBLE_DEVICES=6 python main.py data/nerf_llff_data/fern --workspace trial_llff_fern -O --data_format colmap --bound 4 --downscale 4 --stage 1 --iters 10000

CUDA_VISIBLE_DEVICES=6 python main.py data/nerf_llff_data/flower --workspace trial_llff_flower -O --data_format colmap --bound 4 --downscale 4 --stage 0 --visibility_mask_dilation 50
CUDA_VISIBLE_DEVICES=6 python main.py data/nerf_llff_data/flower --workspace trial_llff_flower -O --data_format colmap --bound 4 --downscale 4 --stage 1 --iters 10000

CUDA_VISIBLE_DEVICES=6 python main.py data/nerf_llff_data/fortress --workspace trial_llff_fortress -O --data_format colmap --bound 4 --downscale 4 --stage 0 --visibility_mask_dilation 50
CUDA_VISIBLE_DEVICES=6 python main.py data/nerf_llff_data/fortress --workspace trial_llff_fortress -O --data_format colmap --bound 4 --downscale 4 --stage 1 --iters 10000

CUDA_VISIBLE_DEVICES=6 python main.py data/nerf_llff_data/horns --workspace trial_llff_horns -O --data_format colmap --bound 4 --downscale 4 --stage 0 --visibility_mask_dilation 50
CUDA_VISIBLE_DEVICES=6 python main.py data/nerf_llff_data/horns --workspace trial_llff_horns -O --data_format colmap --bound 4 --downscale 4 --stage 1 --iters 10000

CUDA_VISIBLE_DEVICES=6 python main.py data/nerf_llff_data/leaves --workspace trial_llff_leaves -O --data_format colmap --bound 4 --downscale 4 --stage 0 --visibility_mask_dilation 50
CUDA_VISIBLE_DEVICES=6 python main.py data/nerf_llff_data/leaves --workspace trial_llff_leaves -O --data_format colmap --bound 4 --downscale 4 --stage 1 --iters 10000

CUDA_VISIBLE_DEVICES=6 python main.py data/nerf_llff_data/orchids --workspace trial_llff_orchids -O --data_format colmap --bound 4 --downscale 4 --stage 0 --visibility_mask_dilation 50
CUDA_VISIBLE_DEVICES=6 python main.py data/nerf_llff_data/orchids --workspace trial_llff_orchids -O --data_format colmap --bound 4 --downscale 4 --stage 1 --iters 10000

CUDA_VISIBLE_DEVICES=6 python main.py data/nerf_llff_data/room --workspace trial_llff_room -O --data_format colmap --bound 1 --downscale 4 --stage 0 --visibility_mask_dilation 50
CUDA_VISIBLE_DEVICES=6 python main.py data/nerf_llff_data/room --workspace trial_llff_room -O --data_format colmap --bound 1 --downscale 4 --stage 1 --iters 10000

CUDA_VISIBLE_DEVICES=6 python main.py data/nerf_llff_data/trex --workspace trial_llff_trex -O --data_format colmap --bound 1 --downscale 4 --stage 0 --visibility_mask_dilation 50
CUDA_VISIBLE_DEVICES=6 python main.py data/nerf_llff_data/trex --workspace trial_llff_trex -O --data_format colmap --bound 1 --downscale 4 --stage 1 --iters 10000 
