CUDA_VISIBLE_DEVICES=7 python main.py data/nerf_synthetic/lego/ --workspace trial_syn_sdf_lego/ -O --bound 1 --scale 0.8 --dt_gamma 0 --stage 0 --sdf
CUDA_VISIBLE_DEVICES=7 python main.py data/nerf_synthetic/lego/ --workspace trial_syn_sdf_lego/ -O --bound 1 --scale 0.8 --dt_gamma 0 --stage 1

CUDA_VISIBLE_DEVICES=7 python main.py data/nerf_synthetic/mic/ --workspace trial_syn_sdf_mic/ -O --bound 1 --scale 0.8 --dt_gamma 0 --stage 0 --sdf
CUDA_VISIBLE_DEVICES=7 python main.py data/nerf_synthetic/mic/ --workspace trial_syn_sdf_mic/ -O --bound 1 --scale 0.8 --dt_gamma 0 --stage 1

CUDA_VISIBLE_DEVICES=7 python main.py data/nerf_synthetic/materials/ --workspace trial_syn_sdf_materials/ -O --bound 1 --scale 0.8 --dt_gamma 0 --stage 0 --sdf
CUDA_VISIBLE_DEVICES=7 python main.py data/nerf_synthetic/materials/ --workspace trial_syn_sdf_materials/ -O --bound 1 --scale 0.8 --dt_gamma 0 --stage 1

CUDA_VISIBLE_DEVICES=7 python main.py data/nerf_synthetic/chair/ --workspace trial_syn_sdf_chair/ -O --bound 1 --scale 0.8 --dt_gamma 0 --stage 0 --sdf
CUDA_VISIBLE_DEVICES=7 python main.py data/nerf_synthetic/chair/ --workspace trial_syn_sdf_chair/ -O --bound 1 --scale 0.8 --dt_gamma 0 --stage 1

CUDA_VISIBLE_DEVICES=7 python main.py data/nerf_synthetic/hotdog/ --workspace trial_syn_sdf_hotdog/ -O --bound 1 --scale 0.7 --dt_gamma 0 --stage 0 --sdf
CUDA_VISIBLE_DEVICES=7 python main.py data/nerf_synthetic/hotdog/ --workspace trial_syn_sdf_hotdog/ -O --bound 1 --scale 0.7 --dt_gamma 0 --stage 1

CUDA_VISIBLE_DEVICES=7 python main.py data/nerf_synthetic/ficus/ --workspace trial_syn_sdf_ficus/ -O --bound 1 --scale 0.8 --dt_gamma 0 --stage 0 --sdf
CUDA_VISIBLE_DEVICES=7 python main.py data/nerf_synthetic/ficus/ --workspace trial_syn_sdf_ficus/ -O --bound 1 --scale 0.8 --dt_gamma 0 --stage 1

CUDA_VISIBLE_DEVICES=7 python main.py data/nerf_synthetic/drums/ --workspace trial_syn_sdf_drums/ -O --bound 1 --scale 0.8 --dt_gamma 0 --stage 0 --sdf
CUDA_VISIBLE_DEVICES=7 python main.py data/nerf_synthetic/drums/ --workspace trial_syn_sdf_drums/ -O --bound 1 --scale 0.8 --dt_gamma 0 --stage 1

CUDA_VISIBLE_DEVICES=7 python main.py data/nerf_synthetic/ship/ --workspace trial_syn_sdf_ship/ -O --bound 1 --scale 0.7 --dt_gamma 0 --stage 0 --sdf
CUDA_VISIBLE_DEVICES=7 python main.py data/nerf_synthetic/ship/ --workspace trial_syn_sdf_ship/ -O --bound 1 --scale 0.7 --dt_gamma 0 --stage 1