#!/bin/sh

for scan in 24 ; # 24 37 40 55 63 65 69 83 97 105 106 110 114 118 122 ;
do
    CUDA_VISIBLE_DEVICES=5 python run.py \
        --project_path out/pixelnerf/CASE_NAME \
        --dict_path /mnt/disk1/wyl/projects/udf-prior-recon/modified/NeuS/public_data/DTU_pixelnerf/dtu_scan${scan}/cameras_sphere.npz \
        --image_path /mnt/disk1/wyl/projects/udf-prior-recon/modified/NeuS/public_data/DTU_pixelnerf/dtu_scan${scan}/image \
        --case dtu_scan${scan} \
        --n_images 3 \
        --width 1600 \
        --height 1200 \
        --scaled ;
done
