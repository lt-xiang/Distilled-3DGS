#!/bin/bash

mip360_path=datadir/db

datadir=${mip360_path}

scenes=(drjohnson playroom)

exp_folder=output_kd/db_30K

for scene in ${scenes[@]}; do
    exp_name="exp_${scene}"

    echo "Training ${scene}..."
    CUDA_VISIBLE_DEVICES=0 python train.py --source_path ${datadir}/${scene} \
                    --model_path ${exp_folder}/${exp_name} \
                    --eval \
                    --imp_metric indoor \
                    --firstPrune \
                    --iterations 30000 \
                    --port 6020 \
                    --start 0 \
                    --num_models 4 \
                    --factor 0.5          
    echo "Done training ${scene}"
    
    python render.py --source_path ${datadir}/${scene} \
                    --model_path ${exp_folder}/${exp_name} \
                    --skip_train \
                    --iteration 30000 \
                    --itrain 3
    python metrics.py --model_path ${exp_folder}/${exp_name}
done

