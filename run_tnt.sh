#!/bin/bash

mip360_path=datadir/tandt

datadir=${mip360_path}

scenes=(train truck)

exp_folder=output_kd/tnt_30K

for scene in ${scenes[@]}; do
    exp_name="exp_${scene}"

    echo "Training ${scene}..."
    CUDA_VISIBLE_DEVICES=0 python train.py --source_path ${datadir}/${scene} \
                    --model_path ${exp_folder}/${exp_name} \
                    --eval \
                    --imp_metric outdoor \
                    --firstPrune \
                    --iterations 30000 \
                    --port 6023 \
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

