#!/bin/bash

mip360_path=datadir/mip360

datadir=${mip360_path}

scenes=(bicycle flowers garden treehill stump)

exp_folder=output_kd/mip360_0.5_0.25_30K

for scene in ${scenes[@]}; do
    exp_name="exp_${scene}"

    echo "Training ${scene}..."
    CUDA_VISIBLE_DEVICES=0 python train.py --source_path ${datadir}/${scene} \
                    --model_path ${exp_folder}/${exp_name} \
                    --resolution 4 \
                    --eval \
                    --firstPrune \
                    --imp_metric outdoor \
                    --iterations 30000 \
                    --port 6024 \
                    --start 0 \
                    --num_models 4             
    echo "Done training ${scene}"
    
    python render.py --source_path ${datadir}/${scene} \
                    --model_path ${exp_folder}/${exp_name} \
                    --skip_train \
                    --iteration 30000 \
                    --itrain 3

    python metrics.py --model_path ${exp_folder}/${exp_name}
done

