
#!/bin/bash

scenes=(bonsai counter kitchen room)

python ./Depth-Anything-V2/run.py \
  --encoder vitl \
  --pred-only \
  --grayscale \
  --img_path ./datadir/tandt/${scene}/images \
  --outdir ./datadir/tandt/${scene}/depths

python utils/make_depth_scale.py \
  --base_dir ./datadir/tandt/${scene} \
  --depths_dir ./datadir/tandt/${scene}/depths