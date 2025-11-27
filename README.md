<h2 align="center" width="100%">
Distilled-3DGS: Distilled 3D Gaussian Splatting
</h2>

<div>
<div align="center">
    <a href="https://distilled3dgs.github.io/" target="_blank">Lintao Xiang</a><sup>*1</sup>&emsp;
    <a href="https://distilled3dgs.github.io/" target="_blank">Xinkai Chen</a><sup>*2</sup>&emsp;
    <a href="https://cse.sysu.edu.cn/teacher/LaiJianhuang" target="_blank">Jianhuang Lai</a><sup>3</sup>&emsp;
    <a href="https://wanggcong.github.io/" target="_blank">Guangcong Wang</a><sup>2</sup><br>
</div>
<div>
<div align="center">
    <sup>1</sup>The University of Manchester&emsp;
    <sup>2</sup>Vision, Graphics, and X Group, Great Bay University<br>
    <sup>3</sup>Sun Yat-Sen University
</div>

<p align="center">
  <a href="https://arxiv.org/abs/2508.14037" target="_blank">
    <img alt="arXiv" src="https://img.shields.io/badge/cs.CV-arXiv%3A2508.14037-B31B1B.svg">
  </a>
  <a href="https://distilled3dgs.github.io/" target="_blank">
    <img alt="Project Page" src="https://img.shields.io/badge/Project%20Page-%F0%9F%93%9A-lightblue">
  </a>
</p>

<div style="text-align:center">
  <img src="assets/fig1.png" width="100%" height="100%" alt="Overview figure">
</div>

> <strong>TL;DR</strong>: <em>We present Distilled-3DGS, a simple yet effective knowledge distillation framework that achieves competitive performance in both rendering quality and storage efficiency.</em>

## ⚡ Updates
- [2025-08-19] Project resources released: <strong><a href="https://distilled3dgs.github.io/" target="_blank">Project Page</a></strong> | <strong><a href="https://arxiv.org/abs/2508.14037" target="_blank">arXiv</a></strong>

## ⭐ Abstract
3D Gaussian Splatting (3DGS) has exhibited remarkable efficacy in novel view synthesis (NVS). However, it suffers from a significant drawback: achieving high-fidelity rendering typically necessitates a large number of 3D Gaussians, resulting in substantial memory consumption and storage requirements. To address this challenge, we propose the first knowledge distillation framework for 3DGS, featuring various teacher models, including vanilla 3DGS, noise-augmented variants, and dropout-regularized versions. The outputs of these teachers are aggregated to guide the optimization of a lightweight student model. To distill the hidden geometric structure, we propose a structural similarity loss to boost the consistency of spatial geometric distributions between the student and teacher model. Through comprehensive quantitative and qualitative evaluations across diverse datasets, the proposed Distilled-3DGS—a simple yet effective framework without bells and whistles—achieves promising results in both rendering quality and storage efficiency compared to state-of-the-art methods.

## ⭐ Pipeline
<div style="text-align:center">
  <img src="assets/fig2.png" width="90%" height="100%" alt="Pipeline figure">
</div>
Our method consists of two stages. First, a standard teacher model G_std is trained, along with two variants: G_perb with random perturbation and G_drop with random dropout. Then, a pruned student model is supervised by the outputs of these teachers. Additionally, a spatial distribution distillation strategy is introduced to help the student learn structural patterns from the teachers.

## 1. Setup
This code has been tested with Python 3.10, PyTorch 2.0.1, CUDA 11.8.

- Clone the repository

```bash
git clone https://github.com/lt-xiang/Distilled-3DGS.git
cd Distilled-3DGS
```

- Setup Python environment

```bash
# Create and activate conda environment
conda create -n DistilledGS python=3.10 -y
conda activate DistilledGS

# Install PyTorch (CUDA 11.8)
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

# Install project dependencies
pip install -r requirements.txt
```

## 2. Dataset

1) Download datasets:
- Mip-NeRF 360: https://jonbarron.info/mipnerf360/
- T&T + DB (COLMAP inputs): https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip

2) Generate depth maps

a. Clone Depth Anything v2:

```bash
git clone https://github.com/DepthAnything/Depth-Anything-V2.git
```

b. Download weights from Depth-Anything-V2-Large and place under `Depth-Anything-V2/checkpoints/`:
- https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true

c. Generate depth maps:

```bash
python Depth-Anything-V2/run.py \
  --encoder vitl \
  --pred-only \
  --grayscale \
  --img-path <path_to_input_images> \
  --outdir <output_path>
```

d. Generate `depth_params.json`:

```bash
python utils/make_depth_scale.py \
  --base_dir <path_to_colmap> \
  --depths_dir <path_to_generated_depths>
```

## 3. Model

Run the following scripts for training and evaluation:

```bash
# Mip-NeRF 360 (indoor)
bash run_mip360_in.sh

# Mip-NeRF 360 (outdoor)
bash run_mip360_out.sh

# Tanks & Temples
bash run_tnt.sh

# DTU/BlendedMVS (DB)
bash run_db.sh
```

Using the Mip-NeRF 360 bonsai scene as an example, we first train three different teacher models. Then, outputs from these teachers are used to supervise the student model.

```bash
# 1) Train (teachers + student)
python train.py \
  --source_path <dataset_path> \
  --model_path <model_path> \
  --resolution 2 \
  --eval \
  --imp_metric indoor \
  --iterations 30000 \
  --num_models 4  # 0,1,2 are teacher models; 3 is the student model.

# 2) Render with the student model
python render.py \
  --source_path <dataset_path> \
  --model_path <model_path> \
  --skip_train \
  --iteration 30000 \
  --itrain 3  # student model index

# 3) Evaluation
python metrics.py --model_path <model_path>
```

## Acknowledgement

Special thanks to the following awesome projects!
- Gaussian Splatting: https://github.com/graphdeco-inria/gaussian-splatting
- Mini Splatting: https://github.com/fatPeter/mini-splatting
- Dropout Gaussian: https://github.com/DCVL-3D/DropGaussian_release

## Citation

```bibtex
@article{Xiang2025Distilled3DGaussianSplatting,
  title   = {Distilled-3DGS: Distilled 3D Gaussian Splatting},
  author  = {Lintao Xiang and Xinkai Chen and Jianhuang Lai and Guangcong Wang},
  journal = {arXiv},
  year    = {2025},
  eprint  = {2508.14037},
  archivePrefix = {arXiv},
  primaryClass   = {cs.CV}
}
```