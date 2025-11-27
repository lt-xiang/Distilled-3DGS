#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
import numpy as np
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui, render_imp,render_depth
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
from utils.sh_utils import SH2RGB, RGB2SH
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

import torch

def voxel_histogram_soft(points, xyz_min, xyz_max, grid_size=128, sigma=0.5):
    device = points.device
    voxel_size = (xyz_max - xyz_min) / grid_size

    indices_float = (points - xyz_min) / voxel_size  # (N, 3)

    grid_1d = torch.arange(grid_size, device=device).float()

    dx = indices_float[:, 0:1] - grid_1d[None, :]   # (N, G)
    dy = indices_float[:, 1:2] - grid_1d[None, :]
    dz = indices_float[:, 2:2+1] - grid_1d[None, :]

    wx = torch.exp(- dx.pow(2) / (2 * sigma**2))    # (N, G)
    wy = torch.exp(- dy.pow(2) / (2 * sigma**2))
    wz = torch.exp(- dz.pow(2) / (2 * sigma**2))

    hist_x = wx.sum(dim=0)  # (G,)
    hist_y = wy.sum(dim=0)
    hist_z = wz.sum(dim=0)

    hist = (hist_x[:, None, None] * hist_y[None, :, None] * hist_z[None, None, :])
    hist = hist.reshape(-1)

    hist = hist / (hist.sum() + 1e-8)

    return hist

def voxel_histogram(points, xyz_min, xyz_max, grid_size=128):

    voxel_size = (xyz_max - xyz_min) / grid_size
    indices = ((points - xyz_min) / voxel_size).long().clamp(0, grid_size-1)
    idx_flat = indices[:,0] * grid_size**2 + indices[:,1] * grid_size + indices[:,2]
    hist = torch.bincount(idx_flat, minlength=grid_size**3).float()
    hist = hist / (hist.sum() + 1e-8) 
    return hist

def paired_voxel_histogram(P1, P2, grid_size=128, margin=0.01):

    all_points = torch.cat([P1, P2], dim=0)
    xyz_min = all_points.min(0)[0] - margin
    xyz_max = all_points.max(0)[0] + margin
    
    hist1 = voxel_histogram_soft(P1, xyz_min, xyz_max, grid_size)
    hist2 = voxel_histogram_soft(P2, xyz_min, xyz_max, grid_size)
    
    return hist1, hist2

def structure_voxel_loss(P1, P2, grid_size=256, margin=0.01):
    
    hist1, hist2 = paired_voxel_histogram(P1, P2, grid_size=grid_size, margin=margin)

    return 1 - torch.nn.functional.cosine_similarity(hist1, hist2, dim=0)


def init_cdf_mask(importance, thres=1.0):
    importance = importance.flatten()   
    if thres!=1.0:
        percent_sum = thres
        vals,idx = torch.sort(importance+(1e-6))
        cumsum_val = torch.cumsum(vals, dim=0)
        split_index = ((cumsum_val/vals.sum()) > (1-percent_sum)).nonzero().min()
        split_val_nonprune = vals[split_index]

        non_prune_mask = importance>split_val_nonprune 
    else: 
        non_prune_mask = torch.ones_like(importance).bool()
        
    return non_prune_mask

def depth_to_invdepth_torch(depth_map, epsilon=1e-6, normalize=True):
    
    inv_depth = 1.0 / (depth_map + epsilon)

    if normalize:
        min_val = inv_depth.amin(dim=(-2, -1), keepdim=True)
        max_val = inv_depth.amax(dim=(-2, -1), keepdim=True)
        inv_depth = (inv_depth - min_val) / (max_val - min_val + epsilon)

    return inv_depth


def training(dataset, opt, pipe, 
             testing_iterations, saving_iterations, 
             checkpoint_iterations, checkpoint, debug_from,
             itrain, factor):

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    
    #Teacher-Student model
    GsDict={}
    if itrain == 3:
        for i in range(itrain):
            print(f"Loading {i}th teacher model")
            GsDict[f"gs_{i}"] = GaussianModel(dataset.sh_degree, opt.optimizer_type)
            GsDict[f"gs_{i}"].load_ply(os.path.join(args.model_path,
                                                    "point_cloud_" + str(i),
                                                    "iteration_" + str(30_000),
                                                    "point_cloud.ply"), args.train_test_exp) 
            
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians, itrain = itrain)
    
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    viewpoint_stack = scene.getTrainCameras().copy()

    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    mask_blur = torch.zeros(gaussians._xyz.shape[0], device='cuda')
    
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        loss = 0
        # Pick a random Camera
        for _ in range(6):
            if not viewpoint_stack:
                viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
            
            # Render
            if (iteration - 1) == debug_from:
                pipe.debug = True

            bg = torch.rand((3), device="cuda") if opt.random_background else background
            
            image_tea = None                        
            if itrain == 3:
                image_t=[]
                for i in range(itrain):
                    render_pkg = render(viewpoint_cam, GsDict[f"gs_{i}"], pipe, bg)
                    image_t.append(render_pkg["render"].detach())
                stacked = torch.stack(image_t, dim=0) 
                image_tea = torch.mean(stacked, dim=0) 
                
                render_pkg = render_imp(viewpoint_cam, gaussians, pipe, bg)
                render_depth_pkg = render_depth(viewpoint_cam, gaussians, pipe, bg)
            else:
                render_pkg = render(viewpoint_cam, gaussians, pipe, bg, itrain=itrain, iteration=iteration, separate_sh=SPARSE_ADAM_AVAILABLE)
            
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

            if viewpoint_cam.alpha_mask is not None:
                alpha_mask = viewpoint_cam.alpha_mask.cuda()
                image *= alpha_mask

            # GT Loss
            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(image, gt_image)
            if FUSED_SSIM_AVAILABLE:
                ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
            else:
                ssim_value = ssim(image, gt_image)

            loss1 = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
            loss += 0.6 * loss1

        if itrain == 3:            
            if FUSED_SSIM_AVAILABLE:
                ssim_value = fused_ssim(image_tea.unsqueeze(0), image.unsqueeze(0))
            else:
                ssim_value = ssim(image_tea, image)
            
            distill_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
                
            loss += 0.4 * distill_loss
                
            if iteration > 20000 and iteration % 1000 == 0 and iteration < 30_000:          
                is_struc_loss=True
                if is_struc_loss: 
                    loss_struc = structure_voxel_loss(gaussians.get_xyz, GsDict[f"gs_2"].get_xyz.detach())
                    loss += loss_struc
                    print(f"iteration: {iteration}, distill struc_loss: {loss_struc}")

        # Depth regularization
        Ll1depth_pure = 0.0
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            
            if itrain == 3:
                invDepth = depth_to_invdepth_torch(render_depth_pkg["rendered_depth"])
            else:
                invDepth = render_pkg["depth"]
                
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()

            Ll1depth_pure = torch.abs((invDepth - mono_invdepth) * depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
            loss += Ll1depth
            Ll1depth = Ll1depth.item()
        else:
            Ll1depth = 0

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, 
                            (pipe, background, 
                             itrain, iteration,
                             1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), 
                            dataset.train_test_exp)
            
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration, itrain)

            #Perturbations model
            if itrain == 0:
                if iteration < opt.noise_from_end:
                    if iteration > opt.noise_from_iter and iteration % opt.noise_interval == 0:  
                        gaussians.add_noises(opt.densify_grad_threshold, scene.cameras_extent)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if itrain == 3:
                    area_max = render_pkg["area_max"]
                    mask_blur = torch.logical_or(mask_blur, area_max > (image.shape[1] * image.shape[2] / 5000))
                    
                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0 and gaussians._xyz.shape[0]<args.num_max:  
                    
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        gaussians.densify_and_prune_split(opt.densify_grad_threshold, 
                                                        0.005, scene.cameras_extent, 
                                                        size_threshold, mask_blur)
                        mask_blur = torch.zeros(gaussians._xyz.shape[0], device='cuda')
                else:
                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)
                    
                    if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                        gaussians.reset_opacity()
            
            #remove Gaussians
            if itrain == 3 and iteration == 15_000 and args.firstPrune:    
                print(f"firstPrune: {args.firstPrune}")     
                imp_score = torch.zeros(gaussians._xyz.shape[0]).cuda()
                accum_area_max = torch.zeros(gaussians._xyz.shape[0]).cuda()
                views = scene.getTrainCameras()
                for view in views:
                    # print(idx)
                    render_pkg = render_imp(view, gaussians, pipe, background)
                    accum_weights = render_pkg["accum_weights"]
                    area_proj = render_pkg["area_proj"]
                    area_max = render_pkg["area_max"]

                    accum_area_max = accum_area_max + area_max

                    if args.imp_metric == 'outdoor':
                        mask_t = area_max!= 0
                        temp = imp_score + accum_weights / area_proj
                        imp_score[mask_t] = temp[mask_t]
                    else:
                        imp_score = imp_score + accum_weights

                imp_score[accum_area_max==0]=0
                prob = imp_score / imp_score.sum()
                prob = prob.detach().cpu().numpy()

                N_xyz=gaussians._xyz.shape[0]
                num_sampled=int(N_xyz * factor * ((prob!=0).sum() / prob.shape[0]))
                indices = np.random.choice(N_xyz, size=num_sampled, 
                                            p=prob, replace=False)

                prune_mask = np.zeros(N_xyz, dtype=bool)
                prune_mask[indices] = True
                
                gaussians.prune_points(prune_mask == False)
                
                gaussians.max_sh_degree=dataset.sh_degree
                
                gaussians.reinitial_pts(gaussians._xyz, 
                                        SH2RGB(gaussians._features_dc+0)[:, 0])
                
                gaussians.training_setup(opt)
                
                torch.cuda.empty_cache()   
                
                viewpoint_stack = scene.getTrainCameras().copy()

            if itrain == 3 and iteration == 20_000:
                imp_score = torch.zeros(gaussians._xyz.shape[0]).cuda()
                accum_area_max = torch.zeros(gaussians._xyz.shape[0]).cuda()
                views=scene.getTrainCameras()
                for view in views:
                    gt = view.original_image[0:3, :, :]
                    
                    render_pkg = render_imp(view, gaussians, pipe, background)
                    accum_weights = render_pkg["accum_weights"]
                    area_proj = render_pkg["area_proj"]
                    area_max = render_pkg["area_max"]

                    accum_area_max = accum_area_max+area_max

                    if args.imp_metric=='outdoor':
                        mask_t=area_max!=0
                        temp=imp_score+accum_weights/area_proj
                        imp_score[mask_t] = temp[mask_t]
                    else:
                        imp_score=imp_score+accum_weights
                    
                imp_score[accum_area_max==0]=0
                non_prune_mask = init_cdf_mask(importance=imp_score, thres=0.99) 
                                
                gaussians.prune_points(non_prune_mask==False)
                gaussians.training_setup(opt)
                
                torch.cuda.empty_cache()   
                
            # Optimizer step
            if iteration < opt.iterations:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[15_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    
    parser.add_argument("--imp_metric", required=True, type=str, default = None)
    parser.add_argument("--start", type=int, default = 0)  
    parser.add_argument("--num_models", type=int, default = 4)
    parser.add_argument("--factor", type=float, default = 0.9)
    parser.add_argument("--num_depth", type=int, default = 3_500_000)
    parser.add_argument("--num_max", type=int, default = 4_500_000)
    parser.add_argument('--firstPrune', action='store_true', default=False)

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    for i in range(args.start, args.num_models):
        print(f"train {i}_th model")
        training(lp.extract(args), op.extract(args), pp.extract(args), 
                 args.test_iterations, args.save_iterations, args.checkpoint_iterations, 
                 args.start_checkpoint, args.debug_from, 
                 itrain = i, factor = args.factor)
        print(f"end {i}_th model")

    # All done
    print("\nAll model training complete.")
