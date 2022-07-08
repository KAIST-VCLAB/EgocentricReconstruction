import numpy as np
import torch
from RAFT.core.raft import RAFT
import cv2
import cupy as cp
import argparse
import os
from utils import video_util
from utils import cuda_util
from scipy.spatial.transform import Rotation as R
from utils.depth_util import get_proper_baseline


def get_test_neighbor_list(frame_num, test_view_num, test_view_range, video_length):
    return_list = []
    for i in range(test_view_range):
        ## from closer frames to farther frames
        j = (i // 2) * ((-1) ** i)

        second_idx = frame_num + j
        if (second_idx < 0) or (second_idx >= video_length):
            continue

        if frame_num == second_idx:
            continue
        return_list.append(second_idx)

        if len(return_list) == test_view_num:
            break
    return return_list

def save_weights(color_threshold, video_frames, video_frame_rt, depth_maps, video_length, args, weight_path):
    device = "cuda:0"
    device_id = [0]

    # load cuda functions
    source = cuda_util.read_cuda_file("cuda/cuda_functions.cu")
    cuda_source = '''{}'''.format(source)
    module = cp.RawModule(code=cuda_source)
    rectify = module.get_function('rectifyLL')
    rectify_depth = module.get_function('rectifyLLdepth')
    warp2ref_depth = module.get_function('warpToRefDepth')

    h, w = depth_maps[0].shape
    block_size_single = (32, 32)
    ll_grid_size_single = (h // block_size_single[1], w // block_size_single[0])
    eq_grid_size_single = (w // block_size_single[1], h // block_size_single[0])

    model = torch.nn.DataParallel(RAFT(args), device_ids=device_id)
    model.load_state_dict(torch.load(args.base_pth, map_location="cuda:0"))
    model.to(device)
    model.module.freeze_bn()
    model.eval()

    baseline_list = get_proper_baseline(video_frames, video_frame_rt, model, rectify)
    frame_idx_list = video_util.get_frame_idx_list(video_frame_rt)

    reference_LL_cupy = cp.zeros((w, h, 3), dtype=cp.uint8)
    reference_LL_cupy_depth = cp.zeros((w, h), dtype=cp.float32)

    sined_ref_depth = cp.zeros((h, w), dtype=cp.float32)
    sined_neigh_depth = cp.zeros((h, w), dtype=cp.float32)
    neighbor_LL_cupy = cp.zeros((w, h, 3), dtype=cp.uint8)
    neighbor_LL_cupy_depth = cp.zeros((w, h), dtype=cp.float32)


    for first_idx in range(video_length):
        if os.path.isfile(f"{weight_path}/{first_idx}.exr"):
            print(f"{weight_path}/{first_idx}.exr exists")
            continue

        reference_LL_cupy = reference_LL_cupy * 0
        reference_LL_cupy_depth = reference_LL_cupy_depth * 0

        ## LL  w is long
        u_for_depth = np.linspace(0, w - 1, w)
        v_for_depth = np.linspace(0, h - 1, h)
        vv_for_depth, uu_for_depth = np.meshgrid(v_for_depth, u_for_depth)
        phi_for_depth = np.pi - np.pi * (vv_for_depth + 0.5) / h

        depth_map = 1/cp.asnumpy(depth_maps[first_idx])
        color_ref = cp.asnumpy(video_frames[first_idx])
        rt_ref = video_frame_rt[first_idx][1]

        u = np.linspace(0, h-1, h)
        v = np.linspace(0, w-1, w)
        vv, uu = np.meshgrid(v, u)
        theta = np.pi * (((h-1)-uu) + 0.5) / h
        phi = 2 * np.pi * (vv + 0.5) / w - 3 * np.pi / 2

        x = depth_map * np.sin(theta) * np.cos(phi)
        y = depth_map * np.cos(theta)
        z = -depth_map * np.sin(theta) * np.sin(phi)

        x = x.reshape(1, h*w)
        y = y.reshape(1, h*w)
        z = z.reshape(1, h*w)
        ones = np.ones_like(x)
        xyz1 = np.vstack([x,y,z,ones])

        test_neighbor_list = video_util.get_second_neighbors(first_idx, frame_idx_list, args.second_neighbor_interval, len(frame_idx_list), args.second_neighbor_size, video_frame_rt, baseline_list[0], baseline_list[-1])

        weight_sum = np.zeros_like(depth_map).astype(np.float32)
        print(f"first_idx:{first_idx}, second_idx_list:{test_neighbor_list}")

        for second_idx in test_neighbor_list:

            depth_map_neigh = 1/cp.asnumpy(depth_maps[second_idx])
            color_neigh = cp.asnumpy(video_frames[second_idx])
            rt_neigh = video_frame_rt[second_idx][1]

            rt = np.matmul(rt_neigh, np.linalg.inv(rt_ref))
            xyz1_neigh = np.matmul(rt, xyz1)
            xyz_neigh = xyz1_neigh[0:3,:]
            xyz_neigh_norm = xyz_neigh / np.linalg.norm(xyz_neigh, axis=0)
            x_neigh = xyz_neigh_norm[0,:].reshape(h,w)
            y_neigh = xyz_neigh_norm[1,:].reshape(h,w)
            z_neigh = xyz_neigh_norm[2,:].reshape(h,w)
            phi_neigh = 3 * np.pi / 2 - np.arctan2(z_neigh, x_neigh)
            theta_neigh = np.arccos(-y_neigh)
            u_neigh = theta_neigh * h / np.pi - 0.5
            v_neigh = phi_neigh * w / (2 * np.pi) - 0.5
            u_neigh[u_neigh < 0] = u_neigh[u_neigh < 0] + h
            u_neigh[u_neigh >= h] = u_neigh[u_neigh >= h] - h
            v_neigh[v_neigh < 0] = v_neigh[v_neigh < 0] + w
            v_neigh[v_neigh >= w] = v_neigh[v_neigh >= w] - w

            sined_ref_depth = sined_ref_depth * 0
            sined_neigh_depth = sined_neigh_depth * 0
            neighbor_LL_cupy = neighbor_LL_cupy * 0
            neighbor_LL_cupy_depth = neighbor_LL_cupy_depth * 0

            rt = np.matmul(rt_ref, np.linalg.inv(rt_neigh))
            v = np.array([1, 0, 0])
            t = rt[0:3, -1]
            r = rt[0:3, 0:3]
            baseline = cp.asarray(np.linalg.norm(t), dtype=cp.float32)
            angle = np.arccos(np.dot(t, v) / np.linalg.norm(t))
            axis = np.cross(v, t) / np.linalg.norm(np.cross(v, t))
            rect_r = R.from_rotvec(angle * axis)
            rect_r_cupy = cp.array(rect_r.as_matrix(), dtype=cp.float32)
            warp_r_cupy = cp.array(np.matmul(np.linalg.inv(r), rect_r.as_matrix()), dtype=cp.float32)
            rect_r_inv_cupy = cp.array(np.linalg.inv(rect_r.as_matrix()), dtype=cp.float32)
            warp_r_inv_cupy = cp.array(np.linalg.inv(np.matmul(np.linalg.inv(r), rect_r.as_matrix())), dtype=cp.float32)

            rectify(ll_grid_size_single, block_size_single,
                    (video_frames[first_idx], video_frames[second_idx], rect_r_cupy, warp_r_cupy, reference_LL_cupy, neighbor_LL_cupy, h, w))
            rectify_depth(ll_grid_size_single, block_size_single,
                          (depth_maps[first_idx], depth_maps[second_idx], rect_r_cupy, warp_r_cupy, reference_LL_cupy_depth, neighbor_LL_cupy_depth, h, w))

            reference_LL_depth = cp.asnumpy(reference_LL_cupy_depth)
            neighbor_LL_depth = cp.asnumpy(neighbor_LL_cupy_depth)

            reference_LL_depth[reference_LL_depth < 0.0000001] = 0.0000001
            neighbor_LL_depth[neighbor_LL_depth < 0.0000001] = 0.0000001
            reference_LL_depth = 1/reference_LL_depth * np.sin(phi_for_depth)
            neighbor_LL_depth = 1/neighbor_LL_depth * np.sin(phi_for_depth)

            warp2ref_depth(eq_grid_size_single, block_size_single, (cp.asarray(reference_LL_depth,dtype=cp.float32), sined_ref_depth, rect_r_inv_cupy, w, h))
            warp2ref_depth(eq_grid_size_single, block_size_single, (cp.asarray(neighbor_LL_depth,dtype=cp.float32), sined_neigh_depth, warp_r_inv_cupy, w, h))

            sined_ref_depth_numpy = cp.asnumpy(sined_ref_depth)
            sined_neigh_depth_numpy = cp.asnumpy(sined_neigh_depth)
            warped_sined_neigh_depth_numpy = cv2.remap(sined_neigh_depth_numpy, v_neigh.astype('float32'), u_neigh.astype('float32'), interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)
            warped_img = cv2.remap(cp.asnumpy(color_neigh), v_neigh.astype('float32'), u_neigh.astype('float32'),interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            depth_diff = np.abs((sined_ref_depth_numpy / warped_sined_neigh_depth_numpy) - 1)

            color_diff = np.mean(np.square(color_ref - warped_img),axis=2)

            ref_depth_mask = sined_ref_depth_numpy < 0
            ref_depth_mask2 = depth_map < 0
            neigh_depth_mask = warped_sined_neigh_depth_numpy < 0
            neigh_depth_mask2 = depth_map_neigh < 0


            depth_control = np.median(depth_map[~ref_depth_mask2]) / 1000
            depth_weight = np.exp(-depth_diff * depth_diff / depth_control)
            color_control = color_threshold
            color_weight = np.exp(-color_diff * color_diff / color_control)
            depth_diff[ref_depth_mask | neigh_depth_mask | ref_depth_mask2 | neigh_depth_mask2] = 0
            color_diff[ref_depth_mask | neigh_depth_mask | ref_depth_mask2 | neigh_depth_mask2] = 0
            color_weight[ref_depth_mask | neigh_depth_mask | ref_depth_mask2 | neigh_depth_mask2] = 0
            depth_weight[ref_depth_mask | neigh_depth_mask | ref_depth_mask2 | neigh_depth_mask2] = 0

            weight_sum = weight_sum + depth_weight * color_weight

        cv2.imwrite(f"{weight_path}/{first_idx}.exr", (weight_sum / args.second_neighbor_size).astype("float32"))

    return


def run_weight(config):
    c_args = []
    for k, v in config['data_path'].items():
        c_args += [f'--{k}', str(v)]
    for k, v in config['depth_estimation'].items():
        c_args += [f'--{k}', str(v)]

    parser = argparse.ArgumentParser()

    ## RAFT arguments
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', default=True, help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    parser.add_argument('--name', default='epi_raft')
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--inference', action='store_true', default=True)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')
    parser.add_argument('--lr', type=float, default=pow(10, -5))
    parser.add_argument('--alpha', type=float, default=1.0, help='exponential weighting')
    parser.add_argument('--beta', type=float, default=0.8, help='exponential weighting')
    parser.add_argument('--eta', type=float, default=0.8, help='exponential weighting')
    parser.add_argument('--color_loss_type', type=str, default="l1")
    parser.add_argument('--coordinate_loss_type', type=str, default="l1")
    parser.add_argument('--disparity_loss_type', type=str, default="l1")

    ## Paths
    parser.add_argument('--data_root_path', type=str, default="")
    parser.add_argument('--video_root_path', type=str, default="")
    parser.add_argument('--depth_root_path', type=str, default="")
    parser.add_argument('--mesh_root_path', type=str, default="")
    parser.add_argument('--rendering_root_path', type=str, default="")
    parser.add_argument('--video_name', type=str, default="")

    ## Options
    parser.add_argument('--sample_ratio', type=int, default=1, help='sample ratio')
    parser.add_argument('--second_neighbor_size', type=int, default=11, help='second_neighbor_size')
    parser.add_argument('--second_neighbor_interval', type=int, default=1, help='second_neighbor_interval ratio')
    parser.add_argument('--mask_radius', type=float, default=0.15, help='mask_radius')
    parser.add_argument('--depth_scale', type=float, default=2 / 5, help='rescale depth map resolution')
    parser.add_argument('--base_pth', help="restore checkpoint", default="./checkpoints/best_000999_epi_raft.pth")

    args = parser.parse_args(args=c_args)


    depth_path = f"{args.data_root_path}/{args.depth_root_path}/{args.video_name}_depth"
    weight_path = f"{args.data_root_path}/{args.depth_root_path}/{args.video_name}_weight"
    video_path = args.data_root_path + "/" + args.video_root_path + "/" + args.video_name

    os.makedirs(weight_path, exist_ok=True)

    video_frame_rt = video_util.read_csv(video_path + "/traj.csv")
    video = cv2.VideoCapture(video_path + "/video.mp4")

    video_frames = video_util.get_all_frames(video, args.depth_scale, args.sample_ratio)
    video_frame_rt = video_util.sample_rt(video_frame_rt, args.sample_ratio)
    depth_maps = video_util.get_all_depths(depth_path, len(video_frame_rt))

    color_threshold = 24
    save_weights(color_threshold, video_frames, video_frame_rt, depth_maps, len(video_frame_rt), args, weight_path)
