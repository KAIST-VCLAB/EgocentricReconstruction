import torch
from RAFT.core.raft import RAFT
import argparse
import cupy as cp
import numpy as np
import os
from scipy.spatial.transform import Rotation as R
from utils import video_util
import cv2
from utils import cuda_util

def get_proper_baseline(video_frames, video_frame_rt, model, rectify):

    h, w, c = video_frames[0].shape
    block_size_single = (32, 32)

    ll_grid_size_single = (h // block_size_single[1], w // block_size_single[0])

    pad_reference_LL_cupy = cp.zeros((w * 5 // 4, h, 3), dtype=cp.uint8)
    pad_neighbor_LL_cupy = cp.zeros((w * 5 // 4, h, 3), dtype=cp.uint8)

    reference_LL_cupy = cp.zeros((w, h, 3), dtype=cp.uint8)
    neighbor_LL_cupy = cp.zeros((w, h, 3), dtype=cp.uint8)


    # find proper baseline
    ref_rt = video_frame_rt[0][1]
    second_idx_list = range(1, len(video_frame_rt))
    baseline_list = []

    ## run stereo
    print("Finding baseline condition...")
    for second_idx_index in range(len(second_idx_list) // 5):

        second_idx = second_idx_list[second_idx_index * 5]

        warp_rt = video_frame_rt[second_idx][1]
        rt = np.matmul(ref_rt, np.linalg.inv(warp_rt))
        v = np.array([1, 0, 0])
        t = rt[0:3, -1]
        r = rt[0:3, 0:3]

        ### rotating points, not axes. We will do backward warping
        angle = np.arccos(np.dot(t, v) / np.linalg.norm(t))
        axis = np.cross(v, t) / np.linalg.norm(np.cross(v, t))
        rect_r = R.from_rotvec(angle * axis)

        rect_r_cupy = cp.array(rect_r.as_matrix(), dtype=cp.float32)
        warp_r = cp.array(np.matmul(np.linalg.inv(r), rect_r.as_matrix()), dtype=cp.float32)

        rectify(ll_grid_size_single, block_size_single, (video_frames[0], video_frames[second_idx], rect_r_cupy, warp_r, reference_LL_cupy, neighbor_LL_cupy, h, w))

        pad_reference_LL_cupy[0:w // 8, :, :] = reference_LL_cupy[w * (1 - 1 / 8):, :, :]
        pad_reference_LL_cupy[w // 8:w + w // 8, :, :] = reference_LL_cupy[0:w, :, :]
        pad_reference_LL_cupy[w + w // 8:, :, :] = reference_LL_cupy[0:w * 1 / 8, :, :]

        pad_neighbor_LL_cupy[0:w // 8, :, :] = neighbor_LL_cupy[w * (1 - 1 / 8):, :, :]
        pad_neighbor_LL_cupy[w // 8:w + w // 8, :, :] = neighbor_LL_cupy[0:w, :, :]
        pad_neighbor_LL_cupy[w + w // 8:, :, :] = neighbor_LL_cupy[0:w * 1 / 8, :, :]

        ref = torch.tensor(pad_reference_LL_cupy).permute(2, 0, 1).unsqueeze(0)
        neigh = torch.tensor(pad_neighbor_LL_cupy).permute(2, 0, 1).unsqueeze(0)

        with torch.no_grad():
            flow_low, flow_up = model(ref, neigh, iters=12, test_mode=True)
        flow = flow_up[0, 0, w // 8:w + w // 8, :].cpu().numpy()

        if len(flow[flow < -5][:]) * 3 > flow.size:
            baseline_list.append(np.linalg.norm(t))

    baseline_list = np.sort(baseline_list)
    return baseline_list

def run_depth(config):
    c_args = []
    for k, v in config['data_path'].items():
        c_args += [f'--{k}', str(v)]
    for k, v in config['depth_estimation'].items():
        c_args += [f'--{k}', str(v)]

    device = "cuda:0"
    device_id = [0]

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
    parser.add_argument('--base_pth', help="restore checkpoint", default="")

    args = parser.parse_args(c_args)

    video_path = args.data_root_path + "/" + args.video_root_path + "/" + args.video_name
    depth_path = args.data_root_path + "/" + args.depth_root_path + "/" + args.video_name + "_depth"

    ## read trajectory and video
    video_frame_rt = video_util.read_csv(video_path + "/traj.csv")
    video = cv2.VideoCapture(video_path + "/video.mp4")
    video_frames = video_util.get_all_frames(video, args.depth_scale, args.sample_ratio)
    video_frame_rt = video_util.sample_rt(video_frame_rt, args.sample_ratio)
    frame_idx_list = video_util.get_frame_idx_list(video_frame_rt)

    ## load RAFT
    model = torch.nn.DataParallel(RAFT(args), device_ids=device_id)
    model.load_state_dict(torch.load(args.base_pth, map_location="cuda:0"))
    model.to(device)
    model.module.freeze_bn()
    model.eval()

    os.makedirs(depth_path, exist_ok=True)

    ## load cuda functions
    source = cuda_util.read_cuda_file("cuda/cuda_functions.cu")
    cuda_source = '''{}'''.format(source)
    module = cp.RawModule(code=cuda_source)
    rectify = module.get_function('rectifyLL')
    flow2depth = module.get_function('depthFromFlow')
    warp2ref = module.get_function('warpToRef')
    weightSum = module.get_function('weightSum')


    # cupy initialize

    h, w, c = video_frames[0].shape
    print("Shape: ", h, w, c)
    block_size_single = (32, 32)

    ll_grid_size_single = (h // block_size_single[1], w // block_size_single[0])
    eq_grid_size_single = (w // block_size_single[1], h // block_size_single[0])

    reference_LL_cupy = cp.zeros((w, h, 3), dtype=cp.uint8)
    neighbor_LL_cupy = cp.zeros((w, h, 3), dtype=cp.uint8)

    pad_reference_LL_cupy = cp.zeros((w * 5 // 4, h, 3), dtype=cp.uint8)
    pad_neighbor_LL_cupy = cp.zeros((w * 5 // 4, h, 3), dtype=cp.uint8)

    distance_LL_temp_cupy = cp.zeros((w, h), dtype=cp.float32)
    ref_weight_horizontal_cupy = cp.zeros((w, h), dtype=cp.float32)

    ref_distance_eq_cupy = cp.zeros((h, w), dtype=cp.float32)
    ref_weight_eq_cupy = cp.zeros((h, w), dtype=cp.float32)

    baseline_list = get_proper_baseline(video_frames, video_frame_rt, model, rectify)

    if len(baseline_list) == 0:
        print("Cannot find good pairs")
        return

    ###################################################################### Spherical Stereo ######################################################################
    for first_idx in frame_idx_list:
        if os.path.isfile(f"{depth_path}/{first_idx}.exr"):
            print(f"{depth_path}/{first_idx}.exr exists")
            continue

        print("first_idx: {}".format(first_idx))

        ## number of frames to estimate a single dynamic depth map
        second_idx_list = video_util.get_second_neighbors(
            first_idx, frame_idx_list, args.second_neighbor_interval, len(frame_idx_list), args.second_neighbor_size, video_frame_rt, baseline_list[0], baseline_list[-1])

        if len(second_idx_list) < args.second_neighbor_size:
            print("No good pairs found")
            continue
        print("second_idx_list: {}".format(second_idx_list))

        ## rt from first to ref
        ref_rt = video_frame_rt[first_idx][1]

        ## second loop
        second_depth_list = cp.zeros((h, w, len(second_idx_list)), dtype=cp.float32)
        second_weight_list = cp.zeros((h, w, len(second_idx_list)), dtype=cp.float32)

        ## run stereo
        for second_idx_index in range(len(second_idx_list)):
            second_idx = second_idx_list[second_idx_index]

            if second_idx == first_idx:
                continue

            warp_rt = video_frame_rt[second_idx][1]
            rt = np.matmul(ref_rt, np.linalg.inv(warp_rt))
            v = np.array([1, 0, 0])
            t = rt[0:3, -1]
            r = rt[0:3, 0:3]
            baseline = cp.asarray(np.linalg.norm(t))
            angle = np.arccos(np.dot(t, v) / np.linalg.norm(t))
            axis = np.cross(v, t) / np.linalg.norm(np.cross(v, t))
            rect_r = R.from_rotvec(angle * axis)
            rect_r_cupy = cp.array(rect_r.as_matrix(), dtype=cp.float32)
            warp_r = cp.array(np.matmul(np.linalg.inv(r), rect_r.as_matrix()), dtype=cp.float32)
            rect_r_inv_cupy = cp.array(np.linalg.inv(rect_r.as_matrix()), dtype=cp.float32)

            rectify(ll_grid_size_single, block_size_single, (video_frames[first_idx], video_frames[second_idx], rect_r_cupy, warp_r, reference_LL_cupy, neighbor_LL_cupy, h, w))

            pad_reference_LL_cupy[0:w // 8, :, :] = reference_LL_cupy[w * (1 - 1 / 8):, :, :]
            pad_reference_LL_cupy[w // 8:w + w // 8, :, :] = reference_LL_cupy[0:w, :, :]
            pad_reference_LL_cupy[w + w // 8:, :, :] = reference_LL_cupy[0:w * 1 / 8, :, :]

            pad_neighbor_LL_cupy[0:w // 8, :, :] = neighbor_LL_cupy[w * (1 - 1 / 8):, :, :]
            pad_neighbor_LL_cupy[w // 8:w + w // 8, :, :] = neighbor_LL_cupy[0:w, :, :]
            pad_neighbor_LL_cupy[w + w // 8:, :, :] = neighbor_LL_cupy[0:w * 1 / 8, :, :]

            ref = torch.tensor(pad_reference_LL_cupy).permute(2, 0, 1).unsqueeze(0)
            neigh = torch.tensor(pad_neighbor_LL_cupy).permute(2, 0, 1).unsqueeze(0)

            with torch.no_grad():
                _, flow_up = model(ref, neigh, iters=12, test_mode=True)

            flow = flow_up[0, 0, w // 8:w + w // 8, :].cpu().numpy()
            flow[flow > 0] = 1
            flow_cupy = cp.asarray(-flow, dtype=cp.float32)

            ## convert flows to depth, and calculate horizontal weights
            flow2depth(ll_grid_size_single, block_size_single,
                       (flow_cupy, distance_LL_temp_cupy, ref_weight_horizontal_cupy, baseline, h, w))

            # rotate from rectified view to original view, and calculate final weight, So return ref_distance_eq_cupy and ref_weight_eq_cupy
            warp2ref(eq_grid_size_single, block_size_single,
                     (distance_LL_temp_cupy, ref_distance_eq_cupy, ref_weight_horizontal_cupy, ref_weight_eq_cupy, rect_r_inv_cupy, w, h, cp.asarray(args.mask_radius, dtype=cp.float32)))

            # store depth maps to run ransac
            weightSum(eq_grid_size_single, block_size_single,
                      (ref_distance_eq_cupy, ref_weight_eq_cupy, second_depth_list, second_weight_list, len(second_idx_list), second_idx_index, w, h))

        # median using numpy
        numpy_second_depth_list = cp.asnumpy(second_depth_list)
        numpy_second_weight_list = cp.asnumpy(second_weight_list)
        and_result = np.all(numpy_second_weight_list, axis=2).astype(np.int)
        median = np.median(numpy_second_depth_list, axis=2)
        median[and_result == 0] = -1

        final_depth_np = 1 / median
        ## Save inverse depth
        cv2.imwrite(depth_path + "/" + "{}.exr".format(video_frame_rt[first_idx][0]), final_depth_np)

    return