import numpy as np
from scipy.spatial.transform import Rotation as R
from utils import video_util
import matplotlib
matplotlib.use('Qt5Agg')
from scipy.interpolate import CubicSpline


def corner_points_from_bbox(bbox, order):
    """
    indexing:
      4----5
     /|   /|
    0----1 |
    | 6--|-7
    |/   |/
    2----3
    """
    return np.vstack([bbox[i, (order >> i) & 1] for i in range(3)])


def sample_euler_angles(num):
    angles_x = []
    angles_y = []
    angles_z = []
    for i in range(num):
        angles_x.append(0)
        angles_y.append(0)
        angles_z.append(0)
    return np.vstack((np.array(angles_x), np.array(angles_y), np.array(angles_z)))

def spline_loc_rot(frame_num, bbox_input, corner_traverse):
    location_sample_num = len(corner_traverse)
    angle_sample_num = len(corner_traverse)

    sorted_points = np.sort(bbox_input, axis=1)
    points = corner_points_from_bbox(bbox_input, corner_traverse)
    angles = sample_euler_angles(angle_sample_num)
    spline_x = CubicSpline(range(location_sample_num), points[0, :])
    spline_y = CubicSpline(range(location_sample_num), points[1, :])
    spline_z = CubicSpline(range(location_sample_num), points[2, :])
    spline_angle_x = CubicSpline(range(angle_sample_num), angles[0, :])
    spline_angle_y = CubicSpline(range(angle_sample_num), angles[1, :])
    spline_angle_z = CubicSpline(range(angle_sample_num), angles[2, :])
    location_sample = np.arange(0, location_sample_num - 1, (location_sample_num - 1) / frame_num)
    angle_sample = np.arange(0, angle_sample_num - 1, (angle_sample_num - 1) / frame_num)
    curve_x = spline_x(location_sample)
    curve_y = spline_y(location_sample)
    curve_z = spline_z(location_sample)
    curve_angle_x = spline_angle_x(angle_sample)
    curve_angle_y = spline_angle_y(angle_sample)
    curve_angle_z = spline_angle_z(angle_sample)
    curve_x[curve_x > sorted_points[0, 1]] = sorted_points[0, 1]
    curve_x[curve_x < sorted_points[0, 0]] = sorted_points[0, 0]
    curve_y[curve_y > sorted_points[1, 1]] = sorted_points[1, 1]
    curve_y[curve_y < sorted_points[1, 0]] = sorted_points[1, 0]
    curve_z[curve_z > sorted_points[2, 1]] = sorted_points[2, 1]
    curve_z[curve_z < sorted_points[2, 0]] = sorted_points[2, 0]
    curve_points = np.vstack((curve_x, curve_y, curve_z))
    curve_angles = np.vstack((curve_angle_x, curve_angle_y, curve_angle_z))

    return curve_points, curve_angles


def save_csv(rt_list, rendering_traj_path, use_num):
    with open(rendering_traj_path, 'w') as csvfile:
        for i in range(use_num):
            csvfile.write(str(rt_list[i][0]) + " ")
            csvfile.write(str(rt_list[i][1][0,0])+ " ")
            csvfile.write(str(rt_list[i][1][0,1])+ " ")
            csvfile.write(str(rt_list[i][1][0,2])+ " ")
            csvfile.write(str(rt_list[i][1][0,3])+ " ")
            csvfile.write(str(rt_list[i][1][1,0])+ " ")
            csvfile.write(str(rt_list[i][1][1,1])+ " ")
            csvfile.write(str(rt_list[i][1][1,2])+ " ")
            csvfile.write(str(rt_list[i][1][1,3])+ " ")
            csvfile.write(str(rt_list[i][1][2, 0])+ " ")
            csvfile.write(str(rt_list[i][1][2, 1])+ " ")
            csvfile.write(str(rt_list[i][1][2, 2])+ " ")
            csvfile.write(str(rt_list[i][1][2, 3])+ " ")
            csvfile.write(str(rt_list[i][1][3, 0])+ " ")
            csvfile.write(str(rt_list[i][1][3, 1])+ " ")
            csvfile.write(str(rt_list[i][1][3, 2])+ " ")
            csvfile.write(str(rt_list[i][1][3, 3]))
            csvfile.write("\n")
    return

import torch
device="cuda:0"

def load_extrinsics_from_csv(csv_path):
    ######################################################
    # Calculate mesh -> world transformation
    ######################################################
    world2frame = video_util.read_csv(csv_path)
    total_video_frames_count = len(world2frame)
    world2frame = torch.tensor([l[1] for l in world2frame]).to(device)

    return total_video_frames_count, world2frame


def calculate_center2world(world2frame):
    first2world = torch.inverse(world2frame[0])
    first2frame = torch.matmul(world2frame, first2world)
    t_in_first_mean = extrinsic2translation(first2frame[1:-1]).mean(axis=0)

    center2first = torch.eye(4, dtype=t_in_first_mean.dtype).to(device)
    center2first[:3, 3] = t_in_first_mean
    center2world = torch.matmul(first2world, center2first)

    return center2world


def extrinsic2translation(extrinsics):
    # extrinsics: CAM * 4 * 4, or 4 * 4
    # return: CAM * XYZ
    expand_batch = (extrinsics.dim() == 2)
    if expand_batch:
        extrinsics = extrinsics[None, ...]
    translations = -torch.bmm(extrinsics[:, :3, :3].transpose(1, 2), extrinsics[:, :3, [3, ]]).squeeze(dim=2)
    if expand_batch:
        translations = extrinsics.squeeze(dim=0)
    return translations


def calc_input_traj_bbox(extrinsics_csv_path):
    total_video_frames_count, extrinsics = load_extrinsics_from_csv(extrinsics_csv_path)
    center2world = calculate_center2world(extrinsics)
    extrinsics = torch.matmul(extrinsics, center2world)
    translations = extrinsic2translation(extrinsics)  # CAM * XYZ
    return torch.stack([translations.min(dim=0)[0], translations.max(dim=0)[0]], dim=1).cpu().numpy()

def calc_input_traj_simple(extrinsics_csv_path):
    total_video_frames_count, extrinsics = load_extrinsics_from_csv(extrinsics_csv_path)
    center2world = calculate_center2world(extrinsics)
    extrinsics = torch.matmul(extrinsics, center2world)
    translations = extrinsic2translation(extrinsics)  # CAM * XYZ
    return translations


def scale_bbox_wrt_center(bbox, ratio):
    bbox_center = bbox.mean(axis=1, keepdims=True)
    return bbox_center + (bbox - bbox_center) * ratio

def scale_simple_wrt_center(simple, ratio):
    simple_center = simple.mean(axis=0, keepdims=True)
    return simple_center + (simple - simple_center) * ratio


def rendering_traj(data_root_path, video_path, video_name, spline_num, rendering_num, scale_simple, scale_bbox):

    print("Save bbox rendering trajectory")
    rendering_traj_path = f"{data_root_path}/{video_path}/{video_name}/traj_extrapolation.csv"
    bbox_input = calc_input_traj_bbox(f"{data_root_path}/{video_path}/{video_name}/traj.csv")
    bbox_input = scale_bbox_wrt_center(bbox_input, scale_bbox)

    corner_traverse = np.array([0, 7, 2, 5, 4, 3, 6, 1, 0])
    curve_points, curve_angles = spline_loc_rot(spline_num, bbox_input, corner_traverse)


    rt_list = []
    for i in range(spline_num):
        r = R.from_rotvec(curve_angles[:, i]).as_matrix()
        t = np.array(curve_points[:, i])
        rt = np.identity(4)
        rt[0:3, 0:3] = r
        rt[0:3, 3] = t
        rt = np.linalg.inv(rt)
        rt_list.append([i, rt])
    save_csv(rt_list, rendering_traj_path, rendering_num)

    print("Save scaled rendering trajectory")
    rendering_traj_path = f"{data_root_path}/{video_path}/{video_name}/traj_interpolation.csv"
    simple_input = calc_input_traj_simple(f"{data_root_path}/{video_path}/{video_name}/traj.csv")
    simple_input = scale_simple_wrt_center(simple_input, scale_simple)

    curve_points = simple_input.cpu().numpy().T
    curve_angles = np.zeros_like(curve_points)

    rt_list = []
    for i in range(curve_points.shape[1]):

        r = R.from_rotvec(curve_angles[:, i]).as_matrix()
        t = np.array(curve_points[:, i])
        rt = np.identity(4)
        rt[0:3, 0:3] = r
        rt[0:3, 3] = t
        rt = np.linalg.inv(rt)
        rt_list.append([i, rt])
    save_csv(rt_list, rendering_traj_path, curve_points.shape[1])


















