import logging
import time
import torch
import numpy as np
from utils import video_util
from numpy import pi
from cv2 import getGaussianKernel

logging.basicConfig(level=logging.INFO)
DEBUG = True
device="cuda:0"
tic_time = 0
MININF = -1000000
visibility_depth_eps = 0.02
raft_depth_eps = 0.5


def get_gaussian_kernel_separated(visibility_perturbation):
    G = torch.from_numpy(getGaussianKernel(visibility_perturbation * 2 + 1, visibility_perturbation * 2 / 3)).float().to(device)
    G_norm = torch.sqrt((G * G.T).sum())
    return G.T / G_norm

def torch2np(x):
    return x if isinstance(x, np.ndarray) else x.cpu().numpy()

def tic():
    global tic_time
    tic_time = time.time()

def toc(name=''):
    elapsed_time = time.time() - tic_time
    logging.info(f'    elapsed time ({name}): {elapsed_time:.2f} sec')
    return name, elapsed_time


def load_extrinsics_from_csv(csv_path, max_video_frames_count):
    world2frame = video_util.read_csv(csv_path)
    if len(world2frame) > max_video_frames_count:
        world2frame = world2frame[:max_video_frames_count]
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
    expand_batch = (extrinsics.dim() == 2)
    if expand_batch:
        extrinsics = extrinsics[None, ...]
    translations = -torch.bmm(extrinsics[:, :3, :3].transpose(1, 2), extrinsics[:, :3, [3, ]]).squeeze(dim=2)
    if expand_batch:
        translations = extrinsics.squeeze(dim=0)
    return translations


def transform_xyzs(xyzs, RT):
    assert xyzs.ndimension() == 2
    return torch.matmul(RT, torch.cat([
        xyzs.T,
        torch.ones((1, xyzs.shape[0]), device=xyzs.device, dtype=xyzs.dtype)
    ], dim=0)).transpose(-1, -2)[..., :3]


def xyzs2equirectxys(xyzs, H, RT=None):
    if not isinstance(xyzs, torch.Tensor):
        xyzs = torch.from_numpy(xyzs).to(RT.device if RT is not None else 'cpu')
    if RT is not None:
        xyzs = transform_xyzs(xyzs, RT)

    new_depth = xyzs.norm(dim=-1, keepdim=True)
    xyzs = xyzs / new_depth

    phi_new = 3 * pi / 2 - torch.atan2(xyzs[..., 2], xyzs[..., 0])
    theta_new = torch.acos(-xyzs[..., 1])

    W = H * 2
    u = theta_new * H / pi
    v = phi_new * W / (2 * pi)

    u[u >= H] -= H
    u[u < 0] += H
    v[v >= W] -= W
    v[v < 0] += W

    uv = torch.stack([u, v], dim=-1)
    return uv, new_depth.squeeze()


class MapInterpolation():
    @staticmethod
    def min(xys, map):
        xs = xys[..., 0]
        xs0 = torch.floor(xs).type(torch.long)
        xs1 = (xs0 + 1).clamp(max=map.shape[0]-1)

        ys = xys[..., 1]
        ys0 = torch.floor(ys).type(torch.long)
        ys1 = (ys0 + 1).clamp(max=map.shape[1]-1)

        interpolated = torch.stack([
            map[xs0, ys0],
            map[xs0, ys1],
            map[xs1, ys0],
            map[xs1, ys1],
        ], dim=-1).min(dim=-1)[0]

        return interpolated

    @staticmethod
    def bilinear(xys, map):
        xs = xys[..., 0]
        xs0 = torch.floor(xs).type(torch.long)
        xs1 = torch.ceil(xs).type(torch.long)
        xs1[xs1 == xs0] = (xs1[xs1 == xs0] + 1)
        xs1 = xs1.clamp(max=map.shape[0]-1)
        xs = xs - xs0

        ys = xys[..., 1]
        ys0 = torch.floor(ys).type(torch.long)
        ys1 = torch.ceil(ys).type(torch.long)
        ys1[ys1 == ys0] = (ys1[ys1 == ys0] + 1)
        ys1 = ys1.clamp(max=map.shape[1]-1)
        ys = ys - ys0

        map_squeezed = False
        if map.ndim == 2:
            map = map[..., None]
            map_squeezed = True
        interpolated =\
            map[xs0, ys0] * ((1-xs) * (1-ys)).unsqueeze(-1) + \
            map[xs0, ys1] * ((1-xs) * ys).unsqueeze(-1) + \
            map[xs1, ys0] * (xs * (1-ys)).unsqueeze(-1) + \
            map[xs1, ys1] * (xs * ys).unsqueeze(-1)

        if map_squeezed:
            interpolated = interpolated.squeeze(-1)
        return interpolated


class SpecialCameraMatrix():

    @staticmethod
    def extrinsic_our2gl(our):
        our2gl = np.eye(our.shape[-2]).astype('float32')
        our2gl[[1, 2], [1, 2]] = -1
        return np.matmul(our2gl, torch2np(our))

    @staticmethod
    def camera2image(H, W, hfov, near, far):
        inv_tan_half_hfov = 1 / np.tan(np.deg2rad(hfov / 2))
        aspect_ratio = H / W

        K = np.zeros((4, 4))
        K[0, 0] = inv_tan_half_hfov
        K[1, 1] = inv_tan_half_hfov / aspect_ratio
        K[2, 2] = - (far + near) / (far - near)
        K[2, 3] = - 2 * far * near / (far - near)
        K[3, 2] = -1

        return K

    @staticmethod
    def circular_translation_offset(radius, n):
        angles = np.linspace(0, 2 * np.pi, n + 1)[:-1].astype('float32')
        cosangles = np.cos(angles)
        sinangles = np.sin(angles)

        zd = 0.5
        return radius * np.vstack([
            cosangles,
            sinangles,
            sinangles * zd
        ]).T

    @staticmethod
    def axis_rotation(axis, angle, ndim=4):
        R = np.eye(ndim)
        R[(axis+1)%3, (axis+1)%3] = np.cos(angle)
        R[(axis+2)%3, (axis+2)%3] = np.cos(angle)
        R[(axis+1)%3, (axis+2)%3] = -np.sin(angle)
        R[(axis+2)%3, (axis+1)%3] = np.sin(angle)
        return R


def visibility_function(mesh_depth, rendered_depth, raft_depth):

    return torch.logical_and(
        mesh_depth <= rendered_depth * (1 + visibility_depth_eps),
        rendered_depth <= raft_depth + raft_depth_eps
    )