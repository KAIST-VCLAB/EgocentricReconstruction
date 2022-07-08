from texture_mapping.texture_utils import *
import cv2
import torch
import os
import gc

from texture_mapping.texture_opengl import framework

class EquirectDepthRenderer:

    def __init__(self, faces, near, far, device, cubeside):
        self.rendering_fov = 110   # 90 deg. + add space
        self.cubeside = cubeside
        self.device = device

        self.opengl_renderer = framework.OpenglRendererPerspectiveWorldCoord(
            'MeshDepthSampler360',
            self.cubeside, self.cubeside,
            faces.numel() // 3, 4,
            faces
        )
        self.intrinsic = SpecialCameraMatrix.camera2image(
            self.cubeside, self.cubeside,
            hfov=self.rendering_fov, near=near, far=far
        )

        self.rotations = np.stack([
            SpecialCameraMatrix.axis_rotation(axis=1, angle=-np.pi / 2 * (i-2))
            for i in range(4)
        ] + [
            SpecialCameraMatrix.axis_rotation(axis=0, angle=-np.pi / 2 * i)
            for i in [1, -1]
        ], axis=0)
        self.extrinsic_our2gl = SpecialCameraMatrix.extrinsic_our2gl(np.eye(4))
        self.extrinsics = []


    def set_extrinsic(self, extrinsic, depth_maps=None):
        if not isinstance(extrinsic, np.ndarray):
            extrinsic = extrinsic.cpu().numpy()
        self.extrinsic = torch.from_numpy(extrinsic).float().to(self.device)
        self.extrinsics = np.matmul(self.rotations, extrinsic)
        self.extrinsics = np.matmul(self.extrinsic_our2gl, self.extrinsics)
        self.camera_matrices = np.matmul(self.intrinsic, self.extrinsics)

        if depth_maps is None:
            rendered = self.opengl_renderer.render(self.camera_matrices, self.extrinsics)
            self.depth_maps = torch.from_numpy(np.stack(rendered, axis=0)).float().to(self.device).norm(dim=3)
        else:
            self.depth_maps = depth_maps
            display_cube_depth_map = False
            if display_cube_depth_map:
                depth_maps_img = 1 / torch.cat([
                    torch.cat([self.depth_maps[i] for i in range(3)], dim=1),
                    torch.cat([self.depth_maps[i] for i in range(3,6)], dim=1),
                ], dim=0).cpu().numpy()
                cv2.imshow('mesh depth cube', depth_maps_img /2)
                cv2.waitKey(100)

        self.camera_matrices = torch.Tensor(self.camera_matrices).float().to(self.device)


    def resample_equirect_depth_to_cube_depth(self, equirect_depth):
        extrinsics = np.matmul(self.extrinsic_our2gl, self.rotations)
        cubeface_camera_matrices = torch.from_numpy(np.matmul(self.intrinsic, extrinsics)).float().to(self.device).inverse()

        W = self.cubeside
        pixels_range = torch.linspace(-1 + 1 / W, 1 - 1 / W, W).float().to(self.device)
        x, y = torch.meshgrid(pixels_range, pixels_range)
        ones = torch.ones_like(x.reshape(-1))
        pixels_ndc = torch.stack([y.reshape(-1), -x.reshape(-1), ones, ones], dim=0)
        rays = torch.matmul(cubeface_camera_matrices, pixels_ndc)[:, :3, :].transpose(1, 2).reshape(6, W, W, 3)

        xys, _ = xyzs2equirectxys(rays, equirect_depth.shape[0])
        cube_depth = MapInterpolation.min(xys, equirect_depth).clamp(max=-MININF)

        return cube_depth

    def sample_depth_from_xyzs(self, xyzs, thetas=None, phis=None, depth_maps=None, interpolate_min=True):
        if thetas is None or phis is None:
            thetas, phis = self.__xyzs2thetaphi(xyzs)
        if depth_maps is None:
            depth_maps = self.depth_maps

        sampled_depth = torch.zeros(xyzs.shape[:-1]).float().to(self.device)
        mask_acc = torch.zeros(xyzs.shape[:-1]).bool().to(self.device)

        phis[phis >= 7/4*np.pi] -= 2 * np.pi
        phis[phis <= -1/4*np.pi] += 2 * np.pi

        theta_mask = torch.logical_and(thetas > np.pi/4, thetas < 3*np.pi/4)
        for i in range(6):
            if i < 4:
                phi_begin = np.pi * (-1/4 + i/2)
                mask = torch.logical_and(
                    ~mask_acc,
                    torch.logical_and(
                        theta_mask,
                        torch.logical_and(
                            phis >= phi_begin,
                            phis <= phi_begin + np.pi/2
                        )
                    )
                )
            elif i == 4:
                mask = torch.logical_and(thetas <= np.pi/4, ~mask_acc)
            else:
                mask = ~mask_acc

            xys = self.__xyzs2xys(xyzs[mask], self.camera_matrices[i])
            if torch.sum(mask) == 0:
                return None

            if interpolate_min:
                sampled_depth[mask] = MapInterpolation.min(xys, depth_maps[i])
            else:
                sampled_depth[mask] = MapInterpolation.bilinear(xys, depth_maps[i])
            mask_acc[mask] = True

        return sampled_depth


    def render_equirect(self, H=512, display=False):
        W = H*2
        x, y = torch.meshgrid(
            torch.arange(0, H).long().to(self.device),
            torch.arange(0, W).long().to(self.device)
        )
        xys = torch.stack([x.reshape(-1), y.reshape(-1)], dim=1)

        theta = np.pi * xys[:, 0] / H
        phi = 3 * np.pi / 2 - 2 * np.pi * xys[:, 1] / W
        sintheta = torch.sin(theta)
        xyzs = torch.stack([
            sintheta * torch.cos(phi),
            -torch.cos(theta),
            sintheta * torch.sin(phi),
        ], dim=1)

        xyzs = torch.cat([xyzs, torch.ones_like(sintheta)[..., None]], dim=1)
        xyzs = torch.matmul(torch.inverse(self.extrinsic), xyzs.T)

        xyzs = xyzs[:3, :].T

        equirect = self.sample_depth_from_xyzs(xyzs).reshape((H, W))

        if display:
            cv2.imshow('equirect', 1/equirect.clamp(min=0.00001).cpu().numpy()/2)
            cv2.waitKey()
        return equirect


    def __xyzs2xys(self, xyzs, camera_matrix):
        multiplied = torch.matmul(
            camera_matrix,
            torch.cat([xyzs.T, torch.ones_like(xyzs.T[[0]])], dim=0)
        )
        xys = (multiplied[[1,0]] / multiplied[[3]] + 1) / 2
        xys[0] = 1 - xys[0]
        return xys.T * self.cubeside


    def __xyzs2thetaphi(self, xyzs):
        xyzs = torch.cat([xyzs, torch.ones_like(xyzs[..., [0]])], dim=-1)
        xyzs = torch.matmul(self.extrinsic, xyzs[..., None]).squeeze(-1)[..., :3]
        xyzs /= xyzs.norm(dim=-1, keepdim=True)
        phi = 3 * np.pi / 2 - torch.atan2(xyzs[..., 2], xyzs[..., 0])
        theta = torch.acos(-xyzs[..., 1])
        return theta, phi


    def dump_equirects(self, extrinsics, out_paths, H):
        for i, out_path in enumerate(out_paths):
            if os.path.exists(out_path):
                logging.info(f'[dump_equirects] mesh_depth_equirect: exist {out_path}, pass')
            else:
                logging.info(f'[dump_equirects] mesh_depth_equirect: save {out_path}')
                self.set_extrinsic(extrinsics[i])
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                depth_map = self.render_equirect(H=H, display=False)
                disparity_map = 1 / depth_map
                disparity_map[depth_map < 0.01] = MININF
                cv2.imwrite(out_path, torch2np(disparity_map))

                gc.collect()
                torch.cuda.empty_cache()


    def convert_raft_depth(self, in_paths, out_paths, H):
        for in_path, out_path in zip(in_paths, out_paths):
            if os.path.exists(out_path):
                logging.info(f'[convert_raft_depth] mesh_depth_equirect: exist {out_path}, pass')
            else:
                logging.info(f'[convert_raft_depth] convert: {in_path} --> {out_path}')
                in_img = cv2.imread(in_path, cv2.IMREAD_UNCHANGED)
                in_img[in_img < 0] = MININF
                W = int(np.round(H / in_img.shape[0] * in_img.shape[1]))

                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                out_img = cv2.resize(in_img, (W, H), interpolation=cv2.INTER_LINEAR)
                cv2.imwrite(out_path, out_img)
