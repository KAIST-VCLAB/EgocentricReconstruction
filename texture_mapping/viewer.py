import cv2
import os
import gc
from texture_mapping.texture_utils import *
from texture_mapping.equirect_texture_sampler import EquirectTextureSampler
from texture_mapping.equitriangle_tiler import EqtriangleTiler


class CameraMatrixLooper():
    def __init__(self, intrinsic, extrinsic, circular_offset_list, enable_autorotate=False):
        self.intrinsic = intrinsic.astype('float32')
        self.intrinsic_aspect_ratio = self.intrinsic[1, 1] / self.intrinsic[0, 0]

        self.camera_matrix_circenter = None
        self.extrinsic_ref = None
        self.extrinsic_circenter_movement = None
        self.extrinsic_circenter_rotation = None
        self.set_extrinsic_ref(extrinsic)

        self.circular_offset_mats = None
        self.set_circular_offset_list(circular_offset_list)

        self.translation_offset_pinned = False

        self.enable_autorotate = enable_autorotate

    def set_extrinsic_ref(self, extrinsic_ref, translation_3=None, rotation_mat_3x3=None):
        extrinsic_ref = torch2np(extrinsic_ref)
        self.extrinsic_ref = extrinsic_ref.astype('float32')
        self.extrinsic_circenter_movement = np.zeros((4,4)).astype('float32')
        self.extrinsic_circenter_rotation = np.eye(4).astype('float32')
        self.set_extrinsic_circenter()

        self.autorotate_angle = 0
        self.autorotating = 1

        if translation_3 is not None:
            self.translate_circenter(translation_3)
        if rotation_mat_3x3 is not None:
            self.rotate_circenter(rotation_mat_3x3)

    def set_extrinsic_circenter(self):
        extrinsic_circenter = np.matmul(
            self.extrinsic_circenter_rotation,
            self.extrinsic_ref + self.extrinsic_circenter_movement)
        self.camera_matrix_circenter = np.matmul(self.intrinsic, extrinsic_circenter)

        extrinsic_circenter[:3, 3] = 0
        self.camera_matrix_circenter_rotationonly = np.matmul(self.intrinsic, extrinsic_circenter)

    def rotate_circenter(self, rotation_mat_3x3):
        self.extrinsic_circenter_rotation = np.eye(4).astype('float32')
        self.extrinsic_circenter_rotation[:3, :3] = rotation_mat_3x3
        self.set_extrinsic_circenter()

    def translate_circenter(self, translation_3):
        translation_mat = np.zeros((4,4)).astype('float32')
        translation_mat[:3, 3] = torch2np(translation_3).reshape(-1)
        self.extrinsic_circenter_movement[:3, 3] += np.matmul(
            self.extrinsic_circenter_rotation[:3,:3].T,
            torch2np(translation_3).reshape(-1, 1)
        ).reshape(-1)
        self.set_extrinsic_circenter()
        return self.extrinsic_circenter_movement[:3, 3]

    def set_circular_offset_list(self, translation_offset_list):
        self.circular_offset_mats = np.hstack([
            np.dstack([
                np.eye(3)[None, ...]. repeat(translation_offset_list.shape[0], axis=0),
                -translation_offset_list[:, :, None]
            ]),
            np.zeros(translation_offset_list.shape[:1] + (1, 4))
        ]).astype('float32')
        self.circular_offset_mats[:, 3, 3] = 1

        self.translation_offset_n = self.circular_offset_mats.shape[0]
        self.i = -1

    def change_hfov(self, delta):
        if not(0.2 <= self.intrinsic[0, 0] + delta <= 10):
            return self.get_hfov()

        self.intrinsic[0, 0] += delta
        self.intrinsic[1, 1] = self.intrinsic[0, 0] * self.intrinsic_aspect_ratio
        self.set_extrinsic_circenter()

        return self.get_hfov()

    def set_hfov(self, hfov):
        if not(30 <= hfov <= 160):
            return self.get_hfov()

        inv_tan_half_hfov = 1 / np.tan(np.deg2rad(hfov / 2))

        self.intrinsic[0, 0] = inv_tan_half_hfov
        self.intrinsic[1, 1] = self.intrinsic[0, 0] * self.intrinsic_aspect_ratio
        self.set_extrinsic_circenter()

        return hfov

    def get_hfov(self):
        return round(np.rad2deg(np.arctan(1 / self.intrinsic[0, 0])) * 2)

    def pin_translation_offset(self):
        self.translation_offset_pinned = ~self.translation_offset_pinned

    def progress(self):
        if self.translation_offset_pinned:
            return self.camera_matrix_circenter
        else:
            self.i += 1
            if self.i >= self.translation_offset_n:
                self.i = 0

            if self.enable_autorotate:
                if self.i % ((self.translation_offset_n + 1) // 4) == 1:
                    self.autorotating = 1 - self.autorotating
                self.autorotate_angle -= self.autorotating * 0.6 / 180 * np.pi
                if abs(self.autorotate_angle) >= 2*np.pi:
                    exit()
                if self.autorotating:
                    self.rotate_circenter(
                        SpecialCameraMatrix.axis_rotation(1, self.autorotate_angle, ndim=3)
                    )

            return np.matmul(
                self.camera_matrix_circenter_rotationonly,
                self.circular_offset_mats[self.i]
            )


class TextureMapProviderBase():

    def get_current_i(self):
        return self.current_i

    def get_current_name(self):
        if self.img_path_list is None:
            return f'frame{self.current_i:04}'
        return os.path.basename(self.img_path_list[self.current_i])

    def get_current_extrinsic(self):
        return self.extrinsics[self.current_i]

    def get_center_extrinsic(self):
        return self.extrinsics[0]

    def get_next(self):
        self.current_i += 1
        if self.current_i >= self.img_n:
            self.current_i = 0
        return self.get_current()

    def get_prev(self):
        self.current_i -= 1
        if self.current_i < 0:
            self.current_i = self.img_n - 1
        return self.get_current()


class TextureMapProviderCubeTiler(TextureMapProviderBase):

    def __init__(self, extrinsics, img_path_list=None, preload=False, img_list=None):
        assert (img_path_list is not None) ^ (img_list is not None)

        self.extrinsics_our = extrinsics
        self.extrinsics = SpecialCameraMatrix.extrinsic_our2gl(extrinsics)
        self.img_n = len(self.extrinsics)
        self.current_i = 0

        self.img_path_list = img_path_list
        self.cubefaces_preloaded = img_list

        if self.img_path_list is not None and preload:
            self.cubefaces_preloaded = []
            for i, path in enumerate(self.img_path_list):
                self.cubefaces_preloaded.append(self.__load_cubefaces_from_file(path))
                logging.info(f'[TextureMapProviderCubeTiler] Loaded {path} ({round(i / self.img_n * 100)}%)')


    def __load_cubefaces_from_file(self, path):
        cube_img = cv2.imread(path)[:,:,[2,1,0]] / 255.0
        cube_side = cube_img.shape[1] // 4
        cubefaces = [   # obey opengl cubemap ordering of cube faces
            cube_img[cube_side*1:cube_side*2, cube_side*0:cube_side*1],
            cube_img[cube_side*1:cube_side*2, cube_side*2:cube_side*3],
            cube_img[cube_side*2:cube_side*3, cube_side*3:cube_side*4],
            cube_img[cube_side*0:cube_side*1, cube_side*1:cube_side*2][::-1, ::-1, :],
            cube_img[cube_side*1:cube_side*2, cube_side*3:cube_side*4],
            cube_img[cube_side*1:cube_side*2, cube_side*1:cube_side*2],
        ]
        return [cubeface[::-1] for cubeface in cubefaces]   # a list of upside-down six cube faces


    def get_current(self):
        gc.collect()
        name = self.get_current_name()
        if self.cubefaces_preloaded is None:
            return self.__load_cubefaces_from_file(self.img_path_list[self.current_i]), name
        else:
            return self.cubefaces_preloaded[self.current_i], name


class TextureMapProviderEquitriangleTiler(TextureMapProviderBase):

    def __init__(self, video, frame_i_list, img_path_list, extrinsics,
                 xyz_map, facei_map, human_mask,
                 visibility_computer, raft_depth_paths):
        self.video = video
        self.frame_i_list = frame_i_list
        self.img_path_list = img_path_list
        self.img_n = len(self.img_path_list)
        self.current_i = 0

        self.extrinsics_our = extrinsics
        self.extrinsics = SpecialCameraMatrix.extrinsic_our2gl(extrinsics)

        self.mode = 0
        self.mode_n = 4
        self.xyz_map = xyz_map
        self.facei_map_our = facei_map
        self.facei_mask = self.facei_map_our >= 0
        self.facei_map = torch2np(facei_map)
        self.human_mask = human_mask

        self.visibility_computer = visibility_computer
        self.raft_depth_paths = raft_depth_paths

        self.last_i = -1

    def get_current(self):

        name = self.get_current_name()
        if self.frame_i_list[self.current_i] == -1:
            texture_map = cv2.imread(self.img_path_list[self.current_i]).astype('float32')[:, :, ::-1]
        else:
            if self.current_i != self.last_i:
                logging.info(f'[TextureMapProvider] image_warp')

                visibility_map = torch.zeros(self.facei_mask.shape)
                visibility_map[self.facei_mask] = self.visibility_computer(
                    self.xyz_map[self.facei_mask], self.extrinsics_our[self.current_i], self.raft_depth_paths[self.current_i]
                ).cpu()

                texture_map = EquirectTextureSampler.from_file(
                    self.video,
                    self.frame_i_list[self.current_i],
                    self.human_mask,
                    'cpu',
                    visibility_map=visibility_map,
                ).sample_from_xyz_map(
                    xyz_map=self.xyz_map,
                    RT=self.extrinsics_our[self.current_i],
                    mask=self.facei_mask,
                    mark_invisible=True,
                )

                logging.info(f'[TextureMapProvider] generate texture_map_candidate')

                texture_map = texture_map[:,:,[2,1,0]]
                texture_map = EqtriangleTiler.simple_inpainting(texture_map, self.facei_map >= 0, inpaint_iteration=2).clip(0, 255)

        return texture_map, name


class BackgroundCubemapper:
    def __init__(self, W, device):
        self.device = device
        self.W = W
        self.cubeface_xyz_maps = self.__calculate_front_cubeface_xyz_maps()
        self.cubefaces = torch.zeros((6, self.W, self.W, 3), dtype=torch.float32, device=device)
        self.cubefaces_mask = torch.zeros_like(self.cubefaces[..., 0], dtype=torch.bool)


    def put_cubeface(self, texture_sampler, depth_renderer, extrinsic):
        extrinsic_notrans = extrinsic.clone()
        extrinsic_notrans[:3, 3] = 0

        depth_renderer.set_extrinsic(extrinsic)
        no_mesh_mask = depth_renderer.sample_depth_from_xyzs(self.cubeface_xyz_maps).to(self.device) <= 0

        for cube_i in range(6):
            if self.cubefaces_mask[cube_i].all():
                continue

            texture_map = texture_sampler.sample_from_xyz_map(
                xyz_map=self.cubeface_xyz_maps[cube_i],
                RT=extrinsic_notrans,
                mask=~self.cubefaces_mask[cube_i],
            )
            update_mask = torch.logical_and(~self.cubefaces_mask[cube_i], no_mesh_mask[cube_i])
            self.cubefaces[cube_i, update_mask] = texture_map[update_mask]
            self.cubefaces_mask[cube_i, update_mask] = True

    def load(self, dump_path_format):
        if not os.path.exists(dump_path_format % 0):
            return False

        cubefaces = []
        for i in range(6):
            cubefaces.append(cv2.imread(dump_path_format % i))
        self.cubefaces = torch.from_numpy(np.stack(cubefaces, axis=0)).float()

        return True

    def dump(self, dump_path_format):
        os.makedirs(os.path.dirname(dump_path_format), exist_ok=True)
        cubefaces = self.cubefaces.clamp(0, 255).cpu().numpy()
        for i in range(6):
            cv2.imwrite(dump_path_format % i, cubefaces[i].astype('uint8'))

    def cubefaces_for_opengl(self):
        self.cubefaces = torch2np(self.cubefaces)
        self.cubefaces_mask = None
        return self.cubefaces[:,::-1,:,::-1] / 255.0


    def __calculate_front_cubeface_xyz_maps(self):

        intrinsic = torch.from_numpy(SpecialCameraMatrix.camera2image(
            self.W, self.W,
            hfov=90, near=0.001, far=10000
        )).float().to(self.device)

        cubeface_rotations = torch.from_numpy(np.stack(
            [SpecialCameraMatrix.axis_rotation(axis=1, angle=-np.pi / 2 * (i - 2)) for i in [3, 1]] + \
            [np.matmul(
                SpecialCameraMatrix.axis_rotation(axis=1, angle=np.pi),
                SpecialCameraMatrix.axis_rotation(axis=0, angle=-np.pi / 2 * i)) for i in [-1, 1]
            ] + \
            [SpecialCameraMatrix.axis_rotation(axis=1, angle=-np.pi / 2 * (i - 2)) for i in [0, 2]],
            axis=0
        )).float().to(self.device)

        pixels_range = torch.linspace(-1 + 1 / self.W, 1 - 1 / self.W, self.W).float().to(self.device)
        x, y = torch.meshgrid(pixels_range, pixels_range)
        ones = torch.ones_like(x.reshape(-1))
        pixels_ndc = torch.stack([-y.reshape(-1), x.reshape(-1), ones, ones], dim=0)

        cubeface_xyz_maps = []
        for cubeface_R in cubeface_rotations:
            camera_matrix = torch.matmul(intrinsic, cubeface_R.T)
            rays = torch.matmul(torch.inverse(camera_matrix), pixels_ndc)[:3].T
            cubeface_xyz_maps.append(rays.reshape((self.W, self.W, -1)))

        return torch.stack(cubeface_xyz_maps, dim=0)