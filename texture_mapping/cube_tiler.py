from texture_mapping.texture_utils import *
from texture_mapping.texture_opengl import framework


class CubeTiler:
    def __init__(self, faces, W, C, device='cpu', put_max_weight=False):
        self.device = device
        self.W = W
        self.C = C
        self.put_max_weight = put_max_weight

        self.opengl_renderer = framework.OpenglRendererPerspectiveWorldCoord(
            'CubeTiler',
            self.W, self.W,
            faces.numel() // 3, 4,
            faces
        )

    def init(self, target_extrinsic):
        self.xyz_map = self.__calculate_xyz_map(target_extrinsic)
        self.texture_map = torch.zeros(self.xyz_map.shape[:2] + (3,), dtype=torch.float32, device=self.device)
        self.weight_map = torch.zeros_like(self.texture_map[..., 0])


    def put_color(self, color, weight):
        if self.put_max_weight:
            maxier_mask = self.weight_map < weight
            self.texture_map[maxier_mask] = color[maxier_mask]
            self.weight_map[maxier_mask] = weight[maxier_mask]
        else:   #average
            self.texture_map += color * weight[..., None]
            self.weight_map += weight


    def get_textured(self):
        if self.put_max_weight:
            return self.texture_map
        else:
            return self.texture_map / (self.weight_map + (self.weight_map == 0))[..., None]


    def __calculate_xyz_map(self, target_extrinsic):
        cubeface_intrinsic = SpecialCameraMatrix.camera2image(
            self.W, self.W,
            hfov=90, near=0.001, far=10000
        )

        cubeface_rotations = np.stack(
            [SpecialCameraMatrix.axis_rotation(axis=1, angle=-np.pi / 2 * (i - 2)) for i in [3, 1]] + \
            [np.matmul(
                SpecialCameraMatrix.axis_rotation(axis=1, angle=np.pi),
                SpecialCameraMatrix.axis_rotation(axis=0, angle=-np.pi / 2 * i)) for i in [-1, 1]
            ] + \
            [SpecialCameraMatrix.axis_rotation(axis=1, angle=-np.pi / 2 * (i - 2)) for i in [0, 2]],
            axis=0
        ).transpose((0, 2, 1))

        target_extrinsic = torch2np(target_extrinsic)
        cubeface_extrinsics = np.matmul(cubeface_rotations, target_extrinsic)

        cubeface_camera_matrices = np.matmul(cubeface_intrinsic, SpecialCameraMatrix.extrinsic_our2gl(cubeface_extrinsics))
        cubeface_xyz_maps = self.opengl_renderer.render(cubeface_camera_matrices)

        pixels_range = torch.linspace(-1 + 1 / self.W, 1 - 1 / self.W, self.W).float().to(self.device)
        x, y = torch.meshgrid(pixels_range, pixels_range)
        ones = torch.ones_like(x.reshape(-1))
        pixels_ndc = torch2np(torch.stack([y.reshape(-1), -x.reshape(-1), ones, ones], dim=0))

        for i, camera_matrix in enumerate(cubeface_camera_matrices):
            no_mesh_mask = (np.linalg.norm(cubeface_xyz_maps[i], axis=2) <= 0.01)   # no-mesh pixels
            rays = np.matmul(np.linalg.inv(camera_matrix), pixels_ndc[:, no_mesh_mask.reshape(-1)])[:3]
            cubeface_xyz_maps[i][no_mesh_mask] = rays.T * (-MININF)

        return torch.Tensor(np.hstack(cubeface_xyz_maps)).float().to(self.device)