import cv2
import os
from texture_mapping.texture_utils import *
from texture_mapping.texture_opengl import framework

MININF = -1000000
#########

class EqtriangleTiler:
    def __init__(self, texture_map_max_sizeK, texture_map_min_triangle_side, faces, K, device='cpu'):
        self.faces = faces.cpu()
        self.face_n = faces.shape[0]

        a = int(np.ceil(np.sqrt(self.face_n / 2 / 2 / 2)))
        self.M = a * 2 * 2
        self.N = np.ceil(self.face_n / self.M)

        triangle_side = texture_map_max_sizeK * 1000 // (self.M / 2)
        assert triangle_side >= texture_map_min_triangle_side,\
            f'must be ... texture_map_min_triangle_side <= {triangle_side}, or , '\
            f'texture_map_max_sizeK >= {texture_map_min_triangle_side * (self.M // 2) / 1000}'
        self.triangle_side = int(triangle_side)
        self.triangle_side_half = int(np.floor(triangle_side / 2))
        self.triangle_height = int(np.round(triangle_side * np.sqrt(3) / 2))
        self.triangle_side_half = triangle_side / 2
        self.triangle_height = triangle_side * np.sqrt(3) / 2

        self.H = int(np.ceil(self.N * self.triangle_height))
        self.W = int(np.ceil(self.M / 2 * self.triangle_side + self.triangle_side_half + 1))

        self.device = device
        self.K = K
        self.texture_mask = torch.zeros((self.H, self.W), dtype=torch.float32).to(self.device)
        self.texture_uvcoord = None

        self.texture_map = None
        self.texture_map_topKoutlier = torch.zeros((self.H, self.W, self.K, 3), dtype=torch.float32) + MININF
        self.texture_merge_topKoutlier_prioritypenalty = None
        self.texture_merge_topKoutlier_colorthreshold = None
        self.texture_merge_topKoutlier_batchsizeM = 0

        a = 800
        self.texture_map_display_size = (int(np.round(self.W / self.H * a)), a)

        self.current_max_score = torch.zeros((self.H, self.W)).float().to(self.device) - 1

        self.xyz_map = None
        self.facei_map = None
        self.facei_count = 0

        logging.info('\n  - '.join([
            '[EqtriangleTiler]',
            f'triangle_side: {triangle_side}',
            f'Texture map size {self.W} x {self.H}'
        ]) + '\n')

        self.__calculate_texture_uvcoord()
        self.__calculate_xyzi_map(framework.OpenglRendererPlanar('EqtriangleTiler'))


    def put_texture_map(self, map, my_score=None):
        my_score = my_score.to(self.device)
        mask = torch.logical_and(self.facei_map >= 0, my_score[self.facei_map] >= -1)
        self.texture_map_topKoutlier[mask, my_score[self.facei_map[mask]].to(torch.long)] = map[mask].to(self.texture_map_topKoutlier.device)
        mask = torch.logical_and(self.facei_map >= 0, my_score[self.facei_map] >= -2)
        self.texture_mask[mask] += 1

    def save_obj_with_texture(self, obj_path, rainbow=False, save_mtl=True):
        mtl_path = obj_path[:-4] + '.mtl'
        png_path = obj_path[:-4] + '.png'

        logging.info(f'[save_obj_with_texture] Saving obj and mtl ...')
        os.makedirs(os.path.dirname(obj_path), exist_ok=True)
        with open(obj_path, "w") as f:
            f.write("mtllib " + os.path.basename(mtl_path) + "\n")
            f.write("usemtl triangle_tiling\n")
            f.write("s off\n")

            faces = self.faces.cpu().numpy()

            for fi, face in enumerate(faces):
                for v in face:
                    f.write(f'v {v[0]:.5f} {v[1]:.5f} {v[2]:.5f}\n')

                if rainbow:
                    f.write(f'vt 0 1\n')
                    f.write(f'vt 0.5 0\n')
                    f.write(f'vt 1 1\n')
                else:
                    for uv in self.texture_uvcoord[fi]:
                        f.write(f'vt {uv[0]:.7f} {uv[1]:.7f}\n')

                vi = fi * 3 + 1
                f.write(f'f {vi}/{vi} {vi + 1}/{vi + 1} {vi + 2}/{vi + 2}\n')

        if save_mtl:
            with open(mtl_path, "w") as f:
                f.write("newmtl triangle_tiling\n")
                f.write("Ka 1 1 1\n")
                f.write("Kd 0 0 0\n")
                f.write("Ks 0 0 0\n")
                f.write("Ns 1\n")
                f.write(f"map_Kd {os.path.basename(png_path)}\n")


    def inpaint_texture_map(self, map=None, mask=None, blur_size=(3,3), inpaint_iteration=2):
        if map is None:
            map = self.get_texture_map()
        if mask is None:
            mask = self.texture_mask > 0.5
        return EqtriangleTiler.simple_inpainting(
            map, mask,
            blur_size=blur_size,
            inpaint_iteration=inpaint_iteration
        )

    @staticmethod
    def simple_inpainting(map, mask, blur_size=(3,3), inpaint_iteration=2):
        if isinstance(map, torch.Tensor):
            map = map.cpu().numpy()
        else:
            map = map.copy()
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        map[~mask] = 0

        while inpaint_iteration > 0:
            logging.info(f'[simple_inpainting] iteration {inpaint_iteration}')
            inpaint_iteration -= 1

            mask_float = cv2.blur(mask.astype('float32'), blur_size)
            add_mask = np.logical_and(~mask, mask_float > 0.01)
            mask[add_mask] = True

            blurred_map = cv2.blur(map, blur_size)
            if map.ndim == 3: mask_float = mask_float[..., None]
            map[add_mask] = blurred_map[add_mask] / mask_float[add_mask]

        return map

    def topKoutlier(self):
        K = self.texture_map_topKoutlier.shape[2]
        topK_colors = self.texture_map_topKoutlier.view(-1, K, 3)

        colors = torch.zeros((0, 3)).cpu()

        batch_size = self.texture_merge_topKoutlier_batchsizeM
        total_batch = topK_colors.shape[0]

        priority_penalty = torch.linspace(self.texture_merge_topKoutlier_prioritypenalty, 0, K)[None, ...].to(self.device)

        last_batch_percent = 0
        for batch_begin in range(0, total_batch, batch_size):
            batch_percent = int(batch_begin / total_batch * 100)
            if batch_percent > last_batch_percent:
                logging.info(f'[EqtriangleTiler] topKoutlier removing ... ({batch_percent}%)')
                last_batch_percent = batch_percent
            batch_end = min(batch_begin + batch_size, total_batch)

            batch_topK_colors = topK_colors[batch_begin:batch_end].to(self.device)
            topK_dists = torch.cdist(
                batch_topK_colors,
                batch_topK_colors,
                p=1
            )
            invisible_penalty = -(batch_topK_colors[:,:,0] < 0).float() * 100000

            colors = torch.cat([
                colors,
                batch_topK_colors[
                    torch._dim_arange(batch_topK_colors, dim=0),
                    ((topK_dists < self.texture_merge_topKoutlier_colorthreshold).sum(axis=2) + invisible_penalty + priority_penalty).max(dim=1)[1]
                ].cpu()
            ], dim=0)

            del batch_topK_colors
            torch.cuda.empty_cache()

        self.texture_mask = self.texture_mask.clamp(max=1)

        return colors.view(self.H, self.W, 3).to(self.device)


    def average_blending(self):
        value_exist = (self.texture_map_topKoutlier >= 0).float()
        summed = (self.texture_map_topKoutlier * value_exist).sum(dim=2)
        return summed.to(self.device)

    def get_texture_map(self):
        if self.texture_map is None:
            use_simple_average = False
            if use_simple_average:
                value_exist = (self.texture_map_topKoutlier[..., [0]] >= 0).float().to(self.device)
                numerator = (self.texture_map_topKoutlier.to(self.device) * value_exist).sum(dim=2)
                self.texture_map = (numerator / value_exist.sum(dim=2).clamp(min=0.00000001)).cpu()
            else:
                self.texture_map = self.topKoutlier()
        return self.texture_map


    def load(self, dump_dir):
        dump_path_format = os.path.join(dump_dir, f'%02d.png')
        if not os.path.exists(dump_path_format % (self.K - 1)):
            return False

        texture_mask = torch.zeros_like(self.texture_mask, device='cpu')
        logging.info(f'[main] Loading topKoutlier dumpfile: {dump_dir}')
        for i in range(self.K):
            logging.info(
                f'[main] Loading topKoutlier dumpfile ... {os.path.basename(dump_path_format % i)}')
            dumped = torch.from_numpy(cv2.imread(dump_path_format % i).astype(np.float32)).float()
            dumped_mask = torch.from_numpy((cv2.imread(dump_path_format[:-4] % i + '_mask.png', -1) > 128).astype(np.bool))
            dumped[~dumped_mask] = MININF
            self.texture_map_topKoutlier[:, :, i, :] = dumped
            texture_mask += dumped_mask

        self.texture_mask = texture_mask.float().to(self.device)
        return True


    def dump(self, dump_dir):
        os.makedirs(dump_dir, exist_ok=True)
        dump_path_format = os.path.join(dump_dir, f'%02d.png')

        for i in range(self.K):
            if not os.path.exists(dump_path_format % i):
                logging.info(f'[main] Saving topKoutlier dumpfile ... {os.path.basename(dump_path_format % i)}')
                cv2.imwrite(
                    dump_path_format % i,
                    self.texture_map_topKoutlier[:, :, i, :].cpu().numpy()
                )
                cv2.imwrite(
                    dump_path_format[:-4] % i + '_mask.png',
                    (self.texture_map_topKoutlier[:, :, i, 0] >= 0).cpu().float().numpy() * 255
                )

    def __calculate_texture_uvcoord(self):
        if self.texture_uvcoord is None:
            self.texture_uvcoord = self.__xy_to_uv(self.__face_idx_to_xy(torch._dim_arange(self.faces, dim=0))).cpu().numpy()
        return self.texture_uvcoord


    def __render_xyzi_map(self, renderer):
        texcoord_data = self.texture_uvcoord * 2 - 1

        logging.info('[generate_xyzi_map] Rendering xyz map ...')
        xyz_map = renderer.render(
            self.H, self.W,
            self.faces.numel() // 3, 4,
            texcoord_data,
            self.faces,
            'xyz_map'
        )

        logging.info('[generate_xyzi_map] Rendering facei map ...')
        facei_map = renderer.render(
            self.H, self.W,
            self.faces.numel() // 3, 4,
            texcoord_data,
            (torch._dim_arange(self.faces, dim=0).float().to(self.device) + 1)[...,None].repeat(1,3).reshape(-1,1),
            'facei_map'
        )
        facei_map = np.round(facei_map[:, :, 0] - 1).astype(np.long)

        return [
            torch.from_numpy(xyz_map.copy()).to(self.device),
            torch.from_numpy(facei_map.copy()).to(self.device)
        ]

    def __calculate_xyzi_map(self, renderer):
        # ------------- generate coordinate map of the triangle tile -------------#
        xyz_map, facei_map = self.__render_xyzi_map(renderer)

        self.xyz_map = xyz_map
        self.facei_map = facei_map
        self.facei_count = torch.sum(facei_map >= 0).item()


    def __face_idx_to_xy(self, face_idx):
        triangle_idx = (face_idx // self.M, face_idx % self.M)

        offset_u_all = triangle_idx[0] * self.triangle_height
        offset_v_all = torch.floor(triangle_idx[1]/2.0) * self.triangle_side

        l = 1
        root3 = np.sqrt(3) * l
        faces_xy = torch.zeros((face_idx.numel(), 3, 2)).to(face_idx.device)

        mask = (face_idx % 2 == 1)
        offset_u = offset_u_all[mask]
        offset_v = offset_v_all[mask]
        faces_xy[mask] = torch.stack([
            torch.stack([
                offset_u + l,
                offset_v + self.triangle_side_half + root3
            ], dim=1),
            torch.stack([
                offset_u + self.triangle_height - root3,
                offset_v + self.triangle_side
            ], dim=1),
            torch.stack([
                offset_u + l,
                offset_v + self.triangle_side_half + self.triangle_side - root3
            ], dim=1),
        ], dim=1)

        mask = ~mask
        offset_u = offset_u_all[mask]
        offset_v = offset_v_all[mask]
        faces_xy[mask] = torch.stack([
            torch.stack([
                offset_u + self.triangle_height - l,
                offset_v + root3
            ], dim=1),
            torch.stack([
                offset_u + self.triangle_height - l,
                offset_v + self.triangle_side - root3
            ], dim=1),
            torch.stack([
                offset_u + root3,
                offset_v + self.triangle_side_half
            ], dim=1),
        ], dim=1)

        faces_xy[..., [0, 1]] = faces_xy[..., [1,0]]
        return faces_xy


    def __xy_to_uv(self, xys):
        xys[..., 0] = xys[..., 0] / self.W
        xys[..., 1] = 1 - xys[..., 1] / self.H
        return xys


    def __serialize_for_opengl(self, data, dim):
        data = data.reshape((-1, data.shape[-1]))
        return torch.cat([
            data,
            torch.ones((data.shape[0], dim - data.shape[1])).float().to(data.device)
        ], dim=1).reshape(-1)


    def __render(self, face_n, coord_data, color_data, renderer,
                 render_imsize=None, camera_matrix_list=None, texture_map=None, window_name=None):

        if render_imsize is None:
            render_imsize = (self.H, self.W)

        if camera_matrix_list is not None and not isinstance(camera_matrix_list, np.ndarray):
            camera_matrix_list = camera_matrix_list.cpu().numpy()

        if isinstance(texture_map, torch.Tensor):
            texture_map = texture_map.cpu().numpy()

        vertex_dim = 4
        vertex_n = face_n * 3

        vertex_data = torch.cat([self.__serialize_for_opengl(data, vertex_dim) for data in [coord_data, color_data]])
        vertex_data = vertex_data.cpu().numpy()

        rendered = renderer.render(render_imsize[0], render_imsize[1],
                                   vertex_data, vertex_dim, vertex_n,
                                   camera_matrix_list, texture_map,
                                   window_name=window_name)
        return rendered
