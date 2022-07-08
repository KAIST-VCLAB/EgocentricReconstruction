import cv2
from texture_mapping.texture_utils import *

class EquirectTextureSampler():
    def __init__(self, image, rendered_depth_map, raft_depth_map, visibility_map, human_mask, device):
        self.device = device

        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image)
        if not isinstance(rendered_depth_map, torch.Tensor):
            rendered_depth_map = torch.from_numpy(rendered_depth_map)
        if not isinstance(human_mask, torch.Tensor):
            human_mask = torch.from_numpy(human_mask)

        self.image = image.to(self.device)
        self.rendered_depth_map = rendered_depth_map.to(self.device)
        self.raft_depth_map = raft_depth_map.to(self.device)
        if not visibility_map is None:
            self.visibility_map = visibility_map.to(self.device)
        else:
            self.visibility_map = None
        self.human_mask = human_mask.to(self.device)

        self.H = image.shape[0]
        self.W = image.shape[1]


    def sample_from_xyz_map(self, xyz_map, RT, mask=None, mark_invisible=False, return_distance_map=False, invalidate_human=False):
        if mask is None:
            mask = torch.ones(xyz_map.shape[:2])
        x, y = torch.nonzero(mask, as_tuple=True)

        xyzs = xyz_map[x, y]
        sampled_rgb, distances, sampled_rendered_depth, sampled_raft_depth = self.sample_from_xyzs(
            xyzs, RT, mark_invisible=mark_invisible, invalidate_human=invalidate_human
        )
        if mark_invisible and self.visibility_map.nelement() > 0:
            sampled_visibility_weight = self.visibility_map[x, y, None].repeat(1,3)
            sampled_visibility_weight[..., 1] = 0
            sampled_visibility_weight[..., 2] *= -1
            sampled_visibility_weight = sampled_visibility_weight.clamp(0,1) * 255
            sampled_rgb += sampled_visibility_weight

        torch.cuda.empty_cache()

        texture_map = torch.zeros(xyz_map.shape, dtype=sampled_rgb.dtype, device=self.device)
        if invalidate_human:
            texture_map -= 1
        texture_map[x, y] = sampled_rgb.to(self.device)

        if return_distance_map:
            distance_map = torch.zeros_like(texture_map[...,0], device='cpu')
            distance_map[x, y] = distances

            frontier_map = torch.zeros_like(texture_map[...,0], device='cpu')
            frontier_map[x, y] = sampled_rendered_depth

            frontier_map_raft = torch.zeros_like(texture_map[..., 0], device='cpu')
            if sampled_raft_depth.nelement() > 0:
                frontier_map_raft[x, y] = sampled_raft_depth

            return texture_map, distance_map, frontier_map, frontier_map_raft
        else:
            return texture_map


    def sample_from_xyzs(self, xyzs, RT, mark_invisible=False, invalidate_human=True):
        xys, zs = xyzs2equirectxys(xyzs, self.H, RT)
        xys = xys.to(self.device)
        zs = zs.to(self.device).clamp(max=-MININF)

        sampled_rgb = MapInterpolation.bilinear(xys, self.image)

        if invalidate_human:
            sampled_rgb[~self.human_mask[xys[:, 0].to(torch.long), xys[:, 1].to(torch.long)]] = -1
        elif mark_invisible:
            sampled_rgb[~self.human_mask[xys[:, 0].to(torch.long), xys[:, 1].to(torch.long)], 0] += 128

        if self.rendered_depth_map.nelement() == 0 or self.raft_depth_map.nelement() == 0:
            return sampled_rgb, torch.Tensor(), torch.Tensor(), torch.Tensor()

        sampled_rendered_depth = MapInterpolation.min(xys, self.rendered_depth_map).clamp(max=-MININF)
        sampled_raft_depth = MapInterpolation.min(xys, self.raft_depth_map).clamp(max=-MININF)

        if mark_invisible:
            invisible_mask = ~visibility_function(zs, sampled_rendered_depth, sampled_raft_depth)
            sampled_rgb[invisible_mask, 2] += 128

        return sampled_rgb, zs.cpu(), sampled_rendered_depth.cpu(), sampled_raft_depth.cpu()

    @staticmethod
    def from_file(video, frame_i, human_mask, device, rendered_depth_equirect=None, raft_depth_equirect=None, visibility_map=None):
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_i)
        success, video_frame = video.read()
        assert (success)

        if rendered_depth_equirect is None:
            rendered_depth_map = torch.Tensor()
        else:
            rendered_depth_map = 1 / torch.from_numpy(cv2.imread(rendered_depth_equirect, -1)).cpu()

        if raft_depth_equirect is None:
            raft_depth_map = torch.Tensor()
        else:
            raft_depth_map = 1 / torch.from_numpy(cv2.imread(raft_depth_equirect, -1)).cpu()

        if visibility_map is None:
            visibility_map = torch.Tensor()

        return EquirectTextureSampler(
            image=video_frame,
            rendered_depth_map=rendered_depth_map,
            raft_depth_map=raft_depth_map,
            visibility_map=visibility_map,
            human_mask=human_mask,
            device=device
        )