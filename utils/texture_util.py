from texture_mapping import texture_main
from utils import rendering_trajectory

def run_texture(config):
    data_config = config['data_path']

    rendering_config = config['rendering_path']
    rendering_trajectory.rendering_traj(
        data_config['data_root_path'],
        data_config['video_root_path'],
        data_config['video_name'],
        rendering_config['spline_num'],
        rendering_config['rendering_num'],
        rendering_config['scale_simple'],
        rendering_config['scale_bbox'])

    texture_args = [
        '--data_root_path', data_config['data_root_path'],
        '--dir_input_video', data_config['video_root_path'],
        '--dir_depth_output', data_config['depth_root_path'],
        '--dir_3d_output', data_config['mesh_root_path'],
        '--dir_rendering_output', data_config['rendering_root_path'],
        '--video_name', data_config['video_name'],
    ]
    for k, v in config['texture_mapping'].items():
        texture_args += [f'--{k}', str(v)]

    texture_main.main(texture_args)