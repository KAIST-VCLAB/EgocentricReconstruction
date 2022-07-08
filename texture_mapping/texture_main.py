from texture_mapping import texture_main_single_texture_map
import torch
import sys
import argparse
import numpy as np

def parse_args(custom_args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root_path', type=str, default="demo")
    parser.add_argument('--dir_input_video', type=str, default="input_videos")
    parser.add_argument('--dir_depth_output', type=str, default="depth")
    parser.add_argument('--dir_3d_output', type=str, default="mesh")
    parser.add_argument('--dir_rendering_output', type=str, default="demo/rendering_output")

    #------------- in/output path params -------------#
    parser.add_argument('--video_name', type=str, default="synthetic_classroom",
                        help='The name of input video file (without extension).')

    parser.add_argument('--stl_name', type=str, default="out_100000",
                        help='The name of geometry STL file (without extension).')

    parser.add_argument('--max_video_frames_count', type=int, default=1024000,
                        help='Maximum number of video frames. Frames after this will be ignored.')

    parser.add_argument('--texture_frame_idx_list_step', type=int, default=1,
                        help='Sample one frame from every () frames. As default, use all the frames.')

    parser.add_argument('--input_trajectory_name', type=str, default='traj',
                        help='Input trajectory csv name (default=traj)')

    parser.add_argument('--input_depth_dir_postfix', type=str, default='_base',
                        help='Input depth directory postfix (Default=_base')

    #------------- texture map size params -------------#
    parser.add_argument('--texture_map_max_sizeK', type=float, default=5,
                        help='Maximum width of result texture map, divided by 1000.')

    parser.add_argument('--texture_map_min_triangle_side', type=int, default=7,
                        help='Minimum length of each equilateral triangle in a texture map.')

    #------------- texture map merging params -------------#
    parser.add_argument('--texture_merge_topK', type=int, default=20,
                        help='K used for topK in texture map merging method.')

    parser.add_argument('--texture_merge_topKoutlier_colorthreshold', type=int, default=20,
                        help='Color threshold used for topKoutlier in texture map merging method (in 0~255 scale).')

    parser.add_argument('--texture_merge_topKoutlier_prioritypenalty', type=float, default=2.0,
                        help='Priority penalty used for topKoutlier in texture map merging method.')

    parser.add_argument('--texture_merge_topKoutlier_batchsizeM', type=float, default=2,
                        help='Batch size of topKoutlier merging loop.')

    parser.add_argument('--blend_closest_C_frames', type=int, default=1,
                        help='Blend closest C frames to the target rendering position')

    parser.add_argument('--texture_merge_take_max', type=int, default=0,
                        help='Take max, not weighted sum for texture merge')

    #------------- result rendering params -------------#
    parser.add_argument('--result_rendering_skip', type=int, default=0,
                        help='Do not display the rendering result via OpenGL.')

    parser.add_argument('--result_rendering_trajectory_name', type=str, default="traj_render",
                        help='Rendering trajectory csv name (no dirname, no extension) (default=traj_render)')

    parser.add_argument('--result_rendering_fov', type=int, default=100,
                        help='Result rendering horizontal fov (field of view)')

    parser.add_argument('--result_rendering_h', type=int, default=960,
                        help='Result rendering image height')

    parser.add_argument('--result_rendering_w', type=int, default=1280,
                        help='Result rendering image width')

    parser.add_argument('--result_rendering_farplane', type=float, default=9999.0,
                        help='Result rendering far plane. Further than this will not be rendered.')

    parser.add_argument('--result_rendering_record', type=int, default=0, help='Record the rendering result.')

    parser.add_argument('--result_rendering_record_speed', type=int, default=1,
                        help='Record one frame per this number of frame.')

    parser.add_argument('--result_rendering_record_one_cycle_only', type=int, default=1,
                        help='Record one cycle and quit. (defulat=1)')

    parser.add_argument('--result_rendering_autorotate', type=int, default=0,
                        help='Auto-rotate the rendering result.')

    parser.add_argument('--result_rendering_along_input_traj', type=int, default=0,
                        help='Result rendering trajectory equals to the input frames trajectory')

    parser.add_argument('--cube_side', type=int, default=-1,
                        help='Cube map one side length.')

    parser.add_argument('--result_rendering_interpolation_ratio', type=float, default=0.75,
                        help='Result rendering translation interpolation ratio')

    parser.add_argument('--result_rendering_frame_step', type=int, default=1,
                        help='Render one frame from every () frames. As default, use all the frames.')

    parser.add_argument('--result_rendering_initial_rotation', type=str, default=None,
                        help='Initial rotation for result rendering. 3x3 matrix.')

    #------------- key_frame_num -------------#
    parser.add_argument('--key_frame_num', type=int, default=5,
                        help='Number of key frames for texture mapping')

    args = parser.parse_args(args=custom_args)

    args.result_rendering_initial_rotation = np.eye(3) if args.result_rendering_initial_rotation is None else np.vstack([
                np.fromstring(row.replace('[', '').replace(']', ''), sep=',')
                for row in args.result_rendering_initial_rotation.split('], [')
            ])

    return args

def main(custom_args):
    args = parse_args(custom_args)

    with torch.no_grad():
        texture_main_single_texture_map.main(args)

if __name__ == "__main__":
    main(sys.argv[1:])
