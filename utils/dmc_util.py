import os
import subprocess

def run_dmc(config):

    if not os.path.exists(config['data_path']['data_root_path'] + "/" +  config['data_path']['mesh_root_path'] + "/" + config['data_path']['video_name']):
        os.makedirs(config['data_path']['data_root_path'] + "/" + config['data_path']['mesh_root_path'] + "/" + config['data_path']['video_name'], exist_ok=True)
    subprocess.run('CUDA_VISIBLE_DEVICES=0 ./dmc/build/dmc {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}'.format(config['data_path']['video_name'], config['mesh_generation']['solid_angle'], config['mesh_generation']['frame_interval'], config['mesh_generation']['min_depth'], config['mesh_generation']['max_depth'], config['mesh_generation']['truncation'], config['data_path']['data_root_path'], config['data_path']['video_root_path'],config['data_path']['depth_root_path'], config['data_path']['mesh_root_path'], config['mesh_generation']['min_sdf_cnt'], config['mesh_generation']['node_sample_interval'], config['mesh_generation']['sky_depth_threshold'], config['mesh_generation']['weight_trunc_std'], config['mesh_generation']['truncation_offset'], config['mesh_generation']['truncation_change']), shell=True)
