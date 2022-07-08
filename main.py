import argparse
import json
from utils import *

def main():

    parser = argparse.ArgumentParser()
    ## Read config file
    parser.add_argument('--config', type=str, default="demo/config/demo.json")
    args = parser.parse_args()
    config = json.load(open(args.config, "r"))

    ## Mask photographer
    if '0' in config['step']:
        print("Select mask")
        print("Press \"S\" when you are done")
        slam_util.select_mask(config)

    ## Run SLAM to save camera trajectory
    if '1' in config['step']:
        print("Run 2-Pass OpenVSLAM...")
        # If it has a mask image, mask_rectangles in config.yaml will be ignored
        slam_util.run_two_pass_slam(config)

    ## Estimate per-frame depth
    if '2' in config['step']:
        print("Depth estimation")
        depth_util.run_depth(config)

    ## Calculate weight
    if '3' in config['step']:
        print("Weight calculation")
        weight_util.run_weight(config)

    ## Generate mesh
    if '4' in config['step']:
        print("Mesh Generation")
        dmc_util.run_dmc(config)

    ## Texture mapping
    if '5' in config['step']:
        print("Texture mapping")
        texture_util.run_texture(config)

if __name__ == "__main__":
    main()