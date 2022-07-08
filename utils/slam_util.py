import os
import subprocess
import cv2
import numpy as np

def select_mask(config):

    video_path = config['data_path']['data_root_path'] + "/" + config['data_path']['video_root_path'] + "/" + config['data_path']['video_name'] + "/"
    video = cv2.VideoCapture(video_path + "video.mp4")
    depth_scale = config['depth_estimation']['depth_scale']

    mask_img_path = video_path + "mask_img.png"
    mask_depth_path = video_path + "mask_depth.png"

    success, frame = video.read()
    h, w, c = frame.shape
    scale = 0.2
    scaled_h = int(h * scale)
    scaled_w = int(w * scale)


    depth_h = int(h * depth_scale)
    depth_w = int(w * depth_scale)

    print("Mask img size: {} x {}".format(w,h))
    print("Mask depth size: {} x {}".format(depth_w, depth_h))

    mask_list = []

    def draw_rectangle(event, x, y, flags, param):
        global x1, y1

        if event == cv2.EVENT_LBUTTONDOWN:
            x1, y1 = x, y

        elif event == cv2.EVENT_LBUTTONUP:

            if x > scaled_w-1:
                x = scaled_w-1
            if y > scaled_h-1:
                y = scaled_h-1
            if x < 0:
                x = 0
            if y < 0:
                y = 0
            if x1 <= x and y1 <= y:
                pass
            elif x1 <= x and y1 > y:
                temp = y
                y = y1
                y1 = temp
            elif x1 > x and y1 <= y:
                temp = x
                x = x1
                x1 = temp
            else:
                temp = y
                y = y1
                y1 = temp
                temp = x
                x = x1
                x1 = temp

            mask_list.append(((x1,y1),(x,y)))

    cv2.namedWindow('video')
    cv2.setWindowProperty('video', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback('video', draw_rectangle)

    x1, y1 = -1, -1

    while success:
        frame = cv2.resize(frame, (scaled_w, scaled_h), cv2.INTER_LINEAR)
        for box in mask_list:
            cv2.rectangle(frame, box[0], box[1], (255, 0, 0), -1)
        cv2.imshow("video", frame)
        key = cv2.waitKey(1)
        if key == ord('s'):
            for box in mask_list:
                print(box)
            mask_list = [((p1 / (scaled_w-1), p2 / (scaled_h-1)), (p3 / (scaled_w-1), p4 / (scaled_h-1))) for ((p1, p2), (p3, p4)) in mask_list]
            mask_img = np.ones((h,w))
            mask_depth = np.ones((depth_h, depth_w))

            print("Feature.mask_rectangles:")
            for box in mask_list:
                mask_img[int(box[0][1] * (h-1)):int(box[1][1] * (h-1))+1,int(box[0][0] * (w-1)):int(box[1][0] * (w-1))+1] = 0
                mask_depth[int(box[0][1] * (depth_h-1)):int(box[1][1] * (depth_h-1))+1,int(box[0][0] * (depth_w-1)):int(box[1][0] * (depth_w-1))+1] = 0
                print([box[0][0], box[1][0], box[0][1],box[1][1]])
            cv2.imwrite(mask_img_path, mask_img * 255)
            cv2.imwrite(mask_depth_path, mask_depth * 255)
            cv2.destroyAllWindows()
            return
        success, frame = video.read()
        if not success:
            video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            success, frame = video.read()


def run_two_pass_slam(config):
    video_path = config['data_path']['data_root_path'] + "/" + config['data_path']['video_root_path'] + "/" + config['data_path']['video_name'] + "/"
    openvslam_path = config['slam_path']['openvslam_path']
    orb_path = config['slam_path']['orb_path']
    traj_path = video_path + "traj.csv"
    map_path = video_path + "map.msg"
    mask_name = "mask_img.png"

    ## first pass - create a map
    if not os.path.isfile(map_path):
        openvslam_script = "%s -v %s -o %s -p %s -t %s -m %s -n 0"%(openvslam_path, video_path, orb_path, map_path, traj_path, mask_name)
        subprocess.check_output(openvslam_script, shell=True, universal_newlines=True)
    ## second pass - save trajectory
    if not os.path.isfile(traj_path):
        openvslam_script = "%s -v %s -o %s -p %s -t %s -m %s -n 1" % (openvslam_path, video_path, orb_path, map_path, traj_path, mask_name)
        subprocess.check_output(openvslam_script, shell=True, universal_newlines=True)