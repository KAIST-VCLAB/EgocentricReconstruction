import cv2
import numpy as np
import cupy as cp
import yaml
import csv

def sample_rt(video_frame_rt, sample_ratio):
    list = []
    for i in range(len(video_frame_rt)):
        if i%sample_ratio == 0:
            list.append(video_frame_rt[i])
    return list

def get_video_rt(video_frame_rt):
    video_rt = []
    for i in video_frame_rt:
        video_rt.append(i[1])
    return np.asarray(video_rt).astype(np.float32)

def get_all_frames(video, resize_scale, sample_ratio):

    frames = []

    success, frame = video.read()
    j = 0
    while success:
        if j%sample_ratio == 0:
            h, w, c = frame.shape
            frame = cv2.resize(frame, (int(w * resize_scale), int(h * resize_scale)), cv2.INTER_LINEAR)
            frame = frame[:, :, ::-1].copy()
            frames.append(frame)
        j = j + 1
        success, frame = video.read()

    return cp.asarray(frames)

def get_all_frames_numpy(video, resize_scale, sample_ratio):

    frames = []

    success, frame = video.read()
    j = 0
    while success:
        if j%sample_ratio == 0:
            h, w, c = frame.shape
            frame = cv2.resize(frame, (int(w * resize_scale), int(h * resize_scale)), cv2.INTER_LINEAR)
            frame = frame[:, :, ::-1].copy()
            frames.append(frame)
        j = j + 1
        success, frame = video.read()

    return np.asarray(frames)

def get_all_depths(depth_path, length):

    depth_list = []
    for i in range(length):
        depth = cv2.imread(f"{depth_path}/{i}.exr", -1)
        depth_list.append(depth)

    return cp.asarray(depth_list)


def visualize_frames(video_frames):
    for i in video_frames:
        cv2.imshow("hi", i)
        cv2.waitKey(1)


def get_second_neighbors(first_idx, frame_idx_list, neighbor_interval, max_size, neighbor_size, video_frame_rt, min_baseline, max_baseline):
    translation_list = []
    return_list = []

    for i in range(max_size):
        ## from closer frames to farther frames
        j = (i // 2) * ((-1) ** i)

        second_idx = first_idx + j * neighbor_interval
        if (second_idx < 0) or (second_idx >= len(frame_idx_list)):
            continue

        ref_rt = video_frame_rt[first_idx][1]
        warp_rt = video_frame_rt[second_idx][1]

        rt = np.matmul(ref_rt, np.linalg.inv(warp_rt))
        t = rt[0:3, -1]

        if np.linalg.norm(t) < min_baseline:
            continue

        if np.linalg.norm(t) > max_baseline:
            continue

        select = 1

        if select == 1:
            translation_list.append(t)
            return_list.append(second_idx)

        if len(translation_list) == neighbor_size:
            break
    return return_list


def get_frame_idx_list(video_rt):
    length = len(video_rt)
    return list(range(int(length)))

def get_frame_num(video_rt, idx):
    return video_rt[idx][1]

def save_yaml(string, file_name):
    rt_list = []
    split_list = string.split("\n")

    length = len(split_list) - 1
    assert length % 5 == 0
    rt_num = int(length / 5)

    with open(file_name, 'w') as file:
        for i in range(rt_num):


            frame_num = int(split_list[5 * i])
            rt_row_0 = split_list[5 * i + 1].split()
            rt_row_1 = split_list[5 * i + 2].split()
            rt_row_2 = split_list[5 * i + 3].split()
            rt_row_3 = split_list[5 * i + 4].split()

            yam = [frame_num, [rt_row_0, rt_row_1, rt_row_2, rt_row_3]]

            yaml.dump(yam, file)
            rt = np.array([rt_row_0, rt_row_1, rt_row_2, rt_row_3], dtype=np.float32)

            rt_list.append([frame_num, rt])

    return rt_list

def read_yaml(file_name):
    rt_list = []
    try:
        with open(file_name) as file:
            data = yaml.load(file, Loader=yaml.FullLoader)
            num = int(len(data) / 2)
            for i in range(num):
                frame_num = int(data[2 * i])
                rt = np.array(data[2 * i + 1], dtype=np.float32)
                rt_list.append([frame_num, rt])
        return rt_list
    except IOError:
        return -1

def read_csv(file_name):
    rt_list = []
    f = open(file_name, 'r')
    file = csv.reader(f)
    for line in file:
        line = line[0].split(" ")
        rt_list.append([int(line[0]), np.array([[line[1],line[2],line[3],line[4]],[line[5],line[6],line[7],line[8]],[line[9],line[10],line[11],line[12]],[line[13],line[14],line[15],line[16]]], dtype=np.float32)])

    return rt_list


def read_frame(video, frame_num):
    video.set(1, frame_num)
    success, frame = video.read()
    return frame

