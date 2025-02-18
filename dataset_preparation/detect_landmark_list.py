import torch
import cv2
import torch.version
from detector import LandmarksDetector, PureFeatureDetector
from video_process import VideoProcess
import numpy as np
import math
from tqdm import tqdm
import argparse
import os
from shutil import copy2
from multiprocessing import Process
import imageio
import pickle


def center_crop(video, crop_size):
    _, h, w, _ = video.shape
    assert crop_size[0] % 2 == 0 and crop_size[1] % 2 == 0

    if h < crop_size[0] or w < crop_size[1]:
        raise ValueError('Error crop size')
    center_h, center_w = h // 2, w // 2
    half_crop_h = crop_size[0] // 2
    half_crop_w = crop_size[1] // 2
    video = video[:, center_h - half_crop_h: center_h + half_crop_h, center_w - half_crop_w: center_w + half_crop_w, :]
    return video


def load_video(data_filename):
    video = imageio.get_reader(data_filename, "ffmpeg")
    first_frame = video.get_data(0)

    height, width = first_frame.shape[:2]

    num_frames = video.count_frames()  # 假设可以获取总帧数
    frames = np.empty((num_frames, height, width, 3), dtype=np.uint8)

    for i, frame in enumerate(video):
        frames[i] = frame  # 将帧数据直接赋值到预分配的数组中

    return frames  # (T, H, W, C)


def extract_lip(landmarks_detector, five_lm_detector, video_process, video, vfn, src_dir, txt_dir, dst_dir):
    landmarks = landmarks_detector(video)
    video = video_process(video, landmarks)
    os.makedirs(dst_dir, exist_ok=True)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 指定编码格式
    
    out = cv2.VideoWriter(f"{dst_dir}/{vfn}.mp4", fourcc, 25, (video.shape[2], video.shape[1]))

    for frame in video:  # 假设 frames 是帧的列表
        out.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))  # 转换为 BGR 格式写入

    out.release()

    copy2(f"{txt_dir}/{vfn}.txt", f"{dst_dir}/{vfn}.txt")
    if five_lm_detector is not None:
        five_landmarks = five_lm_detector(video)
        with open(f"{dst_dir}/{vfn}.pkl", 'wb') as f:
            pickle.dump(five_landmarks, f)


class ExtractLipDataset(torch.utils.data.Dataset):
    def __init__(self, filelist):
        self.filelist = filelist
    
    def __len__(self):
        return len(self.filelist)
    
    def __getitem__(self, idx):
        vfn, src_dir, txt_dir, dst_dir = self.filelist[idx]
        if os.path.exists(f"{dst_dir}/{vfn}.mp4"):
            return -10000
        try:
            video = load_video(f"{src_dir}/{vfn}.mp4")
        except:
            print('load video failed')
            return -10000
        return video, vfn, src_dir, txt_dir, dst_dir


def main(filelist, rank, shard, process_per_gpu, num_workers, region, batch_size):
    gpu = rank // process_per_gpu
    landmarks_detector = LandmarksDetector(device=f"cuda:{gpu}", batch_size=batch_size)
    
    five_lm_detector = None
    
    if region == 'lip':
        crop = 96
        resize = (96, 96)
    elif region == 'face':
        crop = 176
        resize = (176, 176)
        five_lm_detector = PureFeatureDetector(device=f"cuda:{gpu}", batch_size=batch_size)
    else:
        raise ValueError("Unknown region")

    video_process = VideoProcess(convert_gray=False, window_margin=1, region=region, crop_width=crop, crop_height=crop, resize=resize)
    psize = math.ceil(len(filelist) / shard)
    filelist = filelist[rank*psize: (rank+1)*psize]
    pbar = tqdm(total=len(filelist), disable=rank!=0, ncols=70)
    pbar.set_description(f'rank {rank}')
    failed = []
    dataloader = torch.utils.data.DataLoader(ExtractLipDataset(filelist), num_workers=2)

    for _, info in enumerate(dataloader):
        if info != -10000:
            video, vfn, src_dir, txt_dir, dst_dir = info
            video = video.squeeze(0).numpy()
            vfn = vfn[0]
            src_dir = src_dir[0]
            txt_dir = txt_dir[0]
            dst_dir = dst_dir[0]
            try:
                extract_lip(landmarks_detector, five_lm_detector, video_process, video, vfn, src_dir, txt_dir, dst_dir)
            except Exception as e:
                print(f"{vfn} failed")
                failed.append(f"{src_dir}  {vfn}")
        pbar.update()

    print(f"Rank {rank} finish {len(filelist)} video file")
    with open(f"failed.txt", 'a') as fp:
        fp.write('\n'.join(failed))

parser = argparse.ArgumentParser()
parser.add_argument('--list', required=True, help='path contains source video files')
parser.add_argument('--shard', default=2, type=int, help='size of multiprocessing pool')
parser.add_argument('--process_per_gpu', default=1, type=int)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--region', default='lip', type=str)
parser.add_argument('--num_workers', default=2, type=int)

if __name__ == '__main__':
    args = parser.parse_args()
    fl = args.list
    shard = args.shard
    region = args.region
    batch_size = args.batch_size
    filelist = []
    with open(fl) as fp:
        for line in fp.readlines():
            vfn, src_dir, txt_dir, dst_dir = line.strip().split('\t')
            filelist.append((vfn, src_dir, txt_dir, dst_dir))

    procs = []

    for rank in range(shard):
        proc = Process(target=main, args=(filelist, rank, shard, args.process_per_gpu, args.num_workers, region, batch_size))
        procs.append(proc)
        proc.start()

    for proc in procs:
        proc.join()
