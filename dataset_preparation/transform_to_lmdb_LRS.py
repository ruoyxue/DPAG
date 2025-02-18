import os
import pickle
import warnings

import cv2
from turbojpeg import TurboJPEG, TJPF_GRAY
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import lmdb
import torch

from tokenizer import SPMTokenizer
from load_cfg import load_config


def load_video(video_path):
    video = []
    cap = cv2.VideoCapture(video_path)
    while(cap.isOpened()):
        ret, frame = cap.read() # BGR
        if ret:
            frame = jpeg.encode(frame)
            video.append(frame)
        else:
            break
    cap.release()
    return video

def decode_video(video):
    first_frame = jpeg.decode(video[0], pixel_format=TJPF_GRAY)
    video_shape = (len(video), *first_frame.shape[:2], 1)
    decoded_video = torch.zeros(video_shape)
    for i in range(0, len(video)):
        decoded_video[i] = torch.Tensor(jpeg.decode(
            video[i], 
            pixel_format=TJPF_GRAY
        ))
    return decoded_video.permute(0, 3, 1, 2)  # T, C, H, W

def text2idstr(text, transform):
    token_id_str = " ".join(
        map(str, [_.item() for _ in transform.word2token(text)])
    )
    return token_id_str


jpeg = TurboJPEG()
class LipDataset(Dataset):
    def __init__(self, split_txt_path, root, dataset_cfg):
        self.root = root
        self.transform = SPMTokenizer('subword_en', dataset_cfg)
        with open(os.path.join(split_txt_path), 'r') as f:
            self.fns = f.read().splitlines()

    def __getitem__(self, idx):
        fn = self.fns[idx].strip()

        if not os.path.exists(os.path.join(self.root, f'{fn}.mp4')) or \
            not os.path.exists(os.path.join(self.root, f'{fn}.txt')) or \
             not os.path.exists(os.path.join(self.root, f'{fn}.pkl')):
            print('file does not exist')
            return -10000
        
        with open(os.path.join(self.root, f"{fn}.txt"), 'r') as f_txt:
            txt_line = f_txt.readline()[5:].strip()
            if len(txt_line) < 1:  # at least a word 
                return -10000
        
        with open(os.path.join(self.root, f"{fn}.pkl"), 'rb') as f_pkl:
            landmark = pickle.load(f_pkl)

        video = load_video(os.path.join(self.root, f'{fn}.mp4'))        
        length = len(video)
        if length < 1:
            return -10000        
        
        subword_token_string = text2idstr(txt_line, self.transform)
        result = {}
        result['video'] = video
        result['length'] = len(video)
        result['token'] = subword_token_string
        result['landmark'] = landmark
        result['file_name'] = fn

        return pickle.dumps(result)

    def __len__(self):
        return len(self.fns)


if __name__ == '__main__':
    split_txt_path = './lrs2/test.txt'
    root = './lrs2-re176/val'
    save_path = './lmdb/lrs2-re176/val'
    dataset_cfg = load_config('./config/dataset.yaml')
    single_data_bytes = 130 * 176 * 176  # need to be as smaller as possible to save spaces
    convert_to_gray = False  # if save grayscale image instead of RGB
    if os.path.exists(save_path):
        warnings.warn("\n\n\nYou have to delete the previous lmdb data\n\n")
    os.makedirs(save_path, exist_ok=True)

    dataset = LipDataset(split_txt_path, root, dataset_cfg)
    env = lmdb.open(os.path.join(save_path, 'tem'), map_size=int(len(dataset) * single_data_bytes), metasync=False, writemap=True)
    txn = env.begin(write=True)
    loader = DataLoader(dataset,
        batch_size = 1,
        num_workers = 32,
        shuffle = False,
        drop_last = False
    )

    length_list = []
    mapsize = 0
    count = 0
    print("count mapsize")
    with tqdm(total=len(dataset), ncols=80) as pbar:
        for _, batch in enumerate(loader):
            for info in batch:
                if info != -10000:
                    txn.put(str(count).encode(), info)
                    mapsize += len(info)
                    info_dict = pickle.loads(info)
                    length_list.append(info_dict['length'])
                    count += 1
            pbar.update()

    txn.commit()
    env.close()
    
    with open(os.path.join(save_path, "length.pkl"), 'wb') as f:
        pickle.dump(length_list, f)

    save_env = lmdb.open(save_path, map_size=int(1.1*mapsize), metasync=False, writemap=True)
    save_txn = save_env.begin(write=True)
    env = lmdb.open(os.path.join(save_path, 'tem'), map_size=int(len(dataset) * single_data_bytes), metasync=False, writemap=True)
    txn = env.begin(write=True)
    total_length = txn.stat()['entries']

    print("store into lmdb")
    with tqdm(total=total_length, ncols=100) as pbar:
        for idx in range(total_length):
            info_bytes = txn.get(str(idx).encode())
            save_txn.put(str(idx).encode(), info_bytes)
            pbar.update()
    
    save_txn.commit()
    save_env.close()
    env.close()

    os.remove(os.path.join(save_path, 'tem', 'data.mdb'))
    os.remove(os.path.join(save_path, 'tem', 'lock.mdb'))
    os.rmdir(os.path.join(save_path, 'tem'))
