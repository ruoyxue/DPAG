import os
import pickle
import numpy as np

import torch
from turbojpeg import TurboJPEG, TJPF_GRAY, TJPF_BGR
import lmdb


jpeg = TurboJPEG()
class AVDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_path,
        subset,
        transform,
        tokenizer="subword_en",
        convert_to_gray=True,
        max_frame=1800,
        max_length=750
    ):
        self.subset = subset
        self.max_frame = max_frame
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.dataset_path = dataset_path
        self.convert_to_gray = convert_to_gray    
        self.transform = transform
        self.length_list = []
        self.idx_list_for_length = []
        self.data_list = []
        self.filter_data_by_length()
        self.open_lmdb()

    def open_lmdb(self):
        if not hasattr(self, 'txn'):
            self.env = lmdb.Environment(self.dataset_path, readonly=True, lock=False)
            self.txn = self.env.begin()

    def filter_data_by_length(self):
        with open(os.path.join(self.dataset_path, "length.pkl"), 'rb') as f:
            all_length_list = pickle.load(f)

        for idx in range(len(all_length_list)):
            length = all_length_list[idx]

            if length <= self.max_length and length <= self.max_frame:
                self.length_list.append(length)
                self.idx_list_for_length.append(idx)

    def decode_video(self, video):
        first_frame = jpeg.decode(video[0], pixel_format=TJPF_GRAY if self.convert_to_gray else TJPF_BGR)
        channel = 1 if self.convert_to_gray else 3
        video_shape = (len(video), *first_frame.shape[:2], channel)
        decoded_video = torch.zeros(video_shape, dtype=torch.int8)
        for i in range(0, len(video)):
            decoded_video[i] = torch.tensor(jpeg.decode(
                video[i], 
                pixel_format=TJPF_GRAY if self.convert_to_gray else TJPF_BGR
            ))
        return decoded_video.permute(0, 3, 1, 2)  # T, C, H, W

    def __getitem__(self, idx):
        true_idx = self.idx_list_for_length[idx]
        info_dict_bytes = self.txn.get(str(true_idx).encode())
        info_dict = pickle.loads(info_dict_bytes)

        if self.tokenizer in ['word', 'subword_en', 'subword_es', 'subword_fr']:
            tokenizer_token = torch.tensor([int(_) for _ in info_dict['token'].split()])

        file_name = info_dict['file_name']
        landmark = info_dict['landmark']
        landmark = torch.tensor(np.array(landmark), dtype=torch.float)
        
        video = self.decode_video(info_dict['video'])  # T, C, H, W
        
        return {
            "video": video, 
            "tokenizer_token": tokenizer_token,
            "fn": file_name,
            "landmark": landmark
        }

    def __len__(self):
        return len(self.length_list)
