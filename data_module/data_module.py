import torch
from pytorch_lightning import LightningDataModule

from .av_dataset import AVDataset
from .transforms import VideoTransform
from util import load_config


def pad(samples, pad_val=0.0):
    lengths = [len(s) for s in samples]
    max_size = max(lengths)
    sample_shape = list(samples[0].shape[1:])
    collated_batch = samples[0].new_zeros([len(samples), max_size] + sample_shape)
    for i, sample in enumerate(samples):
        diff = len(sample) - max_size
        if diff == 0:
            collated_batch[i] = sample
        else:
            collated_batch[i] = torch.cat(
                [sample, sample.new_full([-diff] + sample_shape, pad_val)]
            )
    if len(samples[0].shape) == 1:
        collated_batch = collated_batch.unsqueeze(1)  # targets
    elif len(samples[0].shape) == 2:
        pass  # collated_batch: [B, T, 1]
    elif len(samples[0].shape) == 4:
        pass  # collated_batch: [B, T, C, H, W]
    return collated_batch, lengths

def collate_pad(batch):
    batch_out = {}
    candidate_keys = list(batch[0].keys())
    candidate_keys.remove('fn')
    
    for data_type in candidate_keys:
        pad_val = -1 if data_type in ["tokenizer_token", "landmark"] else 0.0
        c_batch, sample_lengths = pad(
            [s[data_type] for s in batch if s[data_type] is not None], pad_val
        )
        batch_out[data_type] = c_batch
        batch_out[data_type + "_length"] = torch.tensor(sample_lengths)
        
    return batch_out


class DataModule(LightningDataModule):
    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg
        self.dataset_cfg = load_config(self.cfg.data.dataset_cfg)
        self.cfg.gpus = torch.cuda.device_count()

    def test_dataloader(self):
        test_ds = AVDataset(
            dataset_path=self.dataset_cfg.test,
            subset="test",
            transform=VideoTransform("test", self.dataset_cfg.name, self.cfg.data.crop_size),
            tokenizer=self.cfg.data.tokenizer,
            max_frame=self.cfg.data.max_frames,
            max_length=self.cfg.data.test_max_length,
            convert_to_gray=self.cfg.data.convert_to_gray
        )

        dataloader = torch.utils.data.DataLoader(test_ds, num_workers=self.cfg.num_workers, pin_memory=True, batch_size=None)
        return dataloader
    