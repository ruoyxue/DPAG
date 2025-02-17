import torch
import torch._dynamo.config
import torchaudio
from tokenizer import get_tokenizer
import os
import json

from model_module.dpag import DPAG
from pytorch_lightning import LightningModule
from util import load_config
from data_module.transforms import VideoTransform


def compute_char_level_distance(seq1, seq2):
    return torchaudio.functional.edit_distance(seq1, seq2)

def compute_word_level_distance(seq1, seq2):
    return torchaudio.functional.edit_distance(
        seq1.lower().split(), seq2.lower().split()
    )

class ModelModule(LightningModule):
    def __init__(self, cfg, ckpt=None):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg
        self.dataset_cfg = load_config(cfg.data.dataset_cfg)
        
        self.transform = get_tokenizer(self.cfg.data.tokenizer, self.dataset_cfg)
        self.token_list = self.transform.token_list
        
        self.model_cfg = load_config(cfg['model_cfg'])
        self.model = self.model_select(self.model_cfg)
        
        if ckpt is not None:
            self.load_ckpt(ckpt)

        self.test_aug = VideoTransform("test", self.dataset_cfg.name, self.cfg.data.crop_size)

    def load_ckpt(self, ckpt):
        model_checkpoint = torch.load(ckpt, map_location="cpu")['state_dict']
        model_ckpt = {k.replace("model.", "", 1): v for k, v in model_checkpoint.items()}
        self.model.load_state_dict(model_ckpt)

    def model_select(self, model_cfg):
        return DPAG(
            in_ch=1 if self.cfg.data.convert_to_gray else 3,
            token_list=self.token_list,
            model_cfg=model_cfg,
            decode_cfg=self.dataset_cfg.decode,
            transform=self.transform
        )

    def test_step(self, sample, sample_idx):
        augmented_data = self.test_aug({
            "video": sample['video'].cuda(),
            "landmark": sample['landmark']
        })
        
        sample['video'] = augmented_data['video']
        sample['landmark'] = augmented_data['landmark']
         
        fn = sample['fn']
        if not fn in self.tested.keys():
            test_result = self.model.test_step(sample)
            pred = test_result['pred'].strip()
        else:
            pred = self.tested[fn]['pred'].strip()

        gt = self.transform.token2word(sample["tokenizer_token"]).strip()
        dist = compute_word_level_distance(gt, pred)
        self.distance += dist
        self.length += len(gt.lower().split())
        wer = dist / len(gt.lower().split())
        
        test_item = {
            "fn": sample["fn"],
            "gt": gt,
            "pred": pred,
            "wer": wer,
        }

        if not fn in self.tested.keys():
            with open(os.path.join(self.cfg.exp_path, "test_disorder.txt"), "a") as f:
                for key, value in test_item.items():
                    f.write(f"{key}: {value}\n")
                f.write('\n')

        self.test_results.append(test_item)
        return

    def on_test_epoch_start(self):
        self.distance = 0
        self.length = 0
        self.test_results = []
        self.tested = {}
        if os.path.exists(os.path.join(self.cfg.exp_path, "test_disorder.txt")):
            with open(os.path.join(self.cfg.exp_path, "test_disorder.txt")) as f:
                result_list = f.read().split('\n\n')

            for result in result_list:
                info_list = result.split('\n')
                fn = info_list[0].split(': ')[-1]
                tem_dict = {}
                for info in info_list:
                    if len(info) > 0:
                        key = info.split(': ')[0]
                        value = info.split(': ')[1]
                        tem_dict[key] = value.strip()
                self.tested[fn] = tem_dict

    def on_test_epoch_end(self):
        self.test_results = sorted(self.test_results, key=lambda x: x['wer'])
        with open(os.path.join(self.cfg.exp_path, "test_order.json"), "w", encoding="utf-8") as f:
            json.dump(self.test_results, f, indent=4, ensure_ascii=False)
        del self.test_results
        wer = self.distance / self.length
    
        self.log("wer", wer, sync_dist=True)
        with open(os.path.join(self.cfg.exp_path, "test_result.txt"), "a") as f:
            f.write(f"wer: {wer}\n")
    