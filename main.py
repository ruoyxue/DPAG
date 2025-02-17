import os
import argparse

import torch
from data_module.data_module import DataModule
from model_module.model_module import ModelModule
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import TensorBoardLogger
from util import load_config


def load_args():
    """ Load arguments """
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='test', choices=['test'], help='mode')
    parser.add_argument('--cfg', type=str, help='the config file used to set training process')
    parser.add_argument('--ckpt', type=str, default='', help='the checkpoint file for loading')
    parser.add_argument('--log', type=str, default='tem', help='add log info to the name of save directory')
    args = parser.parse_args()
    return args

def main():
    os.environ["WANDB_SILENT"] = "true"
    torch.set_float32_matmul_precision('high')
    args = load_args()
    cfg = load_config(args.cfg)

    cfg.exp_path = os.path.join(cfg.exp_dir, f"{args.log}")
    cfg.gpus = torch.cuda.device_count()
    cfg.ckpt_path = args.ckpt
    cfg.mode = args.mode

    # Configure logger
    logger = TensorBoardLogger(save_dir=os.path.join(cfg.exp_path, 'tblog'))

    # Set modules and trainer
    modelmodule = ModelModule(cfg, ckpt=cfg.ckpt_path)
    datamodule = DataModule(cfg)
    trainer = Trainer(**cfg.trainer)
    seed_everything(cfg.seed + trainer.local_rank, workers=True)
    os.makedirs(os.path.join(cfg.exp_path, 'tblog', 'lightning_logs'), exist_ok=True)
    trainer.test(model=modelmodule, datamodule=datamodule, ckpt_path=cfg.ckpt_path)


if __name__ == "__main__":
    main()
