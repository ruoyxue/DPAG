import torch

from .conv_resnet import Conv3dResNet34_ds16


def get_visual_frontend(in_ch, frontend_cfg):
    if frontend_cfg.backbone == 'conv3d-resnet34-ds16':
        return Conv3dResNet34_ds16(
            in_ch=in_ch,
            relu_type=frontend_cfg.relu_type
        )
    else:
        raise ValueError('Unknown visual front-end')
