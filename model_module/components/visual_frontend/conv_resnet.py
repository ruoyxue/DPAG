#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2021 Imperial College London (Pingchuan Ma)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
import torch
import torch.nn as nn
from .resnet import BasicBlock, ResNet


def threeD_to_2D_tensor(x):
    n_batch, n_channels, s_time, sx, sy = x.shape
    x = x.transpose(1, 2)
    return x.reshape(n_batch * s_time, n_channels, sx, sy)


class Swish(nn.Module):
    """Construct an Swish object."""

    def forward(self, x):
        """Return Swich activation function."""
        return x * torch.sigmoid(x)


class Conv3dResNet(torch.nn.Module):
    """Conv3dResNet module"""

    def __init__(self, in_ch, relu_type="swish"):
        """__init__.

        :param backbone_type: str, the type of a visual front-end.
        :param relu_type: str, activation function used in an audio front-end.
        """
        super(Conv3dResNet, self).__init__()
        self.frontend_nout = 64
        self.frontend3D = nn.Sequential(
            nn.Conv3d(
                in_ch, self.frontend_nout, (5, 7, 7), (1, 2, 2), (2, 3, 3), bias=False
            ),
            nn.BatchNorm3d(self.frontend_nout),
            Swish(),
            nn.MaxPool3d((1, 3, 3), (1, 2, 2), (0, 1, 1)),
        )
        
        self.trunk = ResNet(BasicBlock, [2, 2, 2, 2], relu_type=relu_type)

    def forward(self, xs_pad, flatten=True):
        xs_pad = xs_pad.transpose(1, 2)  # [B, T, C, H, W] -> [B, C, T, H, W]

        B, C, T, H, W = xs_pad.size()
        xs_pad = self.frontend3D(xs_pad)  #  [B, 64, T, H // 4, W // 4]
        Tnew = xs_pad.shape[2]
        xs_pad = threeD_to_2D_tensor(xs_pad)  # [B * T, 64, H // 4, W // 4]
        xs_pad = self.trunk(xs_pad, flatten)
        if not flatten:
            return xs_pad.view(B, Tnew, *xs_pad.shape[1:]), xs_pad.shape[1]
        xs_pad = xs_pad.view(B, Tnew, xs_pad.size(1)).contiguous()
        return xs_pad, xs_pad.shape[1]


class Conv3dResNet34_ds16(Conv3dResNet):
    """Conv3dResNet module downsample rate = 16"""

    def __init__(self, in_ch, relu_type="swish"):
        super().__init__(in_ch, relu_type)
        self.trunk = ResNet(BasicBlock, [3, 4, 6], relu_type=relu_type)
