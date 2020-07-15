# based on fpn
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init, normal_init

from ..registry import HEADS
from ..builder import build_loss
from mmdet.ops import ConvModule

import torch
import numpy as np


@HEADS.register_module
class MaskFeatHead(nn.Module):
    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels,
                 num_layers,
                 start_level, # 0 means P3 of FPN
                 end_level,
                 num_classes,
                 strides=[8, 16, 32, 64, 128],
                 semantic_loss_on=False,
                 conv_cfg=None,
                 norm_cfg=None):
        super(MaskFeatHead, self).__init__()

        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.start_level = start_level
        self.end_level = end_level
        self.num_classes = num_classes
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.strides = strides
        self.semantic_loss_on = semantic_loss_on
        
        # parallel
        self.refine = nn.ModuleList() 
        for _ in range(self.start_level, self.end_level + 1):
            self.refine.append(
                 ConvModule(
                    self.in_channels, # 256
                    self.mid_channels, # 128
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None
            ))
        # together
        self.tower = nn.ModuleList() # together
        for i in range(self.num_layers):
            self.tower.append(
                ConvModule(
                    self.mid_channels,
                    self.mid_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None))
        self.tower.append(nn.Conv2d(self.mid_channels, self.out_channels, 1))


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)

    def forward(self, inputs):
        for i in range(self.start_level, self.end_level + 1):
            cur_index = i-self.start_level
            if i == self.start_level:
                x = self.refine[cur_index](inputs[cur_index])
            else:
                x_p = self.refine[cur_index](inputs[cur_index])
                target_h, target_w = x.size()[2:]
                h, w = x_p.size()[2:]
                factor_h, factor_w = target_h // h, target_w // w
                x_p = aligned_bilinear(x_p, factor_h)
                x = x + x_p
        for tower_layer in self.tower:
            x = tower_layer(x)

        return x


def aligned_bilinear(tensor, factor):
    assert tensor.dim() == 4
    assert factor >= 1
    assert int(factor) == factor

    if factor == 1:
        return tensor

    h, w = tensor.size()[2:]
    tensor = F.pad(tensor, pad=(0, 1, 0, 1), mode="replicate")
    oh = factor * h + 1
    ow = factor * w + 1
    tensor = F.interpolate(
        tensor, size=(oh, ow),
        mode='bilinear',
        align_corners=True
    )
    tensor = F.pad(
        tensor, pad=(factor // 2, 0, factor // 2, 0),
        mode="replicate"
    )

    return tensor[:, :, :oh - 1, :ow - 1]