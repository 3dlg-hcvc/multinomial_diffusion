import math

import torch
import torch.nn.functional as F
import torch.nn as nn
from einops import rearrange
from inspect import isfunction
from layers import SegmentationUnet


class ConditionalSegmentationUnet(SegmentationUnet):
    # add conditional input to the Unet
    def __init__(self, num_classes, dim, num_steps, dim_mults=(1, 2, 4, 8), groups=8, dropout=0.):
        super().__init__(num_classes, dim, num_steps)

        self.floor_plan_emb = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, dim, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def forward(self, time, x, floor_plan=None):
        x_shape = x.size()[1:]
        if len(x.size()) == 3:
            x = x.unsqueeze(1)

        B, C, H, W = x.size()
        x = self.embedding(x)

        assert x.shape == (B, C, H, W, self.dim)

        x = x.permute(0, 1, 4, 2, 3)

        assert x.shape == (B, C, self.dim, H, W)

        x = x.reshape(B, C * self.dim, H, W)
        t = self.time_pos_emb(time)
        t = self.mlp(t)

        if floor_plan is not None:
            floor_plan_emb = self.floor_plan_emb(floor_plan)
            t = t + floor_plan_emb

        h = []

        for resnet, resnet2, attn, downsample in self.downs:
            x = resnet(x, t)
            x = self.dropout(x)
            x = resnet2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for resnet, resnet2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, t)
            x = resnet2(x, t)
            x = attn(x)
            x = upsample(x)

        final = self.final_conv(x).view(B, self.num_classes, *x_shape)
        return final
