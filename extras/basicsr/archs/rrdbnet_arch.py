import torch
from torch import nn as nn
from torch.nn import functional as F

from extras.basicsr.utils.registry import ARCH_REGISTRY
from .arch_util import default_init_weights, make_layer, pixel_unshuffle


class ResidualDenseBlock(nn.Module):
    """Residual Dense Block.

    Used in RRDB block in ESRGAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for each growth.
    """

    def __init__(self, num_feat=64, num_grow_ch=32):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        default_init_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        # Empirically, we use 0.2 to scale the residual for better performance
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual in Residual Dense Block.

    Used in RRDB-Net in ESRGAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for each growth.
    """

    def __init__(self, num_feat, num_grow_ch=32):
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        # Empirically, we use 0.2 to scale the residual for better performance
        return out * 0.2 + x


@ARCH_REGISTRY.register()
class RRDBNet(nn.Module):
    """Networks consisting of Residual in Residual Dense Block, which is used
    in ESRGAN.

    ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks.

    We extend ESRGAN for scale x2 and scale x1.
    Note: This is one option for scale 1, scale 2 in RRDBNet.
    We first employ the pixel-unshuffle (an inverse operation of pixelshuffle to reduce the spatial size
    and enlarge the channel size before feeding inputs into the main ESRGAN architecture.

    Args:
        num_in_ch (int): Channel number of inputs.
        num_out_ch (int): Channel number of outputs.
        num_feat (int): Channel number of intermediate features.
            Default: 64
        num_block (int): Block number in the trunk network. Defaults: 23
        num_grow_ch (int): Channels for each growth. Default: 32.
        is_real_esrgan (bool): Whether to use the Real-ESRGAN mode. Default is True.
                               If True, adjust num_in_ch based on scale.
                               If False, adjust conv_up2 based on scale.
    """

    def __init__(self, num_in_ch, num_out_ch, scale=4, num_feat=64, num_block=23, num_grow_ch=32, is_real_esrgan=True):
        super(RRDBNet, self).__init__()
        self.scale = scale
        self.num_block = num_block
        self.is_real_esrgan = is_real_esrgan
        if is_real_esrgan:
            if scale == 2:
                num_in_ch = num_in_ch * 4
            elif scale == 1:
                num_in_ch = num_in_ch * 16
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(RRDB, num_block, num_feat=num_feat, num_grow_ch=num_grow_ch)
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        # upsample
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1) if is_real_esrgan or scale == 4 else None
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def load_state_dict(self, state_dict, strict=True):
        if "conv_first" in state_dict:
            return super().load_state_dict(state_dict, strict=strict)
            
        loadnet = {}
        for key, value in state_dict.items():
            new_key = key \
                .replace(f"model.1.sub.{self.num_block}.", "conv_body.") \
                .replace("model.0"    , "conv_first") \
                .replace("model.1.sub", "body") \
                .replace(".0.weight"  , ".weight") \
                .replace(".0.bias"    , ".bias") \
                .replace(".RDB1."     , ".rdb1.") \
                .replace(".RDB2."     , ".rdb2.") \
                .replace(".RDB3."     , ".rdb3.")
            if self.num_block == 23:
                new_key = new_key \
                .replace("model.3."   , "conv_up1.") \
                .replace("model.6."   , "conv_up2.") \
                .replace("model.8."   , "conv_hr.") \
                .replace("model.10."  , "conv_last.")
            else:
                new_key = new_key \
                .replace("model.3."   , "conv_up1.") \
                .replace("model.5."   , "conv_hr.") \
                .replace("model.7."   , "conv_last.") \
                .replace("model.8."   , "conv_last.")
            loadnet[new_key] = value
        super().load_state_dict(loadnet, strict=strict)

    def forward(self, x):
        if self.is_real_esrgan and self.scale != 4:
            if self.scale == 2:
                feat = pixel_unshuffle(x, scale=2)
            elif self.scale == 1:
                feat = pixel_unshuffle(x, scale=4)
        else:
            feat = x
        feat = self.conv_first(feat)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        # upsample
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        if self.conv_up2:
            feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out
