import torch.nn as nn
from mmcv.cnn import build_norm_layer
from mmseg.registry import MODELS
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
import mmcv
from mmseg.registry import MODELS
from ..utils import resize
from mmcv.cnn import ConvModule
import torch
from mmseg.registry import MODELS
from .decode_head import BaseDecodeHead


def nnstacking(x):
    r_pad = F.pad(x, [1, 1, 1, 1], 'constant', 0)
    all_stacked = torch.concat(
        [r_pad[:, :, 2:, 1:-1],  # bottom
         r_pad[:, :, :-2, 1:-1],  # top
         r_pad[:, :, 1:-1, 2:],  # right
         r_pad[:, :, 1:-1, :-2],  # left
         r_pad[:, :, 1:-1, 1:-1]],  # self
        dim=1)
    return all_stacked


class LongRangeDW(torch.nn.Module):
    def __init__(self, in_channels=128, out_channels=128,
                 l=1,
                 k=3,
                 atrous_rates=[1, 8, 12],
                 reduce_method='avg',
                 activation_method='gelu',
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None):
        super().__init__()
        self.l = l
        self.k = k
        self.atrous_rates = atrous_rates
        self.reduce_method = reduce_method
        self.C = in_channels
        self.activation_method = activation_method
        self.activations = {'relu': torch.nn.ReLU(), 'gelu': torch.nn.GELU()}
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        self.depthwise = nn.ModuleList([nn.ModuleList([
            ConvModule(
                5 * self.C,
                5 * self.C,
                kernel_size=1 if atrous_rates[0] == 1 else 3,
                dilation=atrous_rates[0],
                padding=0 if atrous_rates[0] == 1 else atrous_rates[0],
                groups=5 * self.C,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg),
            ConvModule(
                5 * self.C,
                5 * self.C,
                kernel_size=1 if atrous_rates[1] == 1 else 3,
                dilation=atrous_rates[1],
                padding=0 if atrous_rates[1] == 1 else atrous_rates[1],
                groups=5 * self.C,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg),
            ConvModule(
                5 * self.C,
                5 * self.C,
                kernel_size=1 if atrous_rates[2] == 1 else 3,
                dilation=atrous_rates[2],
                padding=0 if atrous_rates[2] == 1 else atrous_rates[2],
                groups=5 * self.C,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg),
            ConvModule(
                5 * self.C,
                self.C,
                kernel_size=1,
                padding=0,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)

        ]) for _ in range(self.l)])

    def forward(self, x):
        """
        Performs a series of multiscale depthwise convolutions  on the given feature map.
        :param x: a feature map of size (B,C,H,W)
        :return: a feature map of same size
        """
        # final_shortcut = x
        for i in range(self.l):
            shortcut = x
            x = nnstacking(x)

            x1 = self.depthwise[i][0](x)  # x1 (Bx5CxHxW)
            x2 = self.depthwise[i][1](x)  # x2 (Bx5CxHxW)
            x3 = self.depthwise[i][2](x)  # x3 (Bx5CxHxW)

            x = x1 + x2 + x3  # x(Bx5CxHxW)
            x = self.depthwise[i][3](x)
            x = x + shortcut

        return x  # + final_shortcut


@MODELS.register_module()
class PyHead(BaseDecodeHead):
    def __init__(self,
                 feat_dim=128,
                 l=3,
                 k=3,
                 atrous_rates=[1, 8, 18],
                 activation_method='gelu',
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)

        # Parameter Initialization
        self.l = l
        self.k = k
        self.atrous_rates = atrous_rates
        self.feat_dim = feat_dim
        self.activation_method = activation_method
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        up_method = torch.nn.UpsamplingNearest2d

        self.reduction_conv_1 = ConvModule(
            in_channels=self.in_channels[0],
            out_channels=self.feat_dim,
            kernel_size=1,
            padding=0,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.reduction_conv_2 = ConvModule(
            in_channels=self.in_channels[2],
            out_channels=self.feat_dim,
            kernel_size=1,
            padding=0,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.reduction_conv_3 = ConvModule(
            in_channels=self.in_channels[3],
            out_channels=self.feat_dim,
            kernel_size=1,
            padding=0,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.reduction_conv_4 = ConvModule(
            in_channels=4 * self.feat_dim,
            out_channels= 2 * self.feat_dim,
            kernel_size=1,
            padding=0,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.long_range_block_1_1 = LongRangeDW(in_channels=2 * self.feat_dim,
                                                l=self.l, k=self.k, atrous_rates=self.atrous_rates,
                                                activation_method=self.activation_method, conv_cfg=self.conv_cfg,
                                                norm_cfg=self.norm_cfg,
                                                act_cfg=self.act_cfg)

        self.long_range_block_1_2 = LongRangeDW(in_channels=2 * self.feat_dim,
                                                l=self.l, k=self.k, atrous_rates=self.atrous_rates,
                                                activation_method=self.activation_method, conv_cfg=self.conv_cfg,
                                                norm_cfg=self.norm_cfg,
                                                act_cfg=self.act_cfg)

        self.long_range_block_3 = LongRangeDW(in_channels=2 * self.feat_dim,
                                              l=self.l, k=self.k, atrous_rates=self.atrous_rates,
                                              activation_method=self.activation_method, conv_cfg=self.conv_cfg,
                                              norm_cfg=self.norm_cfg,
                                              act_cfg=self.act_cfg)

        self.ups1 = up_method(scale_factor=2)
        self.ups2 = up_method(scale_factor=4)
        self.ups3 = up_method(scale_factor=4)
        self.ups4 = up_method(scale_factor=2)

    def forward(self, features):
        fh_, _, fm_, fl_ = features

        fl = self.reduction_conv_3(fl_)
        fm = self.reduction_conv_2(fm_)
        fh = self.reduction_conv_1(fh_)

        up_fl = self.ups1(fl)

        f1 = torch.concat([fm, up_fl], dim=1)

        f1 = self.long_range_block_1_1(f1)

        up_fm = self.ups2(fm)

        f2 = torch.concat([fh, up_fm], dim=1)

        f2 = self.long_range_block_1_2(f2)

        up_f1 = self.ups3(f1)

        f3 = self.reduction_conv_4(torch.concat([f2, up_f1], dim=1))

        f3 = self.long_range_block_3(f3)

        seg_map = self.ups4(self.cls_seg(f3))

        return seg_map
