"""Copied and modified from https://github.com/lyuwenyu/RT-DETR/blob/main/rtdetr_pytorch/src/nn/backbone/presnet.py"""

from collections import OrderedDict
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa

from .common import ConvNormLayer
from .common import FrozenBatchNorm2d
from .common import get_activation

ResNet_cfg = {
    18: [2, 2, 2, 2],
    34: [3, 4, 6, 3],
    50: [3, 4, 6, 3],
    101: [3, 4, 23, 3],
    # 152: [3, 8, 36, 3],
}


donwload_url = {
    18: "https://github.com/lyuwenyu/storage/releases/download/v0.1/ResNet18_vd_pretrained_from_paddle.pth",
    34: "https://github.com/lyuwenyu/storage/releases/download/v0.1/ResNet34_vd_pretrained_from_paddle.pth",
    50: "https://github.com/lyuwenyu/storage/releases/download/v0.1/ResNet50_vd_ssld_v2_pretrained_from_paddle.pth",
    101: "https://github.com/lyuwenyu/storage/releases/download/v0.1/ResNet101_vd_ssld_pretrained_from_paddle.pth",
}


class Block(nn.Module):
    expansion: int


class BasicBlock(Block):
    expansion = 1

    def __init__(
        self, ch_in: int, ch_out: int, stride: int, shortcut: bool, act: str = "relu", variant: str = "b"
    ) -> None:
        super().__init__()
        self.shortcut = shortcut

        self.short: nn.Module | None = None
        if not shortcut:
            if variant == "d" and stride == 2:
                self.short = nn.Sequential(
                    OrderedDict(
                        [
                            ("pool", nn.AvgPool2d(2, 2, 0, ceil_mode=True)),
                            ("conv", ConvNormLayer(ch_in, ch_out, 1, 1)),
                        ]
                    )
                )
            else:
                self.short = ConvNormLayer(ch_in, ch_out, 1, stride)

        self.branch2a = ConvNormLayer(ch_in, ch_out, 3, stride, act=act)
        self.branch2b = ConvNormLayer(ch_out, ch_out, 3, 1, act=None)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        out = self.branch2a(x)
        out = self.branch2b(out)

        short = x if self.shortcut else self.short(x)
        # if self.shortcut:
        #     short = x
        # else:
        #     short = self.short(x)

        out = out + short
        out = self.act(out)

        return out


class BottleNeck(Block):
    expansion = 4
    short: nn.Module

    def __init__(
        self, ch_in: int, ch_out: int, stride: int, shortcut: bool, act: str = "relu", variant: str = "b"
    ) -> None:
        super().__init__()

        if variant == "a":
            stride1, stride2 = stride, 1
        else:
            stride1, stride2 = 1, stride

        width = ch_out

        self.branch2a = ConvNormLayer(ch_in, width, 1, stride1, act=act)
        self.branch2b = ConvNormLayer(width, width, 3, stride2, act=act)
        self.branch2c = ConvNormLayer(width, ch_out * self.expansion, 1, 1)

        self.shortcut = shortcut

        if not shortcut:
            if variant == "d" and stride == 2:
                self.short = nn.Sequential(
                    OrderedDict(
                        [
                            ("pool", nn.AvgPool2d(2, 2, 0, ceil_mode=True)),
                            (
                                "conv",
                                ConvNormLayer(ch_in, ch_out * self.expansion, 1, 1),
                            ),
                        ]
                    )
                )
            else:
                self.short = ConvNormLayer(ch_in, ch_out * self.expansion, 1, stride)

        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.branch2a(x)
        out = self.branch2b(out)
        out = self.branch2c(out)

        short = x if self.shortcut else self.short(x)
        # if self.shortcut:
        #     short = x
        # else:
        #     short = self.short(x)

        out = out + short
        out = self.act(out)

        return out


class Blocks(nn.Module):
    def __init__(
        self,
        block_class: type[Block],
        ch_in: int,
        ch_out: int,
        count: int,
        stage_num: int,
        act: str = "relu",
        variant: str = "b",
    ) -> None:
        super().__init__()

        self.blocks = nn.ModuleList()
        for i in range(count):
            self.blocks.append(
                block_class(
                    ch_in,
                    ch_out,
                    stride=2 if i == 0 and stage_num != 2 else 1,
                    shortcut=i != 0,
                    variant=variant,
                    act=act,
                )
            )

            if i == 0:
                ch_in = ch_out * block_class.expansion

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for block in self.blocks:
            out = block(out)
        return out


class PResNet(nn.Module):
    def __init__(
        self,
        depth: Literal[18, 34, 50, 101],
        variant: str = "d",
        num_stages: int = 4,
        return_idx: list[int] | None = None,
        act: str = "relu",
        freeze_at: int = -1,
        freeze_norm: bool = True,
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        return_idx = return_idx or [0, 1, 2, 3]

        block_nums = ResNet_cfg[depth]
        ch_in = 64

        if variant in ["c", "d"]:
            conv_def = [
                ("conv1_1", [3, ch_in // 2, 3, 2]),
                ("conv1_2", [ch_in // 2, ch_in // 2, 3, 1]),
                ("conv1_3", [ch_in // 2, ch_in, 3, 1]),
            ]
        else:
            conv_def = [("conv1_1", [3, ch_in, 7, 2])]

        self.conv1 = nn.Sequential(
            OrderedDict(
                [
                    (_name, ConvNormLayer(conf_c_in, c_out, k, s, act=act))
                    for _name, (conf_c_in, c_out, k, s) in conv_def
                ]
            )
        )

        ch_out_list = [64, 128, 256, 512]
        block_class = BottleNeck if depth >= 50 else BasicBlock

        _out_channels = [block_class.expansion * v for v in ch_out_list]
        _out_strides = [4, 8, 16, 32]

        self.res_layers = nn.ModuleList()
        for i in range(num_stages):
            stage_num = i + 2
            self.res_layers.append(
                Blocks(
                    block_class,
                    ch_in,
                    ch_out_list[i],
                    block_nums[i],
                    stage_num,
                    act=act,
                    variant=variant,
                )
            )
            ch_in = _out_channels[i]

        self.return_idx = return_idx
        self.out_channels = [_out_channels[_i] for _i in return_idx]
        self.out_strides = [_out_strides[_i] for _i in return_idx]

        if freeze_at >= 0:
            self._freeze_parameters(self.conv1)
            for i in range(min(freeze_at, num_stages)):
                self._freeze_parameters(self.res_layers[i])

        if freeze_norm:
            self._freeze_norm(self)

        if pretrained:
            state = torch.hub.load_state_dict_from_url(donwload_url[depth])
            self.load_state_dict(state)
            print(f"Load PResNet{depth} state_dict")

    def _freeze_parameters(self, m: nn.Module) -> None:
        for p in m.parameters():
            p.requires_grad = False

    def _freeze_norm(self, m: nn.Module) -> nn.Module:
        if isinstance(m, nn.BatchNorm2d):
            m = FrozenBatchNorm2d(m.num_features)
        else:
            for name, child in m.named_children():
                _child = self._freeze_norm(child)
                if _child is not child:
                    setattr(m, name, _child)
        return m

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        conv1 = self.conv1(x)
        x = F.max_pool2d(conv1, kernel_size=3, stride=2, padding=1)
        outs = []
        for idx, stage in enumerate(self.res_layers):
            x = stage(x)
            if idx in self.return_idx:
                outs.append(x)
        return outs
