"""by lyuwenyu"""

import numpy as np
import torch.nn as nn
import torch.nn.functional as F  # noqa

from .nn.decoder import TransformerDecoder
from .nn.encoder import HybridEncoder


class RTDETR(nn.Module):
    __inject__ = [
        "backbone",
        "encoder",
        "decoder",
    ]

    def __init__(
        self,
        backbone: nn.Module,
        encoder: HybridEncoder,
        decoder: TransformerDecoder,
        multi_scale: list[int] | None = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder
        self.multi_scale = multi_scale

    def forward(self, x, targets=None):
        if self.multi_scale and self.training:
            sz = np.random.choice(self.multi_scale)
            x = F.interpolate(x, size=[sz, sz])

        x = self.backbone(x)
        x = self.encoder(x)
        x = self.decoder(x, targets)

        return x
