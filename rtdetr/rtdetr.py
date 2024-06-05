from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa
from PIL import Image
from torchvision.transforms import Compose
from torchvision.transforms import ConvertImageDtype
from torchvision.transforms import Resize
from torchvision.transforms import ToTensor

from .nn.decoder import RTDETRTransformer
from .nn.encoder import HybridEncoder
from .nn.postprocessor import RTDETRPostProcessor
from .nn.presnet import PResNet


class RTDETR(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        encoder: HybridEncoder,
        decoder: RTDETRTransformer,
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


def rtdetr_r18vd_6x_coco(num_classes: int = 80):
    model = RTDETR(
        backbone=PResNet(
            depth=18, variant="d", freeze_at=-1, return_idx=[1, 2, 3], num_stages=4, freeze_norm=False, pretrained=True
        ),
        encoder=HybridEncoder(
            in_channels=[128, 256, 512],
            feat_strides=[8, 16, 32],
            hidden_dim=256,
            use_encoder_idx=[2],
            num_encoder_layers=1,
            nhead=8,
            dim_feedforward=1024,
            dropout=0.0,
            enc_act="gelu",
            pe_temperature=10000,
            expansion=0.5,
            depth_mult=1,
            act="silu",
            eval_spatial_size=[640, 640],
        ),
        decoder=RTDETRTransformer(
            num_classes=num_classes,
            feat_channels=[256, 256, 256],
            feat_strides=[8, 16, 32],
            hidden_dim=256,
            num_levels=3,
            num_queries=300,
            num_decoder_layers=3,
            num_denoising=100,
            eval_idx=-1,
            eval_spatial_size=[640, 640],
        ),
        multi_scale=[480, 512, 544, 576, 608, 640, 640, 640, 672, 704, 736, 768, 800],
    )
    return model


class Detection:
    def __init__(
        self,
        model_path: str,
        num_classes: int = 80,
        num_top_queries: int = 300,
        threshold: float = 0.5,
        device: str = "cpu",
    ) -> None:
        self.threshold = threshold

        self.model = rtdetr_r18vd_6x_coco(num_classes=num_classes)
        self.post_processor = RTDETRPostProcessor(num_classes=num_classes, num_top_queries=num_top_queries)
        self.transform = Compose(
            [
                Resize((640, 640)),
                ToTensor(),
                ConvertImageDtype(torch.float32),
            ]
        )

        self.model.eval()
        state_dict = torch.load(model_path, map_location=device)
        self.model.load_state_dict(state_dict["model"])

    @torch.no_grad()
    def __call__(self, img: Image.Image):
        img = img.convert("RGB")
        x = self.transform(img).unsqueeze(0)
        y = self.model(x)

        w, h = img.size
        orig_target_sizes = torch.tensor([[w, h]])

        result = self.post_processor(y, orig_target_sizes)[0]
        return result
