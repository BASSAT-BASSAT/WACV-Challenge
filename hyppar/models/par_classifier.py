"""Strong PAR classifier for attribute-based retrieval (competition-winning path)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm


class ConvNeXtPAR(nn.Module):
    """ConvNeXt backbone + dropout head for 40-way multi-label PAR."""

    BACKBONES = {
        "convnext_tiny": (tvm.convnext_tiny, tvm.ConvNeXt_Tiny_Weights),
        "convnext_small": (tvm.convnext_small, tvm.ConvNeXt_Small_Weights),
        "convnext_base": (tvm.convnext_base, tvm.ConvNeXt_Base_Weights),
    }

    def __init__(
        self,
        backbone: str = "convnext_base",
        pretrained: bool = True,
        dropout: float = 0.3,
        num_attrs: int = 40,
    ) -> None:
        super().__init__()
        backbone = backbone.lower().replace("-", "_")
        if backbone not in self.BACKBONES:
            raise ValueError(f"Unknown backbone {backbone}; choose from {list(self.BACKBONES)}")
        builder, weights_cls = self.BACKBONES[backbone]
        weights = weights_cls.DEFAULT if pretrained else None
        net = builder(weights=weights)
        self.backbone_name = backbone
        self.features = net.features
        self.avgpool = net.avgpool
        self.out_dim = net.classifier[2].in_features  # type: ignore[index]
        self.dropout = nn.Dropout(p=dropout)
        self.head = nn.Linear(self.out_dim, num_attrs)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        x = self.features(images)
        x = self.avgpool(x)
        return torch.flatten(x, 1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(self.dropout(self.encode(images)))

    @torch.no_grad()
    def predict_proba(self, images: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(images))


class CLIPPAR(nn.Module):
    """CLIP ViT-B/32 image encoder + linear PAR head (strong cross-domain prior)."""

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        dropout: float = 0.2,
        num_attrs: int = 40,
    ) -> None:
        super().__init__()
        try:
            import open_clip
        except ImportError as exc:
            raise ImportError("Install open-clip-torch for CLIP PAR: pip install open-clip-torch") from exc

        clip_model, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.visual = clip_model.visual
        self.out_dim = self.visual.output_dim
        self.dropout = nn.Dropout(p=dropout)
        self.head = nn.Linear(self.out_dim, num_attrs)
        self.backbone_name = f"clip_{model_name.lower().replace('-', '_')}"

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return self.visual(images)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(self.dropout(self.encode(images)))

    @torch.no_grad()
    def predict_proba(self, images: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(images))


def build_par_model(
    backbone: str = "convnext_base",
    pretrained: bool = True,
    dropout: float = 0.3,
) -> nn.Module:
    bb = backbone.lower()
    if bb.startswith("clip"):
        # clip_vit_b_32 or clip
        return CLIPPAR(dropout=dropout)
    return ConvNeXtPAR(backbone=bb, pretrained=pretrained, dropout=dropout)
