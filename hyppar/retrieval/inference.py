"""Batch inference with optional horizontal-flip TTA."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


@torch.no_grad()
def predict_probs(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    tta_flip: bool = False,
    use_amp: bool = True,
) -> torch.Tensor:
    model.eval()
    chunks: list[torch.Tensor] = []
    for batch in tqdm(loader, desc="predict", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
            logits = model(images)
            probs = torch.sigmoid(logits)
            if tta_flip:
                logits_f = model(torch.flip(images, dims=[3]))
                probs = 0.5 * (probs + torch.sigmoid(logits_f))
        chunks.append(probs.float().cpu())
    return torch.cat(chunks, dim=0)
