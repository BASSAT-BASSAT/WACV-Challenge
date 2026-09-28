"""Optional Hyper3-CLIP runtime boundary.

The dependency is imported lazily so the existing ConvNeXt pipeline remains
usable in offline environments and Codabench packages do not require it.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from PIL import Image


class Hyper3ClipRuntime:
    def __init__(
        self,
        model_name: str = "hyper3labs/hyper3-clip-v1",
        device: str | None = None,
        local_files_only: bool = False,
        backend: str = "transformers",
    ) -> None:
        self.backend = backend
        self.device = device or ("cuda" if _cuda_available() else "cpu")
        if backend == "transformers":
            try:
                from transformers import AutoImageProcessor, AutoModel, AutoTokenizer
            except ImportError as error:
                raise ImportError(
                    "Install Hyper3 support with: pip install -e '.[hyper3]'"
                ) from error
            self.model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                local_files_only=local_files_only,
            ).to(self.device).eval()
            self.image_processor = AutoImageProcessor.from_pretrained(
                model_name, trust_remote_code=True, local_files_only=local_files_only
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True, local_files_only=local_files_only
            )
            self._curvature = getattr(self.model, "curvature", 1.0)
        elif backend == "hyper_models":
            try:
                import hyper_models
            except ImportError as error:
                raise ImportError(
                    "Install Hyper3 support with: pip install -e '.[hyper3]'"
                ) from error
            self.model = hyper_models.load(
                model_name,
                device=device,
                local_files_only=local_files_only,
            )
            if getattr(self.model, "geometry", None) != "hyperboloid":
                raise ValueError("Hyper3-CLIP runtime must return hyperboloid embeddings")
            if getattr(self.model, "dim", None) != 513:
                raise ValueError("Expected Hyper3-CLIP embeddings with dimension 513")
            self.image_processor = None
            self.tokenizer = None
            self._curvature = 1.0
        else:
            raise ValueError("backend must be transformers or hyper_models")

    @property
    def dim(self) -> int:
        return 513

    @property
    def curvature(self):
        value = self._curvature
        if hasattr(value, "detach"):
            value = value.detach().cpu().item()
        return float(value)

    def encode_images(self, images: Sequence[Image.Image]) -> np.ndarray:
        if self.backend == "transformers":
            inputs = self.image_processor(images=list(images), return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self.device)
            with _inference_mode():
                embeddings = self.model.encode_image(pixel_values)
            embeddings = embeddings.detach().cpu().numpy()
        else:
            embeddings = self.model.encode_images(list(images))
        return np.asarray(embeddings, dtype=np.float32)

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        if self.backend == "transformers":
            tokens = self.tokenizer(
                list(texts), padding=True, truncation=True, max_length=77, return_tensors="pt"
            )
            with _inference_mode():
                embeddings = self.model.encode_text(
                    tokens["input_ids"].to(self.device),
                    tokens["attention_mask"].to(self.device),
                )
            embeddings = embeddings.detach().cpu().numpy()
        else:
            embeddings = self.model.encode_texts(list(texts))
        return np.asarray(embeddings, dtype=np.float32)


def _cuda_available() -> bool:
    import torch

    return torch.cuda.is_available()


def _inference_mode():
    import torch

    return torch.inference_mode()