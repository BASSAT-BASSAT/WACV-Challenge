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
        model_name: str = "hyper3-clip-v1",
        device: str | None = None,
        local_files_only: bool = False,
    ) -> None:
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

    @property
    def dim(self) -> int:
        return int(self.model.dim)

    def encode_images(self, images: Sequence[Image.Image]) -> np.ndarray:
        embeddings = self.model.encode_images(list(images))
        return np.asarray(embeddings, dtype=np.float32)

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        embeddings = self.model.encode_texts(list(texts))
        return np.asarray(embeddings, dtype=np.float32)