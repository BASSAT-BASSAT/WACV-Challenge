"""HypPAR models: image-query and hyperbolic attribute-prototype retrieval."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm

from .lorentz import (
    EuclideanProjection,
    LorentzProjection,
    entailment_distance,
    euclidean_distance,
    exp_map0,
    lorentz_distance,
)


class ConvNeXtEncoder(nn.Module):
    """ConvNeXt backbone with Tiny, Small, and Base variants."""

    def __init__(
        self,
        backbone: str = "convnext_tiny",
        pretrained: bool = True,
        freeze: bool = True,
    ) -> None:
        super().__init__()
        backbone = backbone.lower().replace("-", "_")
        if backbone in ("convnext_tiny", "tiny"):
            weights = tvm.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
            net = tvm.convnext_tiny(weights=weights)
        elif backbone in ("convnext_small", "small"):
            weights = tvm.ConvNeXt_Small_Weights.DEFAULT if pretrained else None
            net = tvm.convnext_small(weights=weights)
        elif backbone in ("convnext_base", "base"):
            weights = tvm.ConvNeXt_Base_Weights.DEFAULT if pretrained else None
            net = tvm.convnext_base(weights=weights)
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        self.backbone_name = backbone
        self.features = net.features
        self.avgpool = net.avgpool
        self.out_dim = net.classifier[2].in_features  # type: ignore[index]
        if freeze:
            for p in self.features.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)

    def freeze_all(self) -> None:
        for p in self.features.parameters():
            p.requires_grad = False

    def unfreeze_all(self) -> None:
        for p in self.features.parameters():
            p.requires_grad = True

    def unfreeze_last_stages(self, n_stages: int = 2) -> None:
        stages = list(self.features.children())
        for stage in stages[-n_stages:]:
            for p in stage.parameters():
                p.requires_grad = True


# Back-compat alias
ConvNeXtTinyEncoder = ConvNeXtEncoder


class QueryMLP(nn.Module):
    def __init__(self, attr_dim: int = 40, hidden: int = 256, out_dim: int = 128, geometry: str = "euclidean"):
        super().__init__()
        self.geometry = geometry
        self.mlp = nn.Sequential(
            nn.Linear(attr_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
        )
        self.geometry_name = geometry

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        h = self.mlp(q)
        if self.geometry_name == "euclidean":
            return F.normalize(h, dim=-1)
        return exp_map0(h, c=1.0)


class ImageQueryModel(nn.Module):
    """Approach A: shared image / query embedding space."""

    def __init__(
        self,
        geometry: str = "euclidean",
        embed_dim: int = 128,
        curvature: float = 1.0,
        learnable_c: bool = False,
        freeze_backbone: bool = True,
        pretrained: bool = True,
        backbone: str = "convnext_tiny",
        **_kwargs,
    ) -> None:
        super().__init__()
        self.geometry = geometry
        self.score_mode = "distance"
        self.encoder = ConvNeXtEncoder(
            backbone=backbone, pretrained=pretrained, freeze=freeze_backbone
        )
        if geometry == "euclidean":
            self.img_proj = EuclideanProjection(self.encoder.out_dim, embed_dim)
            self.query_encoder = QueryMLP(40, 256, embed_dim, geometry="euclidean")
        else:
            self.img_proj = LorentzProjection(
                self.encoder.out_dim, embed_dim, c=curvature, learnable_c=learnable_c
            )
            self.query_encoder = QueryMLP(40, 256, embed_dim, geometry="lorentz")

    def encode_features(self, images: torch.Tensor) -> torch.Tensor:
        return self.encoder(images)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        return self.img_proj(self.encoder(images))

    def encode_query(self, queries: torch.Tensor) -> torch.Tensor:
        h = self.query_encoder.mlp(queries)
        if self.geometry == "euclidean":
            return F.normalize(h, dim=-1)
        return exp_map0(h, c=self.img_proj.c)

    def pairwise_distance(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if self.geometry == "euclidean":
            return euclidean_distance(a, b)
        return lorentz_distance(a, b, c=self.img_proj.c)

    def score_matrix(self, query_emb: torch.Tensor, gallery: torch.Tensor) -> torch.Tensor:
        """Similarity (higher = better): -distance. Shape (Q, G)."""
        if self.geometry == "euclidean":
            d = (query_emb[:, None, :] - gallery[None, :, :]).pow(2).sum(-1).sqrt()
        else:
            q = query_emb[:, None, :].expand(-1, gallery.size(0), -1)
            g = gallery[None, :, :].expand(query_emb.size(0), -1, -1)
            d = lorentz_distance(
                q.reshape(-1, q.size(-1)), g.reshape(-1, g.size(-1)), c=self.img_proj.c
            ).view(query_emb.size(0), gallery.size(0))
        return -d


class AttributePrototypeModel(nn.Module):
    """Approach B/C: attribute prototypes + optional Lorentz entailment retrieval.

    Hyperbolic story (default for geometry=lorentz, score_mode=entailment):
      - Sparse queries sit near the origin (general).
      - Images with richer attributes sit deeper (specific).
      - Ranking uses cone entailment: gallery should entail the query.
    Strong PAR head (attr_head) supplies attribute BCE + optional L1 ablation scoring.
    """

    def __init__(
        self,
        geometry: str = "lorentz",
        embed_dim: int = 128,
        curvature: float = 1.0,
        learnable_c: bool = False,
        freeze_backbone: bool = True,
        pretrained: bool = True,
        composition: str = "mean",  # mean | learned
        score_mode: str = "entailment",  # entailment | distance | attr_l1 | hybrid
        entail_K: float = 0.1,
        entail_beta: float = 1.0,
        radius_compose: bool = True,
        hybrid_mix: float = 0.7,  # weight on hyperbolic score in hybrid mode
        backbone: str = "convnext_tiny",
    ) -> None:
        super().__init__()
        self.geometry = geometry
        self.composition = composition
        self.embed_dim = embed_dim
        self.score_mode = score_mode
        self.entail_K = entail_K
        self.entail_beta = entail_beta
        self.radius_compose = radius_compose
        self.hybrid_mix = hybrid_mix

        self.encoder = ConvNeXtEncoder(
            backbone=backbone, pretrained=pretrained, freeze=freeze_backbone
        )
        # Euclidean PAR head — main attribute supervision (competition-aligned).
        self.attr_head = nn.Linear(self.encoder.out_dim, 40)

        if geometry == "euclidean":
            self.img_proj = EuclideanProjection(self.encoder.out_dim, embed_dim)
            proto = torch.randn(40, embed_dim) * 0.05
            self.prototypes = nn.Parameter(F.normalize(proto, dim=-1))
            # Fall back: Euclid has no cones; entailment -> distance.
            if score_mode == "entailment":
                self.score_mode = "distance"
        else:
            self.img_proj = LorentzProjection(
                self.encoder.out_dim, embed_dim, c=curvature, learnable_c=learnable_c
            )
            self.proto_tangent = nn.Parameter(torch.randn(40, embed_dim) * 0.05)

        self.attr_bias = nn.Parameter(torch.zeros(40))
        self.attr_scale = nn.Parameter(torch.tensor(1.0))
        if composition == "learned":
            self.compose_mlp = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(inplace=True),
                nn.Linear(embed_dim, embed_dim),
            )

    @property
    def c(self):
        if self.geometry == "lorentz":
            return self.img_proj.c
        return 1.0

    def prototype_points(self) -> torch.Tensor:
        if self.geometry == "euclidean":
            return F.normalize(self.prototypes, dim=-1)
        return exp_map0(self.proto_tangent, c=self.c)

    def encode_features(self, images: torch.Tensor) -> torch.Tensor:
        return self.encoder(images)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(images)
        h = self.img_proj(feat)
        if self.geometry == "lorentz" and self.radius_compose:
            # Deeper embeddings when the image fires more attributes (more specific).
            with torch.no_grad():
                dens = torch.sigmoid(self.attr_head(feat.detach())).sum(dim=-1, keepdim=True) / 40.0
            dens = dens.clamp(0.15, 1.0)
            spatial = h[..., 1:] * dens
            c = self.c
            x0 = torch.sqrt((1.0 / c) + (spatial * spatial).sum(dim=-1, keepdim=True))
            h = torch.cat([x0, spatial], dim=-1)
        return h

    def attribute_logits_from_features(self, feat: torch.Tensor) -> torch.Tensor:
        return self.attr_head(feat)

    def attribute_logits(self, h_img: torch.Tensor) -> torch.Tensor:
        """Prototype-distance logits (auxiliary); prefer attr_head for BCE."""
        protos = self.prototype_points()
        if self.geometry == "euclidean":
            d = (h_img[:, None, :] - protos[None, :, :]).pow(2).sum(-1).sqrt()
        else:
            b = h_img.size(0)
            hi = h_img[:, None, :].expand(-1, 40, -1).reshape(-1, h_img.size(-1))
            pr = protos[None, :, :].expand(b, -1, -1).reshape(-1, protos.size(-1))
            d = lorentz_distance(hi, pr, c=self.c).view(b, 40)
        return self.attr_bias - self.attr_scale.abs() * d

    def compose_query(self, query: torch.Tensor) -> torch.Tensor:
        """Compose active prototypes; sparse queries stay nearer the origin."""
        protos = self.prototype_points()
        w = query.clamp(min=0.0)
        w_sum = w.sum(dim=-1, keepdim=True).clamp(min=1.0)
        if self.geometry == "euclidean":
            mixed = (w.unsqueeze(-1) * protos.unsqueeze(0)).sum(dim=1) / w_sum
            mixed = F.normalize(mixed, dim=-1)
            if self.composition == "learned":
                mixed = F.normalize(self.compose_mlp(mixed), dim=-1)
            return mixed

        spatial = protos[..., 1:]
        mixed_s = (w.unsqueeze(-1) * spatial.unsqueeze(0)).sum(dim=1) / w_sum
        if self.composition == "learned":
            mixed_s = self.compose_mlp(mixed_s)
        if self.radius_compose:
            # Few active attrs => general query => small radius (near origin).
            n_act = w_sum.clamp(min=1.0)
            scale = (n_act / 40.0).clamp(0.05, 1.0)
            mixed_s = mixed_s * scale
        c = self.c
        x0 = torch.sqrt((1.0 / c) + (mixed_s * mixed_s).sum(dim=-1, keepdim=True))
        return torch.cat([x0, mixed_s], dim=-1)

    def _pairwise_retrieval_distance(self, q_emb: torch.Tensor, gallery: torch.Tensor) -> torch.Tensor:
        qn, gn = q_emb.size(0), gallery.size(0)
        if self.geometry == "euclidean" or self.score_mode == "distance":
            if self.geometry == "euclidean":
                return (q_emb[:, None, :] - gallery[None, :, :]).pow(2).sum(-1).sqrt()
            q = q_emb[:, None, :].expand(-1, gn, -1)
            g = gallery[None, :, :].expand(qn, -1, -1)
            return lorentz_distance(
                q.reshape(-1, q.size(-1)), g.reshape(-1, g.size(-1)), c=self.c
            ).view(qn, gn)

        # Entailment: gallery entails query.
        q = q_emb[:, None, :].expand(-1, gn, -1)
        g = gallery[None, :, :].expand(qn, -1, -1)
        return entailment_distance(
            q.reshape(-1, q.size(-1)),
            g.reshape(-1, g.size(-1)),
            c=self.c,
            K=self.entail_K,
            beta=self.entail_beta,
        ).view(qn, gn)

    def score_matrix(
        self,
        queries: torch.Tensor,
        gallery: torch.Tensor,
        gallery_attr_probs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Similarity (higher better). queries are binary attrs (Q,40)."""

        def _attr_l1_sim() -> torch.Tensor:
            if gallery_attr_probs is None:
                raise ValueError("attr_l1/hybrid scoring needs gallery_attr_probs (G,40)")
            return -(
                gallery_attr_probs.sum(1)[None, :]
                + queries @ (1.0 - 2.0 * gallery_attr_probs).T
            )

        if self.score_mode == "attr_l1":
            return _attr_l1_sim()

        # Force entailment geometry for hybrid hyperbolic branch.
        prev = self.score_mode
        if self.score_mode == "hybrid":
            self.score_mode = "entailment" if self.geometry == "lorentz" else "distance"
        q_emb = self.compose_query(queries)
        hyp_sim = -self._pairwise_retrieval_distance(q_emb, gallery)
        self.score_mode = prev

        if prev != "hybrid":
            return hyp_sim

        # Per-query z-score mix: keep hyperbolic dominant (hybrid_mix).
        l1_sim = _attr_l1_sim()

        def _z(x: torch.Tensor) -> torch.Tensor:
            mu = x.mean(dim=1, keepdim=True)
            sd = x.std(dim=1, keepdim=True).clamp(min=1e-6)
            return (x - mu) / sd

        m = float(self.hybrid_mix)
        return m * _z(hyp_sim) + (1.0 - m) * _z(l1_sim)

    def retrieval_distance_fn(self):
        """For training InfoNCE: distance between query and image Lorentz points."""

        def _d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            if self.geometry == "euclidean" or self.score_mode == "distance":
                if self.geometry == "euclidean":
                    return euclidean_distance(a, b)
                return lorentz_distance(a, b, c=self.c)
            return entailment_distance(a, b, c=self.c, K=self.entail_K, beta=self.entail_beta)

        return _d
