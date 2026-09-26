"""Retrieval metrics: mAP, Rank-1, mADM (Specker)."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


def _average_precision(ranked_relevant: np.ndarray) -> float:
    """ranked_relevant: binary array in rank order."""
    if ranked_relevant.sum() == 0:
        return 0.0
    hits = 0
    sum_prec = 0.0
    for i, rel in enumerate(ranked_relevant, start=1):
        if rel:
            hits += 1
            sum_prec += hits / i
    return float(sum_prec / hits)


def map_rank1(
    distances: np.ndarray,
    query_ids: np.ndarray,
    gallery_ids: np.ndarray,
) -> dict[str, float]:
    """Binary relevance = same semantic id.

    distances: (Q, G) smaller = better.
    """
    aps = []
    r1 = []
    for qi in range(distances.shape[0]):
        order = np.argsort(distances[qi], kind="mergesort")
        ranked_ids = gallery_ids[order]
        relevant = (ranked_ids == query_ids[qi]).astype(np.int32)
        aps.append(_average_precision(relevant))
        r1.append(1.0 if relevant.size and relevant[0] == 1 else 0.0)
    return {"mAP": float(np.mean(aps)), "R1": float(np.mean(r1))}


def madm(
    distances: np.ndarray,
    queries: np.ndarray,
    gallery_attrs: np.ndarray,
    query_ids: np.ndarray,
    gallery_ids: np.ndarray,
) -> float:
    """Mean Attribute Degree-of-Match metric (Specker thesis).

    PrecDoM@k = mean of DoMNorm over top-k.
    ADM = mean PrecDoM@k over ranks k that are binary-relevant (exact semantic id).
    DoM = 1 - Hamming(q, gt)/A
    DoMNorm = max(0, (DoM - mean_DoM) / (1 - mean_DoM))
    """
    n_attr = queries.shape[1]
    scores = []
    for qi in range(distances.shape[0]):
        q = queries[qi]
        # Hamming to all gallery
        ham = np.abs(gallery_attrs - q[None, :]).sum(axis=1)
        dom = 1.0 - ham / float(n_attr)
        dom_mean = float(dom.mean())
        denom = max(1.0 - dom_mean, 1e-8)
        dom_norm = np.maximum(0.0, (dom - dom_mean) / denom)

        order = np.argsort(distances[qi], kind="mergesort")
        ranked_rel = (gallery_ids[order] == query_ids[qi]).astype(np.int32)
        ranked_dom = dom_norm[order]

        # PrecDoM@k and AP-style aggregation over relevant ranks
        if ranked_rel.sum() == 0:
            scores.append(0.0)
            continue
        cumsum = np.cumsum(ranked_dom)
        prec_dom = cumsum / np.arange(1, len(ranked_dom) + 1)
        adm = float(prec_dom[ranked_rel.astype(bool)].mean())
        scores.append(adm)
    return float(np.mean(scores))


def evaluate_ranking(
    distances: np.ndarray,
    queries: np.ndarray,
    gallery_attrs: np.ndarray,
    gallery_ids: np.ndarray,
    gallery_domains: list[str] | None = None,
) -> dict[str, Any]:
    """Full Track-2 style eval. query_id for query i is i (row index)."""
    q_ids = np.arange(queries.shape[0], dtype=np.int64)
    overall = map_rank1(distances, q_ids, gallery_ids)
    overall["mADM"] = madm(distances, queries, gallery_attrs, q_ids, gallery_ids)

    # Query complexity: number of positive bits
    n_pos = queries.sum(axis=1)
    buckets = {
        "q1": n_pos == 1,
        "q2_3": (n_pos >= 2) & (n_pos <= 3),
        "q_many": n_pos >= 4,
    }
    by_complexity = {}
    for name, mask in buckets.items():
        if not mask.any():
            continue
        idx = np.where(mask)[0]
        sub = {
            **map_rank1(distances[idx], q_ids[idx], gallery_ids),
            "mADM": madm(distances[idx], queries[idx], gallery_attrs, q_ids[idx], gallery_ids),
            "n_queries": int(mask.sum()),
        }
        by_complexity[name] = sub

    by_domain: dict[str, Any] = {}
    if gallery_domains is not None:
        domains = sorted(set(gallery_domains))
        for dom in domains:
            gmask = np.array([d == dom for d in gallery_domains])
            if not gmask.any():
                continue
            # Keep queries that have at least one gallery hit in this domain
            g_ids_sub = gallery_ids[gmask]
            # Filter queries present in domain gallery
            present = np.isin(q_ids, g_ids_sub)
            if not present.any():
                continue
            q_idx = np.where(present)[0]
            dist_sub = distances[np.ix_(q_idx, np.where(gmask)[0])]
            by_domain[dom] = {
                **map_rank1(dist_sub, q_ids[q_idx], g_ids_sub),
                "mADM": madm(
                    dist_sub,
                    queries[q_idx],
                    gallery_attrs[gmask],
                    q_ids[q_idx],
                    g_ids_sub,
                ),
                "n_gallery": int(gmask.sum()),
                "n_queries": int(present.sum()),
            }

    return {"overall": overall, "by_complexity": by_complexity, "by_domain": by_domain}
