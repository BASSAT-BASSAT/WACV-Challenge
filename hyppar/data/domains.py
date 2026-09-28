"""Domain helpers for leave-one-domain-out (LODO) cross-validation."""

from __future__ import annotations

DOMAINS: tuple[str, ...] = ("Market1501", "PA100k", "PETA")


def normalize_domain(domain: str) -> str:
    d = domain.replace("\\", "/").split("/")[0]
    if d.upper() == "PA100K":
        return "PA100k"
    return d


def domain_in(domain: str, allowed: set[str] | list[str]) -> bool:
    return normalize_domain(domain) in {normalize_domain(a) for a in allowed}


def lodo_folds() -> list[tuple[str, tuple[str, str]]]:
    """(held_out, (train_a, train_b)) for each fold."""
    folds: list[tuple[str, tuple[str, str]]] = []
    for held in DOMAINS:
        train = tuple(d for d in DOMAINS if d != held)
        folds.append((held, (train[0], train[1])))
    return folds
