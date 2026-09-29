"""Lightweight re-exports. Kept lazy (PEP 562) on purpose: importing
``hyppar.models.par_classifier`` must NOT pull in the hyperbolic modules
(``hyppar``/``lorentz``), because the PAR Codabench zip only ships the
classifier + retrieval code.
"""

_LAZY = {
    "AttributePrototypeModel": (".hyppar", "AttributePrototypeModel"),
    "ImageQueryModel": (".hyppar", "ImageQueryModel"),
    "LorentzProjection": (".lorentz", "LorentzProjection"),
    "euclidean_distance": (".lorentz", "euclidean_distance"),
    "lorentz_distance": (".lorentz", "lorentz_distance"),
    "ConvNeXtPAR": (".par_classifier", "ConvNeXtPAR"),
    "build_par_model": (".par_classifier", "build_par_model"),
}

__all__ = sorted(_LAZY)


def __getattr__(name: str):
    if name in _LAZY:
        import importlib

        module_name, attr = _LAZY[name]
        module = importlib.import_module(module_name, package=__name__)
        value = getattr(module, attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + __all__)
