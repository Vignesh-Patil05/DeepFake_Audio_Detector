"""
Dataset package.

Note: Some image pipelines depend on heavy optional deps (albumentations/imgaug).
Audio-only training should not fail if those deps are unavailable or incompatible
with the local NumPy version.
"""

try:
    from .albu import IsotropicResize  # used by image datasets only
except Exception:  # pragma: no cover
    IsotropicResize = None