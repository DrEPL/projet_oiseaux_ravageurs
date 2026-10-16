"""
Package src : Détection d'oiseaux ravageurs du riz – Vallée du fleuve Sénégal.
"""

from .config import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_THRESHOLD,
    IMAGE_SIZE,
    LABEL_NON_PRIO,
    LABEL_PEST,
    PROXY_SPECIES_FAMILIES,
    PROXY_SPECIES_IDS,
)
from .model import BirdPestClassifier, PredictionResult
from .utils import load_image, quick_predict

__all__ = [
    "BirdPestClassifier",
    "PredictionResult",
    "load_image",
    "quick_predict",
    "DEFAULT_MODEL_PATH",
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_THRESHOLD",
    "IMAGE_SIZE",
    "LABEL_PEST",
    "LABEL_NON_PRIO",
    "PROXY_SPECIES_IDS",
    "PROXY_SPECIES_FAMILIES",
]
