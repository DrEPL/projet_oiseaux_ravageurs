"""
Fonctions utilitaires pour le traitement d'images et les tests du modèle.
"""

from pathlib import Path
from typing import Optional, Union

from PIL import Image

from .config import DEFAULT_CONFIG_PATH, DEFAULT_MODEL_PATH
from .model import BirdPestClassifier, PredictionResult


def load_image(image_path: Union[str, Path]) -> Image.Image:
    """Charge une image depuis le disque et la convertit en RGB."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image introuvable : {path}")
    img = Image.open(path)
    return img.convert("RGB")


def quick_predict(
    image_path: Union[str, Path],
    model_path: Union[str, Path] = DEFAULT_MODEL_PATH,
    config_path: Union[str, Path] = DEFAULT_CONFIG_PATH,
    threshold: Optional[float] = None,
) -> PredictionResult:
    """Effectue une prédiction rapide en une ligne de code depuis un chemin d'image.

    Exemple d'utilisation :
    >>> from src.utils import quick_predict
    >>> result = quick_predict("examples/exemple_ravageur_proxy.jpg")
    >>> print(result.status_title)
    """
    img = load_image(image_path)
    classifier = BirdPestClassifier(model_path=model_path, config_path=config_path)
    return classifier.predict(img, threshold=threshold)
