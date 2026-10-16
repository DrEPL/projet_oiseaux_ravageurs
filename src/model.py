"""
Module d'inférence pour la classification binaire d'oiseaux ravageurs.
Architecture : MobileNetV2 fine-tuné avec tête de classification binaire.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
from PIL import Image

from .config import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_THRESHOLD,
    IMAGE_SIZE,
    LABEL_NON_PRIO,
    LABEL_PEST,
)


@dataclass
class PredictionResult:
    """Structure standardisée du résultat d'une prédiction."""
    prob_pest: float
    prob_non_prio: float
    is_pest: bool
    threshold: float
    status_label: str
    status_title: str
    status_description: str
    alert_level: str  # 'warning' | 'safe'
    confidence_scores: Dict[str, float]


class BirdPestClassifier:
    """Classificateur d'oiseaux ravageurs basé sur MobileNetV2.

    Gère le chargement du modèle Keras, la lecture du seuil calibré,
    le prétraitement des images et la décision orientée haut Rappel.
    """

    def __init__(
        self,
        model_path: Union[str, Path] = DEFAULT_MODEL_PATH,
        config_path: Union[str, Path] = DEFAULT_CONFIG_PATH,
    ):
        self.model_path = Path(model_path)
        self.config_path = Path(config_path)
        self.threshold = self._load_threshold()
        self._model = None

    def _load_threshold(self) -> float:
        """Charge le seuil de coupure depuis le fichier config.json."""
        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return float(data.get("threshold", DEFAULT_THRESHOLD))
            except Exception as e:
                print(f"[Avertissement] Erreur de lecture de {self.config_path}: {e}")
        return DEFAULT_THRESHOLD

    @property
    def model(self):
        """Chargement paresseux (lazy loading) du modèle Keras."""
        if self._model is None:
            self.load_model()
        return self._model

    def load_model(self):
        """Charge le modèle Keras sérialisé depuis le disque."""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Modèle introuvable à l'emplacement : {self.model_path}. "
                "Assurez-vous que le fichier bird_pest_mobilenetv2.keras est présent."
            )

        import tensorflow as tf

        try:
            # Chargement standard Keras 3
            self._model = tf.keras.models.load_model(str(self.model_path))
        except Exception as e:
            # Fallback en cas de soucis de désérialisation
            raise RuntimeError(
                f"Impossible de charger le modèle Keras ({self.model_path}): {e}\n"
                "Vérifiez que tensorflow>=2.16 et keras>=3.0 sont installés."
            ) from e

    def preprocess(self, image: Union[Image.Image, np.ndarray]) -> np.ndarray:
        """Prétraite une image en entrée pour le modèle MobileNetV2.

        - Convertit en image PIL RGB si nécessaire
        - Redimensionne en 224x224
        - Convertit en tenseur float32 (valeurs [0, 255])
        - Ajoute la dimension de batch : shape (1, 224, 224, 3)
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype(np.uint8))

        if not isinstance(image, Image.Image):
            raise TypeError(f"Type d'image non supporté : {type(image)}. Utilisez PIL.Image ou np.ndarray.")

        # Conversion RGB pour gérer les images RGBA ou niveaux de gris
        rgb_image = image.convert("RGB")
        resized_image = rgb_image.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)

        # L'architecture inclut déjà la couche Rescaling(1/127.5, offset=-1)
        tensor = np.asarray(resized_image, dtype=np.float32)[None, ...]
        return tensor

    def predict(
        self,
        image: Union[Image.Image, np.ndarray],
        threshold: Optional[float] = None,
    ) -> PredictionResult:
        """Exécute l'inférence sur une image d'oiseau.

        Args:
            image: Image d'entrée (PIL.Image ou tableau NumPy).
            threshold: Seuil personnalisé optionnel. Par défaut, utilise le seuil
                       calibré sur validation (≈ 0.525).

        Returns:
            PredictionResult contenant les scores et recommandations.
        """
        thr = float(threshold) if threshold is not None else self.threshold
        tensor = self.preprocess(image)

        # Inférence
        raw_pred = self.model.predict(tensor, verbose=0)
        p_pest = float(raw_pred[0, 0])
        p_non_prio = 1.0 - p_pest

        # Décision binaire basée sur le seuil calibré pour le recall
        is_pest = p_pest >= thr

        if is_pest:
            status_label = LABEL_PEST
            status_title = "⚠️ ALERTE : Espèce à risque / Ravageur potentiel"
            status_desc = (
                f"La probabilité estimée ({p_pest:.1%}) dépasse le seuil d'alerte ({thr:.1%}). "
                "Risque pour les rizières. Une vérification visuelle par l'opérateur est requise "
                "avant tout effarouchement non létal."
            )
            alert_level = "warning"
        else:
            status_label = LABEL_NON_PRIO
            status_title = "✅ Espèce non prioritaire / Risque faible"
            status_desc = (
                f"La probabilité estimée ({p_pest:.1%}) est sous le seuil d'alerte ({thr:.1%}). "
                "Aucun comportement ravageur majeur suspecté sur la récolte."
            )
            alert_level = "safe"

        confidence_scores = {
            LABEL_PEST: p_pest,
            LABEL_NON_PRIO: p_non_prio,
        }

        return PredictionResult(
            prob_pest=p_pest,
            prob_non_prio=p_non_prio,
            is_pest=is_pest,
            threshold=thr,
            status_label=status_label,
            status_title=status_title,
            status_description=status_desc,
            alert_level=alert_level,
            confidence_scores=confidence_scores,
        )
