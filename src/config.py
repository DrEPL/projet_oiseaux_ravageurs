"""
Configuration et constantes pour le classificateur d'oiseaux ravageurs.
Projet : Détection d'oiseaux ravageurs du riz – Vallée du fleuve Sénégal.
"""

from pathlib import Path

# Chemins relatifs par défaut
BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = BASE_DIR / "bird_pest_mobilenetv2.keras"
DEFAULT_CONFIG_PATH = BASE_DIR / "config.json"

# Paramètres de prétraitement image
IMAGE_SIZE = (224, 224)
INPUT_SHAPE = (224, 224, 3)

# Seuil de décision par défaut (calibré sur validation pour Recall >= 90%)
DEFAULT_THRESHOLD = 0.5252

# Libellés des classes
LABEL_PEST = "Ravageur potentiel"
LABEL_NON_PRIO = "Non prioritaire"

# Proxys biologiques (classes issues de CUB-200-2011 simulant les granivores comme le Quelea)
# Note : CUB-200 ne dispose pas du Quelea quelea (Travailleur à bec rouge).
# Des passereaux granivores apparentés ont été retenus comme proxys.
PROXY_SPECIES_IDS = [8, 9, 10, 11, 12, 13, 14, 15, 16, 33, 34, 46, 47, 53, 56]
PROXY_SPECIES_FAMILIES = [
    "Carouges (Blackbirds / Icteridae)",
    "Bruants & Pinsons (Sparrows & Finches / Passerellidae)",
    "Chardonnerets (Goldfinches / Fringillidae)",
    "Grosbecs (Grosbeaks / Cardinalidae)",
    "Bobolinks (Dolichonyx oryzivorus)"
]

# Messages d'alerte et recommandations agronomiques
ALERT_MESSAGES = {
    "pest": {
        "title": "ALERTE : Ravageur potentiel détecté",
        "description": (
            "Risque potentiel pour la parcelle rizicole. Une validation humaine visuelle "
            "est requise avant d'activer des méthodes d'effarouchement non létales."
        ),
        "level": "warning"
    },
    "non_prio": {
        "title": "Espèce non prioritaire / Risque faible",
        "description": (
            "Probabilité faible d'impact sur les cultures de riz. "
            "Aucune intervention recommandée à ce stade."
        ),
        "level": "safe"
    }
}
