"""
Application Gradio - Détection d'oiseaux ravageurs du riz (Vallée du fleuve Sénégal).
Compatible Hugging Face Spaces et exécution locale.
"""

import os
from pathlib import Path
from typing import Dict, Tuple

import gradio as gr
from PIL import Image

from src.config import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_THRESHOLD,
    LABEL_NON_PRIO,
    LABEL_PEST,
)
from src.model import BirdPestClassifier, PredictionResult

# Initialisation du classificateur
classifier = BirdPestClassifier(
    model_path=DEFAULT_MODEL_PATH,
    config_path=DEFAULT_CONFIG_PATH,
)


def generate_badge_html(res: PredictionResult) -> str:
    """Génère un composant HTML stylisé pour afficher l'alerte agronomique."""
    if res.is_pest:
        bg_color = "rgba(239, 68, 68, 0.12)"
        border_color = "#ef4444"
        text_color = "#b91c1c"
        badge_bg = "#dc2626"
        icon = "⚠️"
        title = "ALERTE RAVAGEUR POTENTIEL"
        badge_text = "RISQUE ÉLEVÉ POUR LES RIZIÈRES"
        advice = (
            "<strong>Action recommandée :</strong> Validation humaine visuelle requise sur la parcelle. "
            "Si confirmé, déclencher un <em>effarouchement non létal</em> (sons, épouvantails, drones d'éloignement). "
            "Ne jamais utiliser de méthodes destructrices pour la faune non cible."
        )
    else:
        bg_color = "rgba(34, 197, 94, 0.12)"
        border_color = "#22c55e"
        text_color = "#15803d"
        badge_bg = "#16a34a"
        icon = "✅"
        title = "ESPÈCE NON PRIORITAIRE"
        badge_text = "RISQUE FAIBLE / NEUTRE"
        advice = (
            "<strong>Action recommandée :</strong> Aucune intervention nécessaire. "
            "L'oiseau ne présente pas les caractéristiques d'un granivore ravageur majeur de riz."
        )

    return f"""
    <div style="
        background-color: {bg_color};
        border: 2px solid {border_color};
        border-radius: 12px;
        padding: 18px 22px;
        margin-bottom: 15px;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    ">
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 10px;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <span style="font-size: 26px;">{icon}</span>
                <span style="font-size: 19px; font-weight: 700; color: {text_color}; letter-spacing: 0.5px;">
                    {title}
                </span>
            </div>
            <span style="
                background: {badge_bg};
                color: #ffffff;
                font-size: 11px;
                font-weight: 700;
                padding: 4px 10px;
                border-radius: 9999px;
                letter-spacing: 0.5px;
                text-transform: uppercase;
            ">{badge_text}</span>
        </div>
        <div style="color: #374151; font-size: 14px; line-height: 1.5; margin-bottom: 10px;">
            {res.status_description}
        </div>
        <div style="border-top: 1px dashed {border_color}; padding-top: 10px; font-size: 13px; color: #1f2937;">
            {advice}
        </div>
    </div>
    """


def generate_details_md(res: PredictionResult) -> str:
    """Génère un tableau récapitulatif détaillé en Markdown."""
    decision_icon = "🔴 Ravageur" if res.is_pest else "🟢 Non prioritaire"
    return f"""
| Métrique | Valeur | Note explicative |
| :--- | :--- | :--- |
| **Score Ravageur** | `{res.prob_pest:.2%}` | Probabilité brute estimée par le modèle |
| **Seuil de coupure** | `{res.threshold:.2%}` | Calibré sur validation pour un Rappel ≥ 90% |
| **Décision finale** | **{decision_icon}** | Déclenchée car `Score {'≥' if res.is_pest else '<'} Seuil` |
| **Priorité de l'alerte** | `{'HAUTE' if res.is_pest else 'BASSE'}` | Évite les faux négatifs destructeurs en récolte |
"""


def analyze_bird(image: Image.Image, threshold: float) -> Tuple[str, Dict[str, float], str]:
    """Point d'entrée de la prédiction pour l'interface Gradio."""
    if image is None:
        empty_html = """
        <div style="padding: 20px; text-align: center; color: #6b7280; border: 2px dashed #d1d5db; border-radius: 10px;">
            📷 Veuillez téléverser ou capturer une photo d'oiseau pour lancer l'analyse.
        </div>
        """
        return empty_html, {}, ""

    try:
        res = classifier.predict(image, threshold=threshold)
        badge_html = generate_badge_html(res)
        details_md = generate_details_md(res)
        return badge_html, res.confidence_scores, details_md
    except Exception as e:
        error_html = f"""
        <div style="padding: 16px; background-color: #fef2f2; border: 1px solid #f87171; border-radius: 8px; color: #991b1b;">
            <strong>Erreur d'analyse :</strong> {str(e)}
        </div>
        """
        return error_html, {}, f"Une erreur s'est produite lors de l'inférence : {e}"


# Chemins des exemples d'images
EXAMPLES_DIR = Path(__file__).resolve().parent / "examples"
example_list = []
if EXAMPLES_DIR.exists():
    for ex in ["exemple_ravageur_proxy.jpg", "exemple_non_ravageur.jpg"]:
        p = EXAMPLES_DIR / ex
        if p.exists():
            example_list.append([str(p), DEFAULT_THRESHOLD])

# Interface Gradio
custom_css = """
.main-header {
    text-align: center;
    padding: 10px 0 20px 0;
}
.header-badge {
    display: inline-block;
    padding: 4px 12px;
    margin: 3px;
    font-size: 12px;
    font-weight: 600;
    border-radius: 9999px;
    background: #f3f4f6;
    color: #374151;
    border: 1px solid #e5e7eb;
}
"""

with gr.Blocks(title="Détection d'oiseaux ravageurs du riz – Sénégal", css=custom_css, theme=gr.themes.Soft()) as demo:
    with gr.Column(elem_classes=["main-header"]):
        gr.Markdown(
            """
            # 🌾 Détection d'Oiseaux Ravageurs du Riz
            ### *Système d'aide à la décision agronomique – Vallée du fleuve Sénégal*
            """
        )
        gr.HTML(
            """
            <div style="display: flex; justify-content: center; flex-wrap: wrap; gap: 6px; margin-top: 4px;">
                <span class="header-badge">🧠 MobileNetV2 (Transfer Learning)</span>
                <span class="header-badge">🎯 Objectif : Recall ≥ 90%</span>
                <span class="header-badge">⚖️ Seuil calibré : 52.5%</span>
                <span class="header-badge">🛡️ Effarouchement non létal uniquement</span>
                <span class="header-badge">👥 Human-in-the-loop</span>
            </div>
            """
        )

    with gr.Tabs():
        with gr.TabItem("🔍 Analyse & Détection", id="tab_detection"):
            with gr.Row():
                with gr.Column(scale=5):
                    input_image = gr.Image(
                        type="pil",
                        label="Photo d'oiseau (parcelle ou piège photographique)",
                        sources=["upload", "webcam", "clipboard"],
                    )

                    with gr.Accordion("⚙️ Réglage du seuil de sensibilité (Recall)", open=False):
                        threshold_slider = gr.Slider(
                            minimum=0.10,
                            maximum=0.90,
                            value=DEFAULT_THRESHOLD,
                            step=0.01,
                            label="Seuil de décision binaire",
                            info=(
                                "Un seuil à 52.5% garantit un Rappel ≥ 90% sur la validation. "
                                "Diminuer le seuil rend le système plus sensible (évite de rater un ravageur mais génère plus de fausses alertes)."
                            ),
                        )

                    with gr.Row():
                        btn_clear = gr.ClearButton(components=[input_image], value="🗑️ Réinitialiser")
                        btn_submit = gr.Button("🔎 Détecter le statut", variant="primary", scale=2)

                    if example_list:
                        gr.Examples(
                            examples=example_list,
                            inputs=[input_image, threshold_slider],
                            label="Images d'exemples pré-chargées",
                        )

                with gr.Column(scale=6):
                    alert_output = gr.HTML(
                        value="""
                        <div style="padding: 24px; text-align: center; color: #6b7280; border: 2px dashed #e5e7eb; border-radius: 12px;">
                            Sélectionnez une image ou un exemple pour afficher le résultat d'analyse et l'alerte agronomique.
                        </div>
                        """
                    )
                    confidence_output = gr.Label(
                        num_top_classes=2,
                        label="Probabilités estimées",
                    )
                    details_output = gr.Markdown(label="Détails techniques")

            btn_submit.click(
                fn=analyze_bird,
                inputs=[input_image, threshold_slider],
                outputs=[alert_output, confidence_output, details_output],
            )
            input_image.change(
                fn=analyze_bird,
                inputs=[input_image, threshold_slider],
                outputs=[alert_output, confidence_output, details_output],
            )
            threshold_slider.change(
                fn=analyze_bird,
                inputs=[input_image, threshold_slider],
                outputs=[alert_output, confidence_output, details_output],
            )

        with gr.TabItem("🌾 Contexte Agronomique", id="tab_context"):
            gr.Markdown(
                """
                ### 🌾 Le Défi des Oiseaux Déprédateurs dans la Vallée du Fleuve Sénégal
                
                La riziculture irriguée dans la **Vallée du fleuve Sénégal** (zones de Richard-Toll, Dagana, Podor, Matam) 
                constitue un pilier stratégique pour l'autosuffisance alimentaire au Sénégal. Cependant, les rizières subissent 
                des attaques massives d'oiseaux granivores déprédateurs, en particulier :
                - **Le Quelea quelea** (*Travailleur à bec rouge*) : surnommé le « criquet à plumes », capable de dévaster des hectares de cultures en quelques heures durant les phases laiteuse et pâteuse du grain de riz.
                - **Les Dendrocygnes** (*Dendrocygna viduata, Dendrocygna bicolor*) : canards siffleurs attaquant fréquemment les semis et jeunes plants en eau.
                - **Autres passereaux granivores** locaux.

                #### ⚠️ Le coût asymétrique des erreurs en agriculture
                - **Faux Négatif (Ravageur manqué) :** **Conséquence critique.** Une nuée non signalée détruit la récolte de paysans vulnérables.
                - **Faux Positif (Fausse alerte) :** **Conséquence mineure.** Un paysan ou garde vérifie la parcelle ; cela entraîne une légère fatigue d'alerte, mais préserve la récolte.
                
                👉 C'est pourquoi ce POC calibre son seuil de décision pour **garantir un taux de Rappel (Recall) supérieur à 90 %**.
                """
            )

        with gr.TabItem("📊 Modèle & Architecture", id="tab_model"):
            gr.Markdown(
                """
                ### 🧠 Architecture Technique du Modèle
                
                1. **Base pré-entraînée :** `MobileNetV2` (poids ImageNet). Ce choix garantit un modèle très léger (~21 Mo), rapide en inférence (~15 ms par image) et facilement déployable sur des micro-ordinateurs embarqués ou des pièges photographiques solaires (Raspberry Pi, IoT).
                2. **Pipeline d'augmentation intégré au modèle :** Flip horizontal, rotation (±10%), zoom (±15%), contraste (±20%). Actif uniquement lors de l'entraînement (`fit`), sans surcharger l'inférence.
                3. **Stratégie d'entraînement en 2 phases :**
                   - *Phase 1 :* Transfert d'apprentissage avec base gelée (`Adam lr=1e-3`, gestion du déséquilibre de classes via `class_weight`).
                   - *Phase 2 :* Fine-tuning chirurgical des 30 dernières couches de MobileNetV2 avec un faible taux d'apprentissage (`Adam lr=1e-5`).
                4. **Gestion du fort déséquilibre de classes :** Les ravageurs ne représentent qu'environ **7,5 %** du dataset CUB-200. Des poids de classes inverses (`class_weight`) ont été appliqués.
                5. **Calibrage strict du seuil :** Le seuil `≈ 0.525` a été sélectionné **exclusivement sur le jeu de validation** pour maximiser la précision sous contrainte stricte de `Recall ≥ 90 %` (évitant tout data leakage avec le jeu de test).
                """
            )

        with gr.TabItem("⚖️ Éthique, Limites & Bonnes Pratiques", id="tab_ethics"):
            gr.Markdown(
                """
                ### 🛡️ Cadre Déontologique, Limites et Perspectives
                
                #### 1. Limite des proxys de données (Domain Shift)
                Le jeu de données d'entraînement `CUB-200-2011` rassemble des oiseaux nord-américains photographiés en haute définition. 
                Ne disposant pas du *Quelea quelea* africain, des **espèces proxys granivores** (carouges, bruants, pinsons, grosbecs) 
                ont été ciblées. 
                > **Important :** En conditions réelles au Sénégal, les arrière-plans, la poussière sahelienne et la morphologie exacte 
                > peuvent dégrader les performances. Une phase d'annotation locale est indispensable avant déploiement opérationnel.

                #### 2. Règle du "Human-in-the-Loop" (Validation Humaine Obligatoire)
                Ce modèle est un **outil d'aide à la décision**, et non un système de déclenchement autonome. Chaque alerte doit 
                être vérifiée visuellement par l'agriculteur ou le surveillant de périmètre.

                #### 3. Protection de la Biodiversité et Lutte Non Létale
                Ce projet s'inscrit rigoureusement dans une démarche agroécologique :
                - ❌ **Interdiction formelle** de promouvoir des poisons chimiques (fenthion, avicides) ou des méthodes létales aveugles polluant les nappes phréatiques et décimant les rapaces et hérons non cibles.
                - ✅ **Encouragement strict des techniques d'effarouchement non létales :** canons effaroucheurs à gaz, drones munis de haut-parleurs répulsifs, filets de protection physique sur pépinières, lasers biomimétiques.

                #### 4. Perspectives d'évolution
                - Conversion du modèle en **TensorFlow Lite (TFLite) quantifié INT8** pour exécution hors-ligne sur caméra piège autonome.
                - Constitution d'un dataset ouvert ouest-africain via crowdsourcing (iNaturalist / parcelles SAED du fleuve Sénégal).
                - Système d'alertes instantanées (SMS / WhatsApp) pour les surveillants de digues.
                """
            )

    gr.Markdown(
        """
        ---
        <div style="text-align: center; font-size: 12px; color: #9ca3af;">
            POC Détection d'Oiseaux Ravageurs – Modèle MobileNetV2 Keras 3 – Adapté pour Hugging Face Spaces & Edge AI.
        </div>
        """
    )

if __name__ == "__main__":
    demo.launch()
