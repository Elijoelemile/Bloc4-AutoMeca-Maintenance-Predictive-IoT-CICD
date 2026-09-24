"""Tests de non-regression du modele — exigence explicite du sujet
(section Bloc 4, "Integration et CI/CD").

Lit uniquement models/metriques.json (petit fichier JSON, suivi en Git
normal) — pas les fichiers .joblib (Git LFS, 196 Mo) : ce test tourne
donc dans la job CI la plus legere, sans avoir besoin de recuperer les
gros fichiers LFS. Si quelqu'un reentraine et commite un modele moins
performant que ce plancher, la CI echoue avant le build de l'image
Docker.

Seuils fixes avec une marge sous la performance reellement obtenue
(0.932 AUC / 0.784 C-index, voir models/metriques.json) — assez bas
pour ne pas etre fragile a une legere variation d'entrainement, assez
haut pour detecter une vraie regression.
"""
import json
from pathlib import Path

METRIQUES_PATH = Path(__file__).resolve().parent.parent / "models" / "metriques.json"

SEUIL_MIN_AUC_ANOMALIE = 0.85
SEUIL_MIN_C_INDEX_RUL = 0.70


def _charger_metriques() -> dict:
    with open(METRIQUES_PATH, encoding="utf-8") as f:
        return json.load(f)


def test_metriques_json_existe_et_est_lisible():
    metriques = _charger_metriques()
    assert "isolation_forest" in metriques
    assert "random_survival_forest" in metriques


def test_isolation_forest_ne_regresse_pas():
    metriques = _charger_metriques()["isolation_forest"]
    assert metriques["auc_cv_moyen"] >= SEUIL_MIN_AUC_ANOMALIE, (
        f"AUC moyen (validation croisee) {metriques['auc_cv_moyen']:.4f} "
        f"sous le seuil minimum {SEUIL_MIN_AUC_ANOMALIE} — regression du modele detectee"
    )


def test_random_survival_forest_ne_regresse_pas():
    metriques = _charger_metriques()["random_survival_forest"]
    assert metriques["c_index_cv_moyen"] >= SEUIL_MIN_C_INDEX_RUL, (
        f"C-index moyen (validation croisee) {metriques['c_index_cv_moyen']:.4f} "
        f"sous le seuil minimum {SEUIL_MIN_C_INDEX_RUL} — regression du modele detectee"
    )
