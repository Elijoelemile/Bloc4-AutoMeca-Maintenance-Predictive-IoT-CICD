"""Tests unitaires — app/explicabilite.py

Explainers mockes (la construction reelle d'un TreeExplainer/
KernelExplainer est lente et deja verifiee manuellement contre le
service reel — voir le journal de developpement : ~13s de construction
en premier appel, ~40-400ms une fois mis en cache, d'ou le
prechauffage au demarrage dans app/main.py). On teste ici la logique
de tri/selection des facteurs, pas SHAP lui-meme.
"""
import numpy as np
from unittest.mock import MagicMock

from app.explicabilite import NB_FACTEURS_AFFICHES, _top_contributions, expliquer

COLONNES = ["a", "b", "c", "d", "e", "f", "g"]


def test_top_contributions_trie_par_valeur_absolue_decroissante():
    shap = np.array([0.1, -0.9, 0.5, -0.05, 0.3, 0.2, -0.6])
    valeurs = [1.0] * len(COLONNES)
    top = _top_contributions(shap, valeurs, COLONNES)
    assert [c.feature for c in top] == ["b", "g", "c", "e", "f"]


def test_top_contributions_limite_au_nombre_configure():
    shap = np.array(list(range(len(COLONNES))), dtype=float)
    valeurs = [1.0] * len(COLONNES)
    top = _top_contributions(shap, valeurs, COLONNES)
    assert len(top) == NB_FACTEURS_AFFICHES


def _modeles_et_explainers_factices():
    scaler_identite = MagicMock()
    scaler_identite.transform.side_effect = lambda X: X

    modeles = {
        "anomalie": {"scaler": scaler_identite, "colonnes_features": ["a", "b"]},
        "rul": {"scaler": scaler_identite, "colonnes_features": ["a", "b", "c"]},
    }

    explainer_anomalie = MagicMock()
    explainer_anomalie.shap_values.return_value = np.array([[0.9, -0.1]])

    explainer_rul = MagicMock()
    explainer_rul.shap_values.return_value = np.array([[0.2, -0.8, 0.1]])

    explainers = {"anomalie": explainer_anomalie, "rul": explainer_rul}
    return modeles, explainers


def test_expliquer_renvoie_les_deux_jeux_de_facteurs():
    modeles, explainers = _modeles_et_explainers_factices()
    resultat = expliquer(modeles, explainers, {"a": 1.0, "b": 2.0, "c": 3.0})

    assert resultat.facteurs_anomalie[0].feature == "a"  # plus forte contribution absolue
    assert resultat.facteurs_rul[0].feature == "b"


def test_expliquer_gere_une_mesure_manquante():
    modeles, explainers = _modeles_et_explainers_factices()
    # "c" absent : ne doit pas planter, imputee comme dans criticite.py
    resultat = expliquer(modeles, explainers, {"a": 1.0, "b": 2.0})
    assert len(resultat.facteurs_rul) == 3
