"""Tests unitaires — app/derive.py

Reference synthetique (pas le vrai reference_distribution.json, deja
verifie manuellement contre le service reel — voir le journal de
developpement, notamment la correction des deciles dupliques et de la
distinction categorique/continue). On teste ici la mecanique de binning
et le seuil de fenetre minimale.
"""
from collections import deque

import pytest

from app.derive import (
    N_MIN_OBSERVATIONS,
    SEUIL_PSI_DERIVE_SIGNIFICATIVE,
    _bornes_ponderees,
    _psi_categorique,
    _psi_continue,
    detecter_derive,
)

REFERENCE = {
    "isolation_forest": {
        "features": {
            "capteur": {
                "type": "continue",
                "quantiles_10_90": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            },
            "indicatrice": {
                "type": "categorique",
                "proportions": {"0.0": 0.8, "1.0": 0.2},
            },
        }
    }
}


def test_bornes_ponderees_fusionne_les_deciles_dupliques():
    # 8 deciles a 0, un decile a 1 (cas reel : indicatrice de modele tres deséquilibrée)
    bornes, poids = _bornes_ponderees([0.0] * 8 + [1.0])
    assert bornes == [float("-inf"), 0.0, 1.0, float("inf")]
    assert poids[0] == pytest.approx(0.8)  # les 8 deciles fusionnes portent 80% de la masse de reference
    assert sum(poids) == pytest.approx(1.0)


def test_psi_continue_nul_quand_echantillon_suit_la_reference():
    # un point pile dans chaque decile de reference : distribution == reference
    quantiles = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    valeurs = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5] * 20  # 10 valeurs, une par bin, repetees
    assert _psi_continue(valeurs, quantiles) < 0.01


def test_psi_continue_eleve_quand_echantillon_concentre_hors_reference():
    quantiles = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    valeurs = [100.0] * 200  # tout au-dela du dernier decile
    assert _psi_continue(valeurs, quantiles) > SEUIL_PSI_DERIVE_SIGNIFICATIVE


def test_psi_categorique_nul_quand_proportions_identiques():
    proportions = {"0.0": 0.8, "1.0": 0.2}
    valeurs = [0.0] * 80 + [1.0] * 20
    assert _psi_categorique(valeurs, proportions) < 0.01


def test_psi_categorique_eleve_quand_proportions_inversees():
    proportions = {"0.0": 0.8, "1.0": 0.2}
    valeurs = [0.0] * 20 + [1.0] * 80
    assert _psi_categorique(valeurs, proportions) > SEUIL_PSI_DERIVE_SIGNIFICATIVE


def test_psi_categorique_valeur_jamais_vue_a_l_entrainement():
    proportions = {"0.0": 0.8, "1.0": 0.2}
    valeurs = [0.0] * 100 + [5.0] * 100  # "5.0" absente de la reference
    assert _psi_categorique(valeurs, proportions) > SEUIL_PSI_DERIVE_SIGNIFICATIVE


def test_detecter_derive_sous_le_seuil_de_fenetre_minimale():
    fenetre = deque([{"capteur": 5.0, "indicatrice": 0.0}] * (N_MIN_OBSERVATIONS - 1))
    resultat = detecter_derive(fenetre, REFERENCE, "isolation_forest", ["capteur", "indicatrice"])
    assert resultat.n_observations == N_MIN_OBSERVATIONS - 1
    assert resultat.features_en_derive == []
    assert resultat.derive_confirmee is False


def test_detecter_derive_confirmee_sur_fenetre_pleine_et_derivee():
    fenetre = deque([{"capteur": 100.0, "indicatrice": 0.0}] * N_MIN_OBSERVATIONS)
    resultat = detecter_derive(fenetre, REFERENCE, "isolation_forest", ["capteur", "indicatrice"])
    assert resultat.derive_confirmee is True
    assert resultat.reentrainement_recommande is True
    assert "capteur" in [f.feature for f in resultat.features_en_derive]


def test_detecter_derive_ignore_les_observations_incompletes():
    # certaines observations de la fenetre n'ont pas toutes les colonnes
    # (mesure de capteur degradee/manquante cote criticite.py) : ne doit
    # pas planter, simplement ignorer ces observations pour la feature
    # concernee.
    fenetre = deque(
        [{"capteur": 5.0, "indicatrice": 0.0} for _ in range(N_MIN_OBSERVATIONS - 5)]
        + [{"indicatrice": 0.0} for _ in range(5)]  # "capteur" absent
    )
    resultat = detecter_derive(fenetre, REFERENCE, "isolation_forest", ["capteur", "indicatrice"])
    assert resultat.n_observations == N_MIN_OBSERVATIONS
