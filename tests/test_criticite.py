"""Tests unitaires — app/criticite.py

Modeles mockes (pas de dependance aux vrais fichiers .joblib, deja
verifies manuellement contre les vrais modeles — voir le journal de
developpement). On teste ici la logique de routage, la degradation
gracieuse et le garde-fou de fiabilite.
"""
from unittest.mock import MagicMock

import pytest

from app.criticite import (
    SEUIL_PROBA_PANNE_7J,
    DegradationCapteurError,
    evaluer_criticite,
    predire_anomalie,
    predire_proba_panne_7j,
)

COLONNES = ["a", "b", "c"]


def _modeles_factices(anomalie_predite: int, proba_panne: float) -> dict:
    modele_anomalie = MagicMock()
    modele_anomalie.predict.return_value = [anomalie_predite]  # -1 = anomalie, 1 = normal

    fonction_survie = MagicMock(return_value=1 - proba_panne)
    modele_rul = MagicMock()
    modele_rul.predict_survival_function.return_value = [fonction_survie]

    scaler_identite = MagicMock()
    scaler_identite.transform.side_effect = lambda X: X

    return {
        "anomalie": {"modele": modele_anomalie, "scaler": scaler_identite, "colonnes_features": COLONNES},
        "rul": {"modele": modele_rul, "scaler": scaler_identite, "colonnes_features": COLONNES},
    }


MESURES_COMPLETES = {"a": 1.0, "b": 2.0, "c": 3.0}


def test_predire_anomalie_detectee():
    modeles = _modeles_factices(anomalie_predite=-1, proba_panne=0.0)
    anomalie, degrade = predire_anomalie(modeles, MESURES_COMPLETES)
    assert anomalie is True
    assert degrade is False


def test_predire_anomalie_normale():
    modeles = _modeles_factices(anomalie_predite=1, proba_panne=0.0)
    anomalie, _ = predire_anomalie(modeles, MESURES_COMPLETES)
    assert anomalie is False


def test_predire_proba_panne_7j():
    modeles = _modeles_factices(anomalie_predite=1, proba_panne=0.42)
    proba, degrade = predire_proba_panne_7j(modeles, MESURES_COMPLETES)
    assert proba == pytest.approx(0.42)
    assert degrade is False


def test_criticite_elevee_si_anomalie_meme_sans_risque_rul():
    modeles = _modeles_factices(anomalie_predite=-1, proba_panne=0.0)
    resultat = evaluer_criticite(modeles, MESURES_COMPLETES)
    assert resultat.criticite == "elevee"


def test_criticite_elevee_si_proba_panne_au_dessus_du_seuil_meme_sans_anomalie():
    modeles = _modeles_factices(anomalie_predite=1, proba_panne=SEUIL_PROBA_PANNE_7J + 0.01)
    resultat = evaluer_criticite(modeles, MESURES_COMPLETES)
    assert resultat.criticite == "elevee"


def test_criticite_standard_si_ni_anomalie_ni_risque():
    modeles = _modeles_factices(anomalie_predite=1, proba_panne=SEUIL_PROBA_PANNE_7J - 0.01)
    resultat = evaluer_criticite(modeles, MESURES_COMPLETES)
    assert resultat.criticite == "standard"


def test_degradation_gracieuse_capteur_manquant():
    modeles = _modeles_factices(anomalie_predite=1, proba_panne=0.1)
    mesures_incompletes = {"a": 1.0, "b": 2.0}  # "c" manquant
    resultat = evaluer_criticite(modeles, mesures_incompletes)
    assert resultat.degrade is True
    assert resultat.criticite == "standard"  # la prediction continue quand meme


def test_trop_de_capteurs_manquants_leve_une_exception():
    modeles = _modeles_factices(anomalie_predite=1, proba_panne=0.1)
    mesures_tres_incompletes = {"a": 1.0}  # 2 des 3 features manquantes (>50%)
    with pytest.raises(DegradationCapteurError):
        evaluer_criticite(modeles, mesures_tres_incompletes)
