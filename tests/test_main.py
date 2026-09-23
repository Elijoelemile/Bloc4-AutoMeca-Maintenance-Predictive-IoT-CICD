"""Tests unitaires — app/main.py

Modeles mockes (voir test_criticite.py) : on teste ici le routage HTTP,
l'authentification et le garde-fou de validation humaine, pas les
predictions elles-memes.
"""
import os
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

os.environ["API_KEY"] = "cle-de-test"

from app.main import TICKETS, app  # noqa: E402

client = TestClient(app)

EN_TETE_AUTH = {"X-API-Key": "cle-de-test"}


def _modeles_factices(anomalie_predite: int, proba_panne: float) -> dict:
    modele_anomalie = MagicMock()
    modele_anomalie.predict.return_value = [anomalie_predite]

    fonction_survie = MagicMock(return_value=1 - proba_panne)
    modele_rul = MagicMock()
    modele_rul.predict_survival_function.return_value = [fonction_survie]

    scaler_identite = MagicMock()
    scaler_identite.transform.side_effect = lambda X: X

    colonnes = ["a", "b"]
    return {
        "anomalie": {"modele": modele_anomalie, "scaler": scaler_identite, "colonnes_features": colonnes},
        "rul": {"modele": modele_rul, "scaler": scaler_identite, "colonnes_features": colonnes},
    }


@pytest.fixture(autouse=True)
def _reset_tickets():
    TICKETS.clear()
    yield
    TICKETS.clear()


def test_sante_ne_necessite_pas_authentification():
    reponse = client.get("/sante")
    assert reponse.status_code == 200
    assert reponse.json() == {"statut": "ok"}


def test_creer_ticket_sans_cle_api_rejete():
    reponse = client.post("/predictions/ticket", json={"machine_id": 1, "mesures": {"a": 1, "b": 2}})
    assert reponse.status_code == 401


def test_creer_ticket_criticite_standard_assigne_directement():
    with patch("app.main.get_modeles", return_value=_modeles_factices(anomalie_predite=1, proba_panne=0.1)):
        reponse = client.post(
            "/predictions/ticket", json={"machine_id": 42, "mesures": {"a": 1, "b": 2}}, headers=EN_TETE_AUTH,
        )
    assert reponse.status_code == 200
    corps = reponse.json()
    assert corps["criticite"] == "standard"
    assert corps["statut"] == "assigne_equipe_maintenance"


def test_creer_ticket_criticite_elevee_en_attente_validation():
    with patch("app.main.get_modeles", return_value=_modeles_factices(anomalie_predite=-1, proba_panne=0.9)):
        reponse = client.post(
            "/predictions/ticket", json={"machine_id": 42, "mesures": {"a": 1, "b": 2}}, headers=EN_TETE_AUTH,
        )
    assert reponse.status_code == 200
    corps = reponse.json()
    assert corps["criticite"] == "elevee"
    assert corps["statut"] == "en_attente_validation_technicien_senior"


def test_garde_fou_ticket_eleve_ne_peut_pas_etre_execute_sans_validation():
    """Le coeur du garde-fou "human oversight" : un ticket eleve reste en
    attente tant que /valider n'a pas ete appele explicitement."""
    with patch("app.main.get_modeles", return_value=_modeles_factices(anomalie_predite=-1, proba_panne=0.9)):
        creation = client.post(
            "/predictions/ticket", json={"machine_id": 42, "mesures": {"a": 1, "b": 2}}, headers=EN_TETE_AUTH,
        )
    ticket_id = creation.json()["ticket_id"]

    lecture = client.get(f"/tickets/{ticket_id}", headers=EN_TETE_AUTH)
    assert lecture.json()["statut"] == "en_attente_validation_technicien_senior"

    validation = client.post(f"/tickets/{ticket_id}/valider", headers=EN_TETE_AUTH)
    assert validation.status_code == 200
    assert validation.json()["statut"] == "assigne_equipe_maintenance"


def test_valider_ticket_deja_assigne_rejete():
    with patch("app.main.get_modeles", return_value=_modeles_factices(anomalie_predite=1, proba_panne=0.1)):
        creation = client.post(
            "/predictions/ticket", json={"machine_id": 42, "mesures": {"a": 1, "b": 2}}, headers=EN_TETE_AUTH,
        )
    ticket_id = creation.json()["ticket_id"]

    validation = client.post(f"/tickets/{ticket_id}/valider", headers=EN_TETE_AUTH)
    assert validation.status_code == 409


def test_ticket_introuvable():
    reponse = client.get("/tickets/inexistant", headers=EN_TETE_AUTH)
    assert reponse.status_code == 404


def test_degradation_capteur_renvoie_422_pas_un_crash():
    from app.criticite import DegradationCapteurError

    with patch("app.main.get_modeles", side_effect=None) as mock_get:
        mock_get.return_value = _modeles_factices(anomalie_predite=1, proba_panne=0.1)
        with patch("app.main.evaluer_criticite", side_effect=DegradationCapteurError("2/2 mesures manquantes")):
            reponse = client.post(
                "/predictions/ticket", json={"machine_id": 42, "mesures": {}}, headers=EN_TETE_AUTH,
            )
    assert reponse.status_code == 422
