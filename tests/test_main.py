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

from app.main import FENETRE_MESURES, LATENCES_MS, TICKETS, app  # noqa: E402

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


def _explainers_factices() -> dict:
    import numpy as np

    explainer_anomalie = MagicMock()
    explainer_anomalie.shap_values.return_value = np.array([[0.9, -0.1]])
    explainer_rul = MagicMock()
    explainer_rul.shap_values.return_value = np.array([[0.2, -0.8]])
    return {"anomalie": explainer_anomalie, "rul": explainer_rul}


@pytest.fixture(autouse=True)
def _reset_tickets():
    TICKETS.clear()
    FENETRE_MESURES.clear()
    LATENCES_MS.clear()
    yield
    TICKETS.clear()
    FENETRE_MESURES.clear()
    LATENCES_MS.clear()


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


def test_lister_tickets_vide_au_depart():
    reponse = client.get("/tickets", headers=EN_TETE_AUTH)
    assert reponse.status_code == 200
    assert reponse.json() == []


def test_lister_tickets_renvoie_tous_les_tickets():
    with patch("app.main.get_modeles", return_value=_modeles_factices(anomalie_predite=1, proba_panne=0.1)):
        client.post("/predictions/ticket", json={"machine_id": 1, "mesures": {"a": 1, "b": 2}}, headers=EN_TETE_AUTH)
    with patch("app.main.get_modeles", return_value=_modeles_factices(anomalie_predite=-1, proba_panne=0.9)):
        client.post("/predictions/ticket", json={"machine_id": 2, "mesures": {"a": 1, "b": 2}}, headers=EN_TETE_AUTH)

    reponse = client.get("/tickets", headers=EN_TETE_AUTH)
    assert reponse.status_code == 200
    assert len(reponse.json()) == 2


def test_lister_tickets_filtre_par_statut():
    with patch("app.main.get_modeles", return_value=_modeles_factices(anomalie_predite=1, proba_panne=0.1)):
        client.post("/predictions/ticket", json={"machine_id": 1, "mesures": {"a": 1, "b": 2}}, headers=EN_TETE_AUTH)
    with patch("app.main.get_modeles", return_value=_modeles_factices(anomalie_predite=-1, proba_panne=0.9)):
        client.post("/predictions/ticket", json={"machine_id": 2, "mesures": {"a": 1, "b": 2}}, headers=EN_TETE_AUTH)

    reponse = client.get(
        "/tickets", params={"statut": "en_attente_validation_technicien_senior"}, headers=EN_TETE_AUTH,
    )
    corps = reponse.json()
    assert len(corps) == 1
    assert corps[0]["machine_id"] == 2


def test_lister_tickets_necessite_authentification():
    reponse = client.get("/tickets")
    assert reponse.status_code == 401


def test_expliquer_ticket_introuvable():
    reponse = client.get("/tickets/inexistant/explication", headers=EN_TETE_AUTH)
    assert reponse.status_code == 404


def test_expliquer_ticket_renvoie_les_facteurs():
    with (
        patch("app.main.get_modeles", return_value=_modeles_factices(anomalie_predite=1, proba_panne=0.1)),
        patch("app.main.get_explainers", return_value=_explainers_factices()),
    ):
        creation = client.post(
            "/predictions/ticket", json={"machine_id": 42, "mesures": {"a": 1, "b": 2}}, headers=EN_TETE_AUTH,
        )
        ticket_id = creation.json()["ticket_id"]

        explication = client.get(f"/tickets/{ticket_id}/explication", headers=EN_TETE_AUTH)

    assert explication.status_code == 200
    corps = explication.json()
    assert corps["facteurs_anomalie"][0]["feature"] == "a"
    assert corps["facteurs_rul"][0]["feature"] == "b"


def test_expliquer_ticket_necessite_authentification():
    reponse = client.get("/tickets/quelconque/explication")
    assert reponse.status_code == 401


def test_cloturer_ticket_assigne_avec_succes():
    with patch("app.main.get_modeles", return_value=_modeles_factices(anomalie_predite=1, proba_panne=0.1)):
        creation = client.post(
            "/predictions/ticket", json={"machine_id": 42, "mesures": {"a": 1, "b": 2}}, headers=EN_TETE_AUTH,
        )
    ticket_id = creation.json()["ticket_id"]

    cloture = client.post(
        f"/tickets/{ticket_id}/cloturer", json={"resultat_reel": "panne_confirmee"}, headers=EN_TETE_AUTH,
    )
    assert cloture.status_code == 200
    corps = cloture.json()
    assert corps["statut"] == "cloture"
    assert corps["resultat_reel"] == "panne_confirmee"


def test_cloturer_ticket_en_attente_validation_rejete():
    """Un ticket de criticite elevee doit d'abord passer par /valider —
    le clore directement contournerait le garde-fou "human oversight"."""
    with patch("app.main.get_modeles", return_value=_modeles_factices(anomalie_predite=-1, proba_panne=0.9)):
        creation = client.post(
            "/predictions/ticket", json={"machine_id": 42, "mesures": {"a": 1, "b": 2}}, headers=EN_TETE_AUTH,
        )
    ticket_id = creation.json()["ticket_id"]

    cloture = client.post(
        f"/tickets/{ticket_id}/cloturer", json={"resultat_reel": "fausse_alerte"}, headers=EN_TETE_AUTH,
    )
    assert cloture.status_code == 409


def test_cloturer_ticket_deja_cloture_rejete():
    with patch("app.main.get_modeles", return_value=_modeles_factices(anomalie_predite=1, proba_panne=0.1)):
        creation = client.post(
            "/predictions/ticket", json={"machine_id": 42, "mesures": {"a": 1, "b": 2}}, headers=EN_TETE_AUTH,
        )
    ticket_id = creation.json()["ticket_id"]
    client.post(f"/tickets/{ticket_id}/cloturer", json={"resultat_reel": "fausse_alerte"}, headers=EN_TETE_AUTH)

    deuxieme_cloture = client.post(
        f"/tickets/{ticket_id}/cloturer", json={"resultat_reel": "panne_confirmee"}, headers=EN_TETE_AUTH,
    )
    assert deuxieme_cloture.status_code == 409


def test_cloturer_ticket_introuvable():
    reponse = client.post(
        "/tickets/inexistant/cloturer", json={"resultat_reel": "fausse_alerte"}, headers=EN_TETE_AUTH,
    )
    assert reponse.status_code == 404


def test_monitoring_performance_sans_ticket_cloture():
    reponse = client.get("/monitoring/performance", headers=EN_TETE_AUTH)
    assert reponse.status_code == 200
    corps = reponse.json()
    assert corps["n_tickets_clotures"] == 0
    assert corps["precision"] is None


def test_monitoring_performance_apres_clotures():
    with patch("app.main.get_modeles", return_value=_modeles_factices(anomalie_predite=1, proba_panne=0.1)):
        for _ in range(3):
            creation = client.post(
                "/predictions/ticket", json={"machine_id": 1, "mesures": {"a": 1, "b": 2}}, headers=EN_TETE_AUTH,
            )
            ticket_id = creation.json()["ticket_id"]
            client.post(f"/tickets/{ticket_id}/cloturer", json={"resultat_reel": "panne_confirmee"}, headers=EN_TETE_AUTH)

    reponse = client.get("/monitoring/performance", headers=EN_TETE_AUTH)
    corps = reponse.json()
    assert corps["n_tickets_clotures"] == 3
    assert corps["precision"] == 1.0
    assert corps["latence_moyenne_ms"] is not None


def test_monitoring_derive_necessite_authentification():
    reponse = client.get("/monitoring/derive")
    assert reponse.status_code == 401


def test_monitoring_derive_fenetre_vide_pas_de_derive_signalee():
    """Fenetre vide (rien reçu depuis le demarrage) : sous le seuil de
    fenetre minimale, aucune derive ne doit etre affirmee — voir
    N_MIN_OBSERVATIONS dans app/derive.py (calibre pour eviter les
    fausses alertes a faible echantillon)."""
    reference_factice = {
        "isolation_forest": {"features": {}},
        "random_survival_forest": {"features": {}},
    }
    with (
        patch("app.main.get_reference_derive", return_value=reference_factice),
        patch("app.main.get_modeles", return_value=_modeles_factices(anomalie_predite=1, proba_panne=0.1)),
    ):
        reponse = client.get("/monitoring/derive", headers=EN_TETE_AUTH)
    assert reponse.status_code == 200
    corps = reponse.json()
    assert corps["isolation_forest"]["derive_confirmee"] is False
    assert corps["random_survival_forest"]["derive_confirmee"] is False
