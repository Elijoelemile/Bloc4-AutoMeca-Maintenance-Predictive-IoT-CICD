"""Tests fonctionnels — interface de supervision (ui/app.py) contre le
vrai service API (vrais modeles, vraie reference de derive) — pas de
mock. Necessite les vrais fichiers models/*.joblib (Git LFS) : tourne
dans une job CI dediee avec checkout LFS, pas dans la job "tests"
(volontairement legere, modeles mockes — voir test_main.py).

Utilise `AppTest` (streamlit.testing.v1), le framework de test officiel
de Streamlit : execute reellement le script ui/app.py (pas une
simulation), ce qui a permis de trouver deux bugs reels invisibles a
des tests unitaires mockes, corriges en developpement :
1. Un ticket nouvellement cree n'apparaissait pas dans l'onglet
   "Tickets" sans action supplementaire (l'onglet Tickets s'execute
   avant l'onglet Nouvelle alerte dans le script — la liste etait donc
   recuperee avant la creation du ticket).
2. `st.success(...)` suivi immediatement de `st.rerun()` effacait le
   message avant qu'il soit jamais affiche (sur Valider, Cloturer et
   Creer un ticket) — corrige via `st.session_state`.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
import requests

RACINE = Path(__file__).resolve().parent.parent
PORT_TEST = 8123
API_URL = f"http://127.0.0.1:{PORT_TEST}"
CLE_API = "cle-test-ui"


@pytest.fixture(scope="module")
def api_reelle():
    env = {**os.environ, "API_KEY": CLE_API}
    processus = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(PORT_TEST)],
        cwd=RACINE, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        for _ in range(120):  # jusqu'a 60s : le prechauffage modeles/explainers est lent (~13-27s, mesure reelle)
            try:
                if requests.get(f"{API_URL}/sante", timeout=1).ok:
                    break
            except requests.exceptions.ConnectionError:
                pass
            time.sleep(0.5)
        else:
            raise RuntimeError("L'API reelle n'a pas demarre a temps")
        yield API_URL
    finally:
        processus.terminate()
        processus.wait(timeout=10)


@pytest.fixture
def mesures_reelles():
    ref = json.load(open(RACINE / "models" / "reference_distribution.json", encoding="utf-8"))
    mesures = {}
    for modele in ["isolation_forest", "random_survival_forest"]:
        for feature, stats in ref[modele]["features"].items():
            if stats["type"] == "continue":
                mesures[feature] = stats["moyenne"]
            else:
                mesures[feature] = float(max(stats["proportions"], key=stats["proportions"].get))
    return mesures


def _app_test(api_reelle):
    from streamlit.testing.v1 import AppTest

    os.environ["API_URL"] = api_reelle
    os.environ["API_KEY"] = CLE_API
    at = AppTest.from_file(str(RACINE / "ui" / "app.py"), default_timeout=30)
    at.run()
    assert not at.exception
    return at


def test_creation_ticket_visible_immediatement_dans_onglet_tickets(api_reelle, mesures_reelles):
    at = _app_test(api_reelle)
    tab_alerte = at.tabs[1]
    tab_alerte.number_input[0].set_value(1)
    tab_alerte.text_area[0].set_value(json.dumps(mesures_reelles))
    tab_alerte.button[0].click()
    at.run()

    assert not at.exception
    assert at.tabs[1].success, "message de creation attendu dans l'onglet Nouvelle alerte"

    tab_tickets = at.tabs[0]
    assert len(tab_tickets.selectbox) == 2, "le ticket cree doit etre selectionnable sans action supplementaire"


def test_workflow_complet_criticite_elevee(api_reelle, mesures_reelles):
    mesures_derivees = dict(mesures_reelles)
    mesures_derivees["vibration_moy24h"] *= 6
    mesures_derivees["vibration_std24h"] *= 6

    at = _app_test(api_reelle)
    tab_alerte = at.tabs[1]
    tab_alerte.number_input[0].set_value(2)
    tab_alerte.text_area[0].set_value(json.dumps(mesures_derivees))
    tab_alerte.button[0].click()
    at.run()
    assert not at.exception
    assert "elevee" in at.tabs[1].success[0].value

    tab_tickets = at.tabs[0]
    bouton_valider = next(b for b in tab_tickets.button if "Valider" in b.label)
    bouton_valider.click()
    at.run()
    assert not at.exception
    assert any("validé" in s.value for s in at.tabs[0].success)

    tab_tickets = at.tabs[0]
    radio = tab_tickets.radio[0]
    radio.set_value("panne_confirmee")
    bouton_cloturer = next(b for b in tab_tickets.button if "Clôturer" in b.label)
    bouton_cloturer.click()
    at.run()
    assert not at.exception
    assert any("clôturé" in s.value for s in at.tabs[0].success)

    metriques = {m.label: m.value for m in at.tabs[2].metric}
    assert metriques["Tickets clôturés"] == "1"
    assert metriques["Précision"] == "100%"


def test_explication_shap_sans_erreur(api_reelle, mesures_reelles):
    at = _app_test(api_reelle)
    tab_alerte = at.tabs[1]
    tab_alerte.number_input[0].set_value(3)
    tab_alerte.text_area[0].set_value(json.dumps(mesures_reelles))
    tab_alerte.button[0].click()
    at.run()

    tab_tickets = at.tabs[0]
    bouton_expliquer = next(b for b in tab_tickets.button if b.label == "Calculer l'explication")
    bouton_expliquer.click()
    at.run()

    assert not at.exception
    # 1 dataframe pour les mesures brutes + 2 pour les facteurs SHAP (anomalie + RUL)
    assert len(at.tabs[0].dataframe) >= 3
