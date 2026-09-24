"""Tests unitaires — app/performance.py"""
from app.performance import ResultatReel, calculer_performance


def test_aucun_ticket_cloture_renvoie_none():
    stats = calculer_performance([], [])
    assert stats.n_tickets_clotures == 0
    assert stats.precision is None
    assert stats.taux_fausses_alertes is None
    assert stats.latence_moyenne_ms is None


def test_precision_et_taux_fausses_alertes():
    resultats = [ResultatReel.PANNE_CONFIRMEE] * 3 + [ResultatReel.FAUSSE_ALERTE] * 1
    stats = calculer_performance(resultats, [])
    assert stats.n_tickets_clotures == 4
    assert stats.precision == 0.75
    assert stats.taux_fausses_alertes == 0.25


def test_toutes_fausses_alertes():
    resultats = [ResultatReel.FAUSSE_ALERTE] * 5
    stats = calculer_performance(resultats, [])
    assert stats.precision == 0.0
    assert stats.taux_fausses_alertes == 1.0


def test_latence_moyenne():
    stats = calculer_performance([], [100.0, 200.0, 300.0])
    assert stats.n_latences_mesurees == 3
    assert stats.latence_moyenne_ms == 200.0


def test_aucune_latence_mesuree_renvoie_none():
    stats = calculer_performance([], [])
    assert stats.latence_moyenne_ms is None
