"""Suivi de performance en production — exigence du sujet (Bloc 4,
"Derive et monitoring") : "precision de detection, taux de faux
positifs" surveilles en continu.

Mesure operationnelle, distincte de l'AUC/C-index calcules hors ligne
sur donnees historiques etiquetees (notebooks du depot solution-IA) :
ici, on ne dispose que des tickets effectivement crees (donc des
alertes positives), jamais des vrais negatifs (mesures normales n'ayant
jamais genere de ticket) — impossible en conditions reelles de calculer
un taux de faux positifs au sens classification (FP / vrais negatifs).
Ce qui est mesurable, et ce que ce module calcule, c'est la proportion
d'alertes levees qui se revelent fondees une fois le ticket clos par un
technicien (`POST /tickets/{id}/cloturer`) : precision = pannes
confirmees / tickets clos, taux de fausses alertes = 1 - precision.
"""
from dataclasses import dataclass
from enum import Enum


class ResultatReel(str, Enum):
    PANNE_CONFIRMEE = "panne_confirmee"
    FAUSSE_ALERTE = "fausse_alerte"


@dataclass(frozen=True)
class StatistiquesPerformance:
    n_tickets_clotures: int
    precision: float | None
    taux_fausses_alertes: float | None
    n_latences_mesurees: int
    latence_moyenne_ms: float | None


def calculer_performance(resultats_reels: list[ResultatReel], latences_ms: list[float]) -> StatistiquesPerformance:
    n = len(resultats_reels)
    if n == 0:
        precision = None
        taux_fausses_alertes = None
    else:
        vrais_positifs = sum(1 for r in resultats_reels if r == ResultatReel.PANNE_CONFIRMEE)
        precision = vrais_positifs / n
        taux_fausses_alertes = 1 - precision

    return StatistiquesPerformance(
        n_tickets_clotures=n,
        precision=precision,
        taux_fausses_alertes=taux_fausses_alertes,
        n_latences_mesurees=len(latences_ms),
        latence_moyenne_ms=(sum(latences_ms) / len(latences_ms)) if latences_ms else None,
    )
