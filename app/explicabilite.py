"""Explicabilite des alertes (SHAP) — exigence du sujet (Bloc 4,
"Conformite") : "methodes type SHAP permettant aux techniciens de
comprendre les facteurs declencheurs".

Deux explainers, un par modele (meme choix que dans les notebooks du
depot solution-IA — voir notebooks/01 et 02, section 5.3) :
- `TreeExplainer` pour l'IsolationForest (rapide, natif aux arbres).
- `KernelExplainer` pour le RandomSurvivalForest (`TreeExplainer` leve
  `InvalidModelError` sur ce modele, deja constate en notebook).

Cout : construire un explainer est lent (~13s pour le TreeExplainer sur
300 arbres, mesure reelle en developpement) mais c'est un cout
d'initialisation ponctuel — les deux explainers sont donc caches en
memoire (meme principe que `get_modeles()` dans main.py) et reutilises
a chaque requete. Une fois caches, le cout marginal par requete est
mesure a ~40ms (anomalie) et ~350-400ms (RUL, `KernelExplainer` etant
plus couteux par nature) — verifie manuellement, acceptable pour un
usage interactif (interface de supervision), pas pour un flux temps
reel haute frequence.
"""
from dataclasses import dataclass

import numpy as np
import shap

NB_FACTEURS_AFFICHES = 5


@dataclass(frozen=True)
class ContributionFeature:
    feature: str
    valeur: float
    contribution_shap: float


@dataclass(frozen=True)
class ResultatExplication:
    facteurs_anomalie: list[ContributionFeature]
    facteurs_rul: list[ContributionFeature]


def construire_explainer_anomalie(modeles: dict):
    return shap.TreeExplainer(modeles["anomalie"]["modele"])


def construire_explainer_rul(modeles: dict):
    # echantillon de fond reduit (10 points nuls, dans l'espace standardise
    # ou 0 = valeur moyenne d'entrainement) : suffisant pour KernelExplainer
    # et garde le cout d'initialisation et par requete faible, voir le
    # docstring du module pour les temps mesures.
    nb_features = len(modeles["rul"]["colonnes_features"])
    fond = np.zeros((10, nb_features))
    return shap.KernelExplainer(modeles["rul"]["modele"].predict, fond)


def _top_contributions(valeurs_shap: np.ndarray, valeurs_brutes: list[float], colonnes: list[str]) -> list[ContributionFeature]:
    contributions = [
        ContributionFeature(feature=col, valeur=float(v), contribution_shap=float(s))
        for col, v, s in zip(colonnes, valeurs_brutes, valeurs_shap)
    ]
    contributions.sort(key=lambda c: abs(c.contribution_shap), reverse=True)
    return contributions[:NB_FACTEURS_AFFICHES]


def expliquer(modeles: dict, explainers: dict, mesures: dict[str, float]) -> ResultatExplication:
    from app.criticite import _completer_features_manquantes

    colonnes_anomalie = modeles["anomalie"]["colonnes_features"]
    valeurs_anomalie, _ = _completer_features_manquantes(mesures, colonnes_anomalie)
    X_anomalie = modeles["anomalie"]["scaler"].transform([valeurs_anomalie])
    shap_anomalie = explainers["anomalie"].shap_values(X_anomalie)[0]

    colonnes_rul = modeles["rul"]["colonnes_features"]
    valeurs_rul, _ = _completer_features_manquantes(mesures, colonnes_rul)
    X_rul = modeles["rul"]["scaler"].transform([valeurs_rul])
    shap_rul = explainers["rul"].shap_values(X_rul, nsamples=50, silent=True)[0]

    return ResultatExplication(
        facteurs_anomalie=_top_contributions(shap_anomalie, valeurs_anomalie, colonnes_anomalie),
        facteurs_rul=_top_contributions(shap_rul, valeurs_rul, colonnes_rul),
    )
