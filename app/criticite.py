"""Calcul de la criticite d'une alerte, a partir des deux modeles du
depot solution-IA (detection d'anomalies + RUL, copies dans models/ —
voir le README pour la discussion "pas de dependance technique entre
depots").

Ne declenche pas l'alerte elle-meme (role du monitoring, cf. Bloc 3) :
classe une alerte deja levee en "standard" (assignation directe a
l'equipe maintenance) ou "elevee" (validation humaine obligatoire par
un technicien senior avant toute action corrective — garde-fou
"human oversight" explicitement exige par le sujet).

Degradation gracieuse : une mesure de capteur manquante ne fait jamais
planter la prediction (imputee, signalee via `degrade=True`) — sauf si
trop de mesures manquent a la fois, auquel cas la prediction ne serait
plus fiable et une exception est levee plutot que de renvoyer un
resultat silencieusement faux.
"""
from dataclasses import dataclass
from pathlib import Path

import joblib

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# Seuil calcule sur les donnees de validation reelles du modele RUL
# (85e percentile de P(panne sous 7 jours) parmi les observations
# evaluees dans le depot solution-IA, notebooks/02_prediction_rul.ipynb
# — ~15% des observations flaguees, ~38% des vraies pannes couvertes).
SEUIL_PROBA_PANNE_7J = 0.335
HORIZON_JOURS = 7

# Au-dela de cette proportion de features manquantes, la prediction
# n'est plus consideree fiable (voir DegradationCapteurError).
SEUIL_MAX_FEATURES_MANQUANTES = 0.5


class DegradationCapteurError(Exception):
    """Levee quand trop de capteurs sont manquants pour une prediction fiable."""


@dataclass(frozen=True)
class ResultatCriticite:
    anomalie_detectee: bool
    proba_panne_7j: float
    criticite: str  # "standard" ou "elevee"
    degrade: bool  # True si une imputation a ete necessaire (capteur manquant)


def charger_modeles(models_dir: Path = MODELS_DIR) -> dict:
    return {
        "anomalie": joblib.load(models_dir / "isolation_forest.joblib"),
        "rul": joblib.load(models_dir / "random_survival_forest.joblib"),
    }


def _completer_features_manquantes(mesures: dict, colonnes: list[str]) -> tuple[list[float], bool]:
    valeurs = []
    nb_manquantes = 0
    for col in colonnes:
        valeur = mesures.get(col)
        if valeur is None:
            valeurs.append(0.0)  # valeur neutre apres mise a l'echelle
            nb_manquantes += 1
        else:
            valeurs.append(valeur)

    taux_manquant = nb_manquantes / len(colonnes)
    if taux_manquant > SEUIL_MAX_FEATURES_MANQUANTES:
        raise DegradationCapteurError(
            f"{nb_manquantes}/{len(colonnes)} mesures manquantes — prediction non fiable"
        )
    return valeurs, nb_manquantes > 0


def predire_anomalie(modeles: dict, mesures: dict) -> tuple[bool, bool]:
    paquet = modeles["anomalie"]
    valeurs, degrade = _completer_features_manquantes(mesures, paquet["colonnes_features"])
    X = paquet["scaler"].transform([valeurs])
    anomalie = paquet["modele"].predict(X)[0] == -1
    return bool(anomalie), degrade


def predire_proba_panne_7j(modeles: dict, mesures: dict) -> tuple[float, bool]:
    paquet = modeles["rul"]
    valeurs, degrade = _completer_features_manquantes(mesures, paquet["colonnes_features"])
    X = paquet["scaler"].transform([valeurs])
    fonction_survie = paquet["modele"].predict_survival_function(X)[0]
    proba = 1 - fonction_survie(HORIZON_JOURS)
    return float(proba), degrade


def evaluer_criticite(modeles: dict, mesures: dict) -> ResultatCriticite:
    anomalie, degrade_1 = predire_anomalie(modeles, mesures)
    proba_panne, degrade_2 = predire_proba_panne_7j(modeles, mesures)

    criticite = "elevee" if (anomalie or proba_panne >= SEUIL_PROBA_PANNE_7J) else "standard"

    return ResultatCriticite(
        anomalie_detectee=anomalie,
        proba_panne_7j=proba_panne,
        criticite=criticite,
        degrade=degrade_1 or degrade_2,
    )
