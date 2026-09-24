"""Detection de derive (data drift) — exigence du sujet (Bloc 4,
"Derive et monitoring") : "suivi statistique de la distribution des
variables d'entree", declenchant une alerte de reentrainement en cas
de derive averee.

Reference : distribution reelle des features au moment de
l'entrainement (models/reference_distribution.json, genere depuis les
memes donnees et le meme code que les notebooks du depot solution-IA —
voir scratchpad/build_reference_distribution.py cite dans le README).
Pas de recalcul depuis les .joblib : le scaler ne stocke que
moyenne/ecart-type, insuffisant pour une vraie comparaison de
distribution (PSI a besoin de la forme complete, pas seulement des deux
premiers moments).

Indicateur : PSI (Population Stability Index), standard en MLOps pour
la derive de features — compare la repartition d'une fenetre recente a
la reference. Variables continues (capteurs) : deciles de reference.
Variables a faible cardinalite, <= 10 valeurs distinctes a
l'entrainement (indicatrices de modele, nb_erreurs_7j) : proportions
exactes par valeur — les deciles seuls ne suffisent pas a representer
une proportion reelle (ex: 14,76% arrondis a 10%/20% par des deciles),
ce qui gonflait artificiellement le PSI en verification manuelle avant
cette distinction (voir reference_distribution.json, champ "type").
Seuils usuels du domaine : PSI < 0.1 stable, 0.1-0.25 derive moderee,
> 0.25 derive significative.

Limite connue, acceptee : a n=200 (fenetre pleine), le PSI garde un peu
de bruit residuel meme sans derive reelle sur les variables continues a
deciles peu contrastes (ex: "age" a franchi occasionnellement le seuil
en verification manuelle repetee, ~1 fois sur 4). Une fausse alerte
occasionnelle sur une seule feature reste conforme a l'esprit du
sujet (signal a destination de l'equipe data pour revue, pas une action
automatique) — un lissage plus poussé (fenetre plus grande, exigence de
plusieurs features simultanement en derive) est delibérément ecarte ici
pour ne pas complexifier au-dela du besoin reel de ce projet.

Reentrainement : la derive confirmee ne declenche pas un job
automatique ici (le jeu d'entrainement fait plusieurs Go et n'a pas sa
place en CI gratuite) mais un signal explicite (`GET /monitoring/derive`)
a destination de l'equipe data, qui relance alors les notebooks
versionnes du depot solution-IA — deterministes (random_state=42 fixe
partout) et donc reproductibles, ce qui satisfait l'exigence "donnees
et parametres versionnes" sans sur-ingenierie d'un orchestrateur pour
un seul reentrainement occasionnel.
"""
import json
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path

REFERENCE_PATH = Path(__file__).resolve().parent.parent / "models" / "reference_distribution.json"

SEUIL_PSI_DERIVE_SIGNIFICATIVE = 0.25
TAILLE_FENETRE = 200

# Calibre empiriquement (simulation multinomiale sous H0, voir
# scratchpad de verification cite dans le README) : avec ~10 bins de
# reference, le PSI sous pur bruit d'echantillonnage a n=30-40 depasse
# regulierement 0.25 (p99 a n=40 : ~1.05, largement au-dessus du seuil
# de derive significative) — inutilisable, fausses alertes garanties.
# A n=200 (fenetre pleine), p99 sous H0 tombe a ~0.11, confortablement
# sous le seuil : on attend donc la fenetre pleine avant tout jugement.
N_MIN_OBSERVATIONS = TAILLE_FENETRE


def charger_reference(path: Path = REFERENCE_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@dataclass(frozen=True)
class DeriveFeature:
    feature: str
    psi: float


@dataclass(frozen=True)
class ResultatDerive:
    modele: str
    n_observations: int
    features_en_derive: list[DeriveFeature]
    derive_confirmee: bool
    reentrainement_recommande: bool


def _bornes_ponderees(quantiles_reference: list[float]) -> tuple[list[float], list[float]]:
    """Construit les bornes de bins a partir des 9 deciles de reference,
    en fusionnant les deciles dupliques (frequents sur les variables peu
    variees : age, indicatrices de modele, nb_erreurs_7j). Sans cette
    fusion, un decile duplique (ex: 8 deciles a 0 sur une variable
    binaire) cree des bins de largeur nulle, comptes comme "vides" et
    lissees a 1e-4 — ce qui gonfle artificiellement le PSI meme sur des
    donnees non derivees (repere en verification manuelle : PSI > 6 sur
    des indicatrices de modele alors que l'echantillon suivait la
    reference). Chaque decile duplique n fois porte le poids de reference
    de ces n bins fusionnes (n * 10%), au lieu de 10% par bin nominal."""
    uniques: list[float] = []
    poids: list[float] = []
    for q in quantiles_reference:
        if uniques and q == uniques[-1]:
            poids[-1] += 0.1
        else:
            uniques.append(q)
            poids.append(0.1)
    bornes = [float("-inf")] + uniques + [float("inf")]
    poids.append(1.0 - sum(poids))  # dernier bin, au-dela du 9e decile
    return bornes, poids


def _psi_continue(valeurs_recentes: list[float], quantiles_reference: list[float]) -> float:
    """PSI sur des bins definis par les deciles de reference (deciles
    dupliques fusionnes, voir _bornes_ponderees) — variables continues."""
    bornes, poids_reference = _bornes_ponderees(quantiles_reference)
    n = len(valeurs_recentes)

    psi = 0.0
    for i, part_reference in enumerate(poids_reference):
        bas, haut = bornes[i], bornes[i + 1]
        compte = sum(1 for v in valeurs_recentes if bas < v <= haut)
        part_actuelle = compte / n
        # lissage pour eviter log(0) sur un bin vide de la fenetre recente
        part_actuelle = max(part_actuelle, 1e-4)
        part_reference = max(part_reference, 1e-4)
        psi += (part_actuelle - part_reference) * math.log(part_actuelle / part_reference)
    return psi


def _psi_categorique(valeurs_recentes: list[float], proportions_reference: dict[str, float]) -> float:
    """PSI a partir des proportions exactes par valeur (variables a
    faible cardinalite : indicatrices de modele, nb_erreurs_7j — les 9
    deciles ne suffiraient pas a representer leurs proportions reelles,
    voir le module docstring)."""
    n = len(valeurs_recentes)
    proportions = {float(valeur): part for valeur, part in proportions_reference.items()}

    psi = 0.0
    compte_connu = 0
    for valeur_ref, part_reference in proportions.items():
        compte = sum(1 for v in valeurs_recentes if v == valeur_ref)
        compte_connu += compte
        part_actuelle = max(compte / n, 1e-4)
        part_reference = max(part_reference, 1e-4)
        psi += (part_actuelle - part_reference) * math.log(part_actuelle / part_reference)

    # valeurs jamais observees a l'entrainement : bin residuel, reference ~0
    compte_inconnu = n - compte_connu
    if compte_inconnu > 0:
        part_actuelle = compte_inconnu / n
        psi += (part_actuelle - 1e-4) * math.log(part_actuelle / 1e-4)
    return psi


def _psi(valeurs_recentes: list[float], feature_reference: dict) -> float:
    if feature_reference["type"] == "categorique":
        return _psi_categorique(valeurs_recentes, feature_reference["proportions"])
    return _psi_continue(valeurs_recentes, feature_reference["quantiles_10_90"])


def detecter_derive(fenetre: deque, reference: dict, nom_modele: str, colonnes: list[str]) -> ResultatDerive:
    n = len(fenetre)
    ref_modele = reference[nom_modele]["features"]

    features_en_derive: list[DeriveFeature] = []
    if n >= N_MIN_OBSERVATIONS:
        for col in colonnes:
            valeurs = [obs[col] for obs in fenetre if col in obs]
            if len(valeurs) < N_MIN_OBSERVATIONS:
                continue
            psi = _psi(valeurs, ref_modele[col])
            if psi >= SEUIL_PSI_DERIVE_SIGNIFICATIVE:
                features_en_derive.append(DeriveFeature(feature=col, psi=round(psi, 4)))

    derive_confirmee = len(features_en_derive) > 0
    return ResultatDerive(
        modele=nom_modele,
        n_observations=n,
        features_en_derive=features_en_derive,
        derive_confirmee=derive_confirmee,
        reentrainement_recommande=derive_confirmee,
    )
