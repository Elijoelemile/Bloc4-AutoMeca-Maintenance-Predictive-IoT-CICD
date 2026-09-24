"""Service de prediction et creation de tickets GMAO — Bloc 4
(Integration et CI/CD).

Cree automatiquement un ticket de maintenance a partir d'une alerte,
avec le niveau de criticite calcule par les modeles (app/criticite.py).

Garde-fou "human oversight" exige par le sujet : un ticket de
criticite elevee n'est **jamais** execute automatiquement — il reste
"en attente de validation" jusqu'a l'approbation explicite d'un
technicien senior (endpoint /tickets/{id}/valider). L'automatisation
technique (ce service, le pipeline CI/CD) ne concerne que la creation
et le routage du ticket, jamais la decision operationnelle qui en
decoule.

GMAO simulee : ce projet n'a pas de vrai systeme GMAO a integrer — les
tickets sont stockes en memoire, avec la meme structure (statut,
criticite, horodatage) qu'une vraie integration API GMAO utiliserait.

Derive et performance (voir app/derive.py, app/performance.py) : la
fenetre glissante des mesures recues et les latences mesurees sont
elles aussi en memoire, coherent avec la simulation GMAO ci-dessus —
une vraie mise en production remplacerait TICKETS/FENETRE_MESURES/
LATENCES_MS par un stockage persistant, sans changer la logique.
"""
import os
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from enum import Enum

from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from app.criticite import DegradationCapteurError, charger_modeles, evaluer_criticite
from app.derive import TAILLE_FENETRE, ResultatDerive, charger_reference, detecter_derive
from app.performance import ResultatReel, StatistiquesPerformance, calculer_performance

app = FastAPI(title="AutoMeca — Service de maintenance predictive")

_modeles_charges: dict | None = None
_reference_derive: dict | None = None


def get_modeles() -> dict:
    """Charge les modeles une seule fois (paresseux) — evite de recharger
    186 Mo a chaque requete."""
    global _modeles_charges
    if _modeles_charges is None:
        _modeles_charges = charger_modeles()
    return _modeles_charges


def get_reference_derive() -> dict:
    global _reference_derive
    if _reference_derive is None:
        _reference_derive = charger_reference()
    return _reference_derive


API_KEY_HEADER = APIKeyHeader(name="X-API-Key")


def verifier_cle_api(cle: str = Security(API_KEY_HEADER)) -> None:
    cle_attendue = os.environ.get("API_KEY")
    if not cle_attendue:
        raise HTTPException(status_code=500, detail="API_KEY non configuree cote serveur")
    if cle != cle_attendue:
        raise HTTPException(status_code=401, detail="Cle API invalide")


class StatutTicket(str, Enum):
    ASSIGNE = "assigne_equipe_maintenance"
    EN_ATTENTE_VALIDATION = "en_attente_validation_technicien_senior"
    CLOTURE = "cloture"


class MesuresEntree(BaseModel):
    machine_id: int
    mesures: dict[str, float] = Field(
        ..., description="Features attendues par les modeles (voir models/*.joblib) — une mesure manquante degrade la prediction sans la faire echouer"
    )


class TicketGMAO(BaseModel):
    ticket_id: str
    machine_id: int
    horodatage: str
    anomalie_detectee: bool
    proba_panne_7j: float
    criticite: str
    statut: StatutTicket
    degrade: bool
    resultat_reel: ResultatReel | None = None


class ClotureTicket(BaseModel):
    resultat_reel: ResultatReel


TICKETS: dict[str, TicketGMAO] = {}

# Fenetres glissantes en memoire pour le monitoring (voir app/derive.py,
# app/performance.py) — bornees (deque a taille fixe) pour ne jamais
# grossir indefiniment.
FENETRE_MESURES: deque[dict[str, float]] = deque(maxlen=TAILLE_FENETRE)
LATENCES_MS: deque[float] = deque(maxlen=TAILLE_FENETRE)


@app.post("/predictions/ticket", response_model=TicketGMAO)
def creer_ticket(entree: MesuresEntree, _: None = Depends(verifier_cle_api)) -> TicketGMAO:
    debut = time.perf_counter()
    modeles = get_modeles()
    try:
        resultat = evaluer_criticite(modeles, entree.mesures)
    except DegradationCapteurError as exc:
        # degradation gracieuse : echec propre et explicite, jamais un crash serveur
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    statut = StatutTicket.EN_ATTENTE_VALIDATION if resultat.criticite == "elevee" else StatutTicket.ASSIGNE
    ticket = TicketGMAO(
        ticket_id=str(uuid.uuid4()),
        machine_id=entree.machine_id,
        horodatage=datetime.now(timezone.utc).isoformat(),
        anomalie_detectee=resultat.anomalie_detectee,
        proba_panne_7j=resultat.proba_panne_7j,
        criticite=resultat.criticite,
        statut=statut,
        degrade=resultat.degrade,
    )
    TICKETS[ticket.ticket_id] = ticket

    FENETRE_MESURES.append(entree.mesures)
    LATENCES_MS.append((time.perf_counter() - debut) * 1000)

    return ticket


@app.post("/tickets/{ticket_id}/valider", response_model=TicketGMAO)
def valider_ticket(ticket_id: str, _: None = Depends(verifier_cle_api)) -> TicketGMAO:
    """Validation humaine obligatoire (technicien senior) — le garde-fou
    "human oversight" : aucune action corrective n'est engagee avant cet
    appel explicite, pour un ticket de criticite elevee."""
    ticket = TICKETS.get(ticket_id)
    if ticket is None:
        raise HTTPException(status_code=404, detail="Ticket introuvable")
    if ticket.statut != StatutTicket.EN_ATTENTE_VALIDATION:
        raise HTTPException(status_code=409, detail="Ce ticket ne necessite pas de validation")
    ticket.statut = StatutTicket.ASSIGNE
    TICKETS[ticket_id] = ticket
    return ticket


@app.get("/tickets/{ticket_id}", response_model=TicketGMAO)
def lire_ticket(ticket_id: str, _: None = Depends(verifier_cle_api)) -> TicketGMAO:
    ticket = TICKETS.get(ticket_id)
    if ticket is None:
        raise HTTPException(status_code=404, detail="Ticket introuvable")
    return ticket


@app.post("/tickets/{ticket_id}/cloturer", response_model=TicketGMAO)
def cloturer_ticket(ticket_id: str, cloture: ClotureTicket, _: None = Depends(verifier_cle_api)) -> TicketGMAO:
    """Cloture d'un ticket par un technicien avec le resultat reel
    (panne confirmee ou fausse alerte) — c'est ce retour terrain qui
    alimente /monitoring/performance (voir app/performance.py)."""
    ticket = TICKETS.get(ticket_id)
    if ticket is None:
        raise HTTPException(status_code=404, detail="Ticket introuvable")
    if ticket.statut != StatutTicket.ASSIGNE:
        raise HTTPException(status_code=409, detail="Seul un ticket assigne peut etre cloture")
    ticket.statut = StatutTicket.CLOTURE
    ticket.resultat_reel = cloture.resultat_reel
    TICKETS[ticket_id] = ticket
    return ticket


@app.get("/monitoring/derive", response_model=dict[str, ResultatDerive])
def monitoring_derive(_: None = Depends(verifier_cle_api)) -> dict[str, ResultatDerive]:
    """Derive des variables d'entree, par modele (suivi statistique de
    la distribution — voir app/derive.py). Calculee sur la fenetre des
    dernieres requetes recues par ce service."""
    reference = get_reference_derive()
    modeles = get_modeles()
    return {
        "isolation_forest": detecter_derive(
            FENETRE_MESURES, reference, "isolation_forest", modeles["anomalie"]["colonnes_features"]
        ),
        "random_survival_forest": detecter_derive(
            FENETRE_MESURES, reference, "random_survival_forest", modeles["rul"]["colonnes_features"]
        ),
    }


@app.get("/monitoring/performance", response_model=StatistiquesPerformance)
def monitoring_performance(_: None = Depends(verifier_cle_api)) -> StatistiquesPerformance:
    """Precision et taux de fausses alertes (a partir des tickets clotures
    par un technicien) + latence moyenne du service — voir app/performance.py."""
    resultats_reels = [t.resultat_reel for t in TICKETS.values() if t.resultat_reel is not None]
    return calculer_performance(resultats_reels, list(LATENCES_MS))


@app.get("/sante")
def sante() -> dict:
    """Pas d'authentification requise — utilise par le monitoring d'infrastructure."""
    return {"statut": "ok"}
