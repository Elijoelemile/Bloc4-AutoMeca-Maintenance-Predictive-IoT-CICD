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
"""
import os
import uuid
from datetime import datetime, timezone
from enum import Enum

from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from app.criticite import DegradationCapteurError, charger_modeles, evaluer_criticite

app = FastAPI(title="AutoMeca — Service de maintenance predictive")

_modeles_charges: dict | None = None


def get_modeles() -> dict:
    """Charge les modeles une seule fois (paresseux) — evite de recharger
    186 Mo a chaque requete."""
    global _modeles_charges
    if _modeles_charges is None:
        _modeles_charges = charger_modeles()
    return _modeles_charges


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


TICKETS: dict[str, TicketGMAO] = {}


@app.post("/predictions/ticket", response_model=TicketGMAO)
def creer_ticket(entree: MesuresEntree, _: None = Depends(verifier_cle_api)) -> TicketGMAO:
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


@app.get("/sante")
def sante() -> dict:
    """Pas d'authentification requise — utilise par le monitoring d'infrastructure."""
    return {"statut": "ok"}
