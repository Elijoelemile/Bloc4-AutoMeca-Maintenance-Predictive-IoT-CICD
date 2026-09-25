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
tickets sont stockes dans un fichier JSON (voir CHEMIN_PERSISTANCE),
avec la meme structure (statut, criticite, horodatage) qu'une vraie
integration API GMAO utiliserait. Persistance volontairement minimale
(un fichier, pas une base de donnees) : l'objectif est de survivre a un
redemarrage du conteneur, pas de servir plusieurs instances en
parallele — une vraie mise en production remplacerait ce fichier par la
base de donnees GMAO cible, sans changer la logique metier.

Derive et performance (voir app/derive.py, app/performance.py) : la
fenetre glissante des mesures recues et les latences mesurees restent
en memoire (FENETRE_MESURES/LATENCES_MS) — leur perte au redemarrage
est sans consequence (elles se reconstituent au fil des requetes
suivantes), contrairement a l'historique des tickets.
"""
import json
import os
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from app.criticite import DegradationCapteurError, charger_modeles, evaluer_criticite
from app.derive import TAILLE_FENETRE, ResultatDerive, charger_reference, detecter_derive
from app.explicabilite import ResultatExplication, construire_explainer_anomalie, construire_explainer_rul, expliquer
from app.performance import ResultatReel, StatistiquesPerformance, calculer_performance

_modeles_charges: dict | None = None
_reference_derive: dict | None = None
_explainers: dict | None = None


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


def get_explainers() -> dict:
    """Construit les explainers SHAP une seule fois (paresseux) — leur
    construction est lente (~13s pour le TreeExplainer, mesure reelle),
    voir app/explicabilite.py."""
    global _explainers
    if _explainers is None:
        modeles = get_modeles()
        _explainers = {
            "anomalie": construire_explainer_anomalie(modeles),
            "rul": construire_explainer_rul(modeles),
        }
    return _explainers


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Prechauffage au demarrage du service (pas a la premiere requete) :
    # sans ca, le premier technicien a consulter une explication
    # attendrait ~13s (construction du TreeExplainer, mesure reelle en
    # verification manuelle) — un cout invisible au demarrage du
    # conteneur est largement preferable a une latence surprise cote
    # utilisateur.
    get_modeles()
    get_reference_derive()
    get_explainers()
    _charger_tickets()
    yield


app = FastAPI(title="AutoMeca — Service de maintenance predictive", lifespan=lifespan)


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
    mesures: dict[str, float] = Field(
        default_factory=dict, description="Conservees pour permettre l'explicabilite a posteriori (voir /tickets/{id}/explication)"
    )


class ClotureTicket(BaseModel):
    resultat_reel: ResultatReel


TICKETS: dict[str, TicketGMAO] = {}

# Fenetres glissantes en memoire pour le monitoring (voir app/derive.py,
# app/performance.py) — bornees (deque a taille fixe) pour ne jamais
# grossir indefiniment.
FENETRE_MESURES: deque[dict[str, float]] = deque(maxlen=TAILLE_FENETRE)
LATENCES_MS: deque[float] = deque(maxlen=TAILLE_FENETRE)

# Persistance de l'historique des tickets (voir docstring du module) —
# chemin par defaut adapte au developpement local ; en production, le
# volume Docker monte sur /data rend ce fichier persistant entre deux
# redemarrages du conteneur (voir docker-compose.yml).
CHEMIN_PERSISTANCE = Path(os.environ.get("TICKETS_PERSISTANCE_PATH", "data/tickets.json"))


def _charger_tickets() -> None:
    """Recharge l'historique des tickets depuis le disque au demarrage
    du service — sans cela, chaque redemarrage de conteneur effacerait
    l'historique GMAO, ce qui n'aurait pas de sens pour un systeme cense
    en simuler un vrai."""
    if not CHEMIN_PERSISTANCE.exists():
        return
    donnees = json.loads(CHEMIN_PERSISTANCE.read_text(encoding="utf-8"))
    for ticket_id, ticket_dict in donnees.items():
        TICKETS[ticket_id] = TicketGMAO(**ticket_dict)


def _sauvegarder_tickets() -> None:
    """Ecrit l'integralite de TICKETS sur disque — appele apres chaque
    creation/modification. Volume de tickets attendu (dizaines a
    quelques centaines pour une demonstration) : reecrire le fichier en
    entier a chaque fois reste largement suffisant, pas besoin d'un
    format d'ajout incremental."""
    CHEMIN_PERSISTANCE.parent.mkdir(parents=True, exist_ok=True)
    donnees = {ticket_id: json.loads(ticket.model_dump_json()) for ticket_id, ticket in TICKETS.items()}
    CHEMIN_PERSISTANCE.write_text(json.dumps(donnees, ensure_ascii=False, indent=2), encoding="utf-8")


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
        mesures=entree.mesures,
    )
    TICKETS[ticket.ticket_id] = ticket
    _sauvegarder_tickets()

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
    _sauvegarder_tickets()
    return ticket


@app.get("/tickets", response_model=list[TicketGMAO])
def lister_tickets(statut: StatutTicket | None = None, _: None = Depends(verifier_cle_api)) -> list[TicketGMAO]:
    """Liste des tickets, les plus recents en premier — sans ca,
    l'interface de supervision devrait connaitre a l'avance chaque
    ticket_id, ce qui n'a pas de sens pour un technicien qui doit
    decouvrir les tickets en attente."""
    tickets = list(TICKETS.values())
    if statut is not None:
        tickets = [t for t in tickets if t.statut == statut]
    return sorted(tickets, key=lambda t: t.horodatage, reverse=True)


@app.get("/tickets/{ticket_id}", response_model=TicketGMAO)
def lire_ticket(ticket_id: str, _: None = Depends(verifier_cle_api)) -> TicketGMAO:
    ticket = TICKETS.get(ticket_id)
    if ticket is None:
        raise HTTPException(status_code=404, detail="Ticket introuvable")
    return ticket


@app.get("/tickets/{ticket_id}/explication", response_model=ResultatExplication)
def expliquer_ticket(ticket_id: str, _: None = Depends(verifier_cle_api)) -> ResultatExplication:
    """Facteurs declencheurs (SHAP) — exigence "IA ethique" du sujet :
    permet a un technicien de comprendre pourquoi l'alerte a ete levee,
    pas seulement de la constater. Recalcule a la demande (non stocke a
    la creation du ticket) — coute ~40ms (anomalie) a ~400ms (RUL), voir
    app/explicabilite.py."""
    ticket = TICKETS.get(ticket_id)
    if ticket is None:
        raise HTTPException(status_code=404, detail="Ticket introuvable")
    modeles = get_modeles()
    explainers = get_explainers()
    return expliquer(modeles, explainers, ticket.mesures)


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
    _sauvegarder_tickets()
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
