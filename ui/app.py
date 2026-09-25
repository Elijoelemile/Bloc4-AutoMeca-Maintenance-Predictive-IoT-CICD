"""Interface de supervision — technicien (Bloc 4, exigence "Conformite"
du sujet : "l'interface de supervision est concue pour rester
accessible aux techniciens en situation de handicap").

Consomme l'API du meme depot (app/main.py) par HTTP — aucune
dependance technique directe au code de l'API, seulement au contrat
HTTP qu'elle expose (meme principe de separation que le reste du
projet : deux services independants, un point de passage explicite).

Choix d'accessibilite (voir aussi CONFORMITE.md) :
- uniquement des widgets Streamlit natifs, aucun HTML/JS injecte
  (`unsafe_allow_html`) qui casserait leur gestion clavier/ARIA native.
- chaque widget a un label explicite (jamais de label vide/masque).
- la criticite et le statut sont toujours donnes en texte (+ icone),
  jamais par la couleur seule.
- l'information critique (criticite, statut, decision) est affichee en
  texte simple (st.metric/st.write), le tableau (st.dataframe) ne sert
  qu'a la liste recapitulative, en complement — les tableaux
  interactifs de Streamlit ont des limites connues cote lecteur
  d'ecran, on ne s'appuie donc pas sur eux pour l'information critique.
"""
import os

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="AutoMeca — Supervision", layout="wide")

CRITICITE_LABELS = {"elevee": "⚠️ CRITICITÉ ÉLEVÉE", "standard": "✅ Criticité standard"}
STATUT_LABELS = {
    "en_attente_validation_technicien_senior": "🟠 En attente de validation",
    "assigne_equipe_maintenance": "🔵 Assigné à l'équipe maintenance",
    "cloture": "⚪ Clôturé",
}

# Noms de colonnes techniques (voir Bloc 2, data_dictionary/dictionnaire_donnees.md
# pour les libelles source "Tension mesuree" / "Vitesse de rotation" / "Pression" /
# "Vibration" — aucune unite precise n'y est documentee, le dataset source
# n'en fournit pas non plus, donc aucune n'est inventee ici) -> libelle lisible
# pour un technicien qui ne connait pas les noms de colonnes en base.
VARIABLE_LABELS = {
    "volt_moy24h": "Tension moyenne (24 h)",
    "rotate_moy24h": "Vitesse de rotation moyenne (24 h)",
    "pressure_moy24h": "Pression moyenne (24 h)",
    "vibration_moy24h": "Vibration moyenne (24 h)",
    "volt_std24h": "Tension — variabilité (24 h)",
    "rotate_std24h": "Vitesse de rotation — variabilité (24 h)",
    "pressure_std24h": "Pression — variabilité (24 h)",
    "vibration_std24h": "Vibration — variabilité (24 h)",
    "volt_moy7j": "Tension moyenne (7 jours)",
    "rotate_moy7j": "Vitesse de rotation moyenne (7 jours)",
    "pressure_moy7j": "Pression moyenne (7 jours)",
    "vibration_moy7j": "Vibration moyenne (7 jours)",
    "volt_std7j": "Tension — variabilité (7 jours)",
    "rotate_std7j": "Vitesse de rotation — variabilité (7 jours)",
    "pressure_std7j": "Pression — variabilité (7 jours)",
    "vibration_std7j": "Vibration — variabilité (7 jours)",
    "nb_erreurs_7j": "Nombre d'erreurs machine (7 jours)",
    "age": "Âge de la machine",
    "model_model1": "Modèle de machine — type 1",
    "model_model2": "Modèle de machine — type 2",
    "model_model3": "Modèle de machine — type 3",
    "model_model4": "Modèle de machine — type 4",
}


def _libelle_variable(nom: str) -> str:
    return VARIABLE_LABELS.get(nom, nom)


def _config_api() -> tuple[str, str]:
    """Lit la config depuis les variables d'environnement (injectees par
    docker-compose.yml, voir README) — jamais un champ visible par le
    technicien : ni l'URL ni la cle API n'ont a etre connues ou saisies
    par la personne qui utilise cette interface au quotidien. Exposer
    la cle dans un champ d'UI (meme masque) la rendrait lisible en un
    clic par n'importe quel utilisateur authentifie, ce qui contredit
    le principe "pas de secret en dur" (voir CONFORMITE.md)."""
    url = os.environ.get("API_URL", "http://localhost:8000")
    cle = os.environ.get("API_KEY", "")
    return url.rstrip("/"), cle


def _appel_api(methode: str, url_base: str, cle_api: str, chemin: str, **kwargs):
    try:
        reponse = requests.request(
            methode, f"{url_base}{chemin}", headers={"X-API-Key": cle_api}, timeout=30, **kwargs,
        )
    except requests.exceptions.ConnectionError:
        st.error(f"Impossible de joindre l'API à l'adresse « {url_base} ». Contactez l'équipe technique.")
        return None
    except requests.exceptions.Timeout:
        st.error("L'API n'a pas répondu à temps (délai de 30 secondes dépassé).")
        return None

    if reponse.status_code == 401:
        st.error("Clé API refusée par le service. Contactez l'équipe technique.")
        return None
    if not reponse.ok:
        detail = reponse.json().get("detail", reponse.text) if reponse.content else reponse.text
        st.error(f"Erreur {reponse.status_code} : {detail}")
        return None
    return reponse.json() if reponse.content else {}


def _afficher_facteurs(titre: str, facteurs: list[dict]) -> None:
    st.markdown(f"**{titre}**")
    if not facteurs:
        st.write("Aucun facteur disponible.")
        return
    df = pd.DataFrame(facteurs).rename(
        columns={"feature": "Variable", "valeur": "Valeur mesurée", "contribution_shap": "Contribution SHAP"}
    )
    df["Variable"] = df["Variable"].map(_libelle_variable)
    # Table texte en premier (source d'information de reference,
    # accessible) puis graphique en complement visuel.
    st.dataframe(df, hide_index=True, use_container_width=True)
    st.bar_chart(df.set_index("Variable")["Contribution SHAP"])


def onglet_tickets(url_base: str, cle_api: str) -> None:
    st.subheader("Tickets de maintenance")

    with st.container(border=True):
        col_bouton, col_filtre = st.columns([1, 2])
        with col_bouton:
            # Cliquer ce bouton n'a pas d'effet propre : sa seule fonction
            # est de provoquer une nouvelle execution du script
            # (comportement natif de tout bouton Streamlit), qui refait
            # l'appel /tickets ci-dessous.
            st.button("Actualiser la liste", icon="🔄")
        with col_filtre:
            filtre = st.selectbox(
                "Filtrer par statut",
                options=["Tous", "en_attente_validation_technicien_senior", "assigne_equipe_maintenance", "cloture"],
                format_func=lambda s: "Tous les tickets" if s == "Tous" else STATUT_LABELS[s],
            )
    params = {} if filtre == "Tous" else {"statut": filtre}
    tickets = _appel_api("GET", url_base, cle_api, "/tickets", params=params)
    if tickets is None:
        return
    if not tickets:
        st.info("Aucun ticket pour ce filtre.")
        return

    df = pd.DataFrame(tickets)[["ticket_id", "machine_id", "criticite", "statut", "horodatage"]]
    df["criticite"] = df["criticite"].map(lambda c: CRITICITE_LABELS.get(c, c))
    df["statut"] = df["statut"].map(lambda s: STATUT_LABELS.get(s, s))
    st.dataframe(df, hide_index=True, use_container_width=True)

    st.divider()
    options_id = [t["ticket_id"] for t in tickets]
    ticket_choisi = st.selectbox(
        "Sélectionner un ticket à examiner",
        options=options_id,
        format_func=lambda tid: f"{tid[:8]}… — machine {next(t['machine_id'] for t in tickets if t['ticket_id'] == tid)}",
    )
    if ticket_choisi:
        _afficher_detail_ticket(url_base, cle_api, ticket_choisi)


def _afficher_detail_ticket(url_base: str, cle_api: str, ticket_id: str) -> None:
    ticket = _appel_api("GET", url_base, cle_api, f"/tickets/{ticket_id}")
    if ticket is None:
        return

    dernier_message = st.session_state.pop("dernier_message_ticket", None)
    if dernier_message is not None:
        st.success(dernier_message)

    st.markdown(f"### Ticket {ticket_id[:8]}… — Machine {ticket['machine_id']}")

    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        col1.metric("Criticité", CRITICITE_LABELS.get(ticket["criticite"], ticket["criticite"]))
        col2.metric("Probabilité de panne sous 7 jours", f"{ticket['proba_panne_7j']:.1%}")
        col3.metric("Anomalie détectée", "Oui" if ticket["anomalie_detectee"] else "Non")
        st.write(f"**Statut :** {STATUT_LABELS.get(ticket['statut'], ticket['statut'])}")
        if ticket["degrade"]:
            st.warning("Prédiction en mode dégradé : une ou plusieurs mesures capteur manquaient et ont été imputées.", icon="⚠️")
        if ticket["resultat_reel"]:
            st.write(f"**Résultat réel constaté :** {ticket['resultat_reel'].replace('_', ' ')}")

    with st.expander("🔍 Facteurs déclencheurs (explicabilité SHAP)"):
        st.caption(
            "Calcul à la demande (non stocké à la création du ticket) — peut prendre jusqu'à une seconde."
        )
        if st.button("Calculer l'explication", icon="🔍", key=f"expliquer_{ticket_id}"):
            explication = _appel_api("GET", url_base, cle_api, f"/tickets/{ticket_id}/explication")
            if explication is not None:
                _afficher_facteurs("Modèle de détection d'anomalies", explication["facteurs_anomalie"])
                _afficher_facteurs("Modèle de durée de vie résiduelle (RUL)", explication["facteurs_rul"])

    with st.expander("📊 Mesures brutes envoyées par les capteurs"):
        mesures_lisibles = [(_libelle_variable(k), v) for k, v in ticket["mesures"].items()]
        st.dataframe(
            pd.DataFrame(mesures_lisibles, columns=["Variable", "Valeur"]),
            hide_index=True, use_container_width=True,
        )

    st.divider()
    if ticket["statut"] == "en_attente_validation_technicien_senior":
        with st.container(border=True):
            st.warning(
                "Ce ticket de criticité élevée nécessite la validation explicite d'un technicien senior "
                "avant toute action corrective (garde-fou de supervision humaine).",
                icon="🟠",
            )
            if st.button("Valider ce ticket (technicien senior)", icon="✅", type="primary", key=f"valider_{ticket_id}"):
                resultat = _appel_api("POST", url_base, cle_api, f"/tickets/{ticket_id}/valider")
                if resultat is not None:
                    st.session_state["dernier_message_ticket"] = "Ticket validé et assigné à l'équipe maintenance."
                    st.rerun()
    elif ticket["statut"] == "assigne_equipe_maintenance":
        with st.container(border=True):
            st.markdown("**Clôturer ce ticket avec le résultat constaté sur le terrain :**")
            resultat_reel = st.radio(
                "Résultat réel", options=["panne_confirmee", "fausse_alerte"],
                format_func=lambda r: "Panne confirmée" if r == "panne_confirmee" else "Fausse alerte",
                key=f"resultat_{ticket_id}",
            )
            if st.button("Clôturer le ticket", icon="🔒", type="primary", key=f"cloturer_{ticket_id}"):
                resultat = _appel_api(
                    "POST", url_base, cle_api, f"/tickets/{ticket_id}/cloturer", json={"resultat_reel": resultat_reel},
                )
                if resultat is not None:
                    st.session_state["dernier_message_ticket"] = (
                        "Ticket clôturé. Merci — ce retour alimente le suivi de performance (onglet Monitoring)."
                    )
                    st.rerun()


def onglet_nouvelle_alerte(url_base: str, cle_api: str) -> None:
    st.subheader("Simuler la réception d'une alerte")

    dernier_ticket = st.session_state.pop("dernier_ticket_cree", None)
    if dernier_ticket is not None:
        st.success(f"Ticket créé : {dernier_ticket['ticket_id']} — criticité {dernier_ticket['criticite']}")

    with st.container(border=True):
        machine_id = st.number_input("Identifiant machine", min_value=1, max_value=100, value=1, step=1)
        mesures_texte = st.text_area(
            "Mesures (JSON)", height=200,
            help="Dictionnaire des features attendues par les modèles (voir models/*.joblib dans ce dépôt).",
            placeholder='{"volt_moy24h": 170.6, "rotate_moy24h": 446.6, ...}',
        )
        bouton_envoyer = st.button("Envoyer l'alerte", icon="📩", type="primary")
    if bouton_envoyer:
        import json

        try:
            mesures = json.loads(mesures_texte) if mesures_texte.strip() else {}
        except json.JSONDecodeError as exc:
            st.error(f"JSON invalide : {exc}")
            return
        ticket = _appel_api(
            "POST", url_base, cle_api, "/predictions/ticket",
            json={"machine_id": int(machine_id), "mesures": mesures},
        )
        if ticket is not None:
            # message affiche au prochain run (juste apres), pas ici : un
            # st.rerun() immediat l'effacerait avant que l'utilisateur ne
            # le voie. Le rerun est necessaire pour que l'onglet Tickets
            # (qui s'execute avant celui-ci dans le script, voir main())
            # affiche ce nouveau ticket sans action supplementaire.
            st.session_state["dernier_ticket_cree"] = ticket
            st.rerun()


def onglet_monitoring(url_base: str, cle_api: str) -> None:
    st.subheader("Suivi de dérive et de performance")

    with st.container(border=True):
        # Meme principe que le bouton "Actualiser la liste" de l'onglet
        # Tickets : force une nouvelle execution du script, qui refait
        # les appels /monitoring/* ci-dessous.
        st.button("Actualiser", icon="🔄", key="actualiser_monitoring")

    derive = _appel_api("GET", url_base, cle_api, "/monitoring/derive")
    if derive is not None:
        with st.container(border=True):
            st.markdown("#### 📉 Dérive des variables d'entrée (PSI)")
            icones_modele = {"isolation_forest": "🔍", "random_survival_forest": "⏳"}
            for nom_modele, libelle in [("isolation_forest", "Détection d'anomalies"), ("random_survival_forest", "Durée de vie résiduelle")]:
                resultat = derive[nom_modele]
                st.write(f"{icones_modele[nom_modele]} **{libelle}** — {resultat['n_observations']} observations dans la fenêtre glissante")
                if resultat["derive_confirmee"]:
                    st.error("Dérive significative détectée — réentraînement recommandé.", icon="🚨")
                    df_derive = pd.DataFrame(resultat["features_en_derive"]).rename(columns={"feature": "Variable", "psi": "PSI"})
                    df_derive["Variable"] = df_derive["Variable"].map(_libelle_variable)
                    st.dataframe(df_derive, hide_index=True, use_container_width=True)
                else:
                    st.success("Aucune dérive significative détectée.", icon="✅")

    performance = _appel_api("GET", url_base, cle_api, "/monitoring/performance")
    if performance is not None:
        with st.container(border=True):
            st.markdown("#### 📈 Performance en production (tickets clôturés)")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Tickets clôturés", performance["n_tickets_clotures"])
            col2.metric("Précision", f"{performance['precision']:.0%}" if performance["precision"] is not None else "—")
            col3.metric(
                "Taux de fausses alertes",
                f"{performance['taux_fausses_alertes']:.0%}" if performance["taux_fausses_alertes"] is not None else "—",
            )
            col4.metric(
                "Latence moyenne du service",
                f"{performance['latence_moyenne_ms']:.0f} ms" if performance["latence_moyenne_ms"] is not None else "—",
            )

            if performance["n_tickets_clotures"] > 0:
                st.caption("Répartition des tickets clôturés, derrière les taux ci-dessus :")
                df_repartition = pd.DataFrame(
                    {"Tickets": [performance["n_pannes_confirmees"], performance["n_fausses_alertes"]]},
                    index=["Pannes confirmées", "Fausses alertes"],
                )
                st.bar_chart(df_repartition, color="#2E7D32")

        tickets_clotures = _appel_api("GET", url_base, cle_api, "/tickets", params={"statut": "cloture"})
        if tickets_clotures:
            with st.container(border=True):
                st.markdown("#### 🗂️ Journal des tickets clôturés")
                df_clotures = pd.DataFrame(tickets_clotures)[
                    ["machine_id", "criticite", "proba_panne_7j", "resultat_reel", "horodatage"]
                ].rename(columns={
                    "machine_id": "Machine",
                    "criticite": "Criticité",
                    "proba_panne_7j": "Probabilité de panne (7j)",
                    "resultat_reel": "Résultat réel",
                    "horodatage": "Horodatage",
                })
                df_clotures["Criticité"] = df_clotures["Criticité"].map(lambda c: CRITICITE_LABELS.get(c, c))
                df_clotures["Résultat réel"] = df_clotures["Résultat réel"].map(
                    lambda r: "Panne confirmée" if r == "panne_confirmee" else "Fausse alerte"
                )
                df_clotures["Probabilité de panne (7j)"] = df_clotures["Probabilité de panne (7j)"].map(lambda p: f"{p:.1%}")
                st.dataframe(df_clotures, hide_index=True, use_container_width=True)


def main() -> None:
    st.title("AutoMeca Systems — Supervision maintenance prédictive")
    url_base, cle_api = _config_api()

    if not cle_api:
        st.error(
            "Configuration serveur incomplète (API_KEY manquante). "
            "Contactez l'équipe technique — voir docker-compose.yml."
        )
        return

    onglet1, onglet2, onglet3 = st.tabs(["📋 Tickets", "➕ Nouvelle alerte", "📈 Monitoring"])
    with onglet1:
        onglet_tickets(url_base, cle_api)
    with onglet2:
        onglet_nouvelle_alerte(url_base, cle_api)
    with onglet3:
        onglet_monitoring(url_base, cle_api)


if __name__ == "__main__":
    main()
