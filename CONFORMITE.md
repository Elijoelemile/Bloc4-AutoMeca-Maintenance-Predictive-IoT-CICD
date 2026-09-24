# Conformité du déploiement

Ce document répond au point "Conformité" du sujet (Bloc 4 — Déploiement de la solution IA) :

> *Le déploiement respecte le RGPD et la loi Informatique et Libertés sur le volet opérateurs, ainsi
> qu'ISO 27001. L'IA éthique est assurée par l'explicabilité des alertes (méthodes type SHAP permettant
> aux techniciens de comprendre les facteurs déclencheurs), la non-discrimination entre équipes et
> postes, et le respect de la vie privée. L'interface de supervision est conçue pour rester accessible
> aux techniciens en situation de handicap.*

Ce n'est pas un livrable séparé (voir la table officielle des livrables — Bloc 4 = deux dépôts GitHub +
slides + vidéo, pas de rapport à part) : c'est la base factuelle des slides, qui pointe vers du code et
des tests réels plutôt que vers des affirmations non vérifiables.

## 1. RGPD et loi Informatique et Libertés

**Périmètre réel de ce service** : `app/criticite.py`, `app/derive.py`, `app/performance.py` et
`app/main.py` ne traitent que `machine_id` et des mesures capteur agrégées (`volt_moy24h`,
`vibration_std7j`, `nb_erreurs_7j`, etc. — voir `models/*.joblib`, champ `colonnes_features`). Aucune
donnée liée à un opérateur, un badge ou un planning n'entre jamais dans ce système. Le croisement
mentionné dans le sujet (données machine × données d'affectation du personnel) est un **autre** cas
d'usage analytique, hors périmètre de ce dépôt — le dire explicitement évite de prétendre appliquer une
analyse d'impact (AIPD) qui ne concerne pas ce composant.

Le registre de traitement RGPD et l'AIPD eux-mêmes sont des livrables du **Bloc 1** (gouvernance des
données) — ce dépôt ne les duplique pas, il s'y conforme : aucune donnée personnelle n'est introduite ici
qui nécessiterait une entrée supplémentaire dans ce registre.

**Minimisation et conservation** : la GMAO simulée (`TICKETS`, dict en mémoire) ne conserve que
l'identifiant machine, l'horodatage et le résultat de calcul — jamais l'identité de qui a saisi une
mesure ou validé un ticket. Rien n'est persisté au-delà du cycle de vie du processus (voir le
`[!NOTE]` du README sur le caractère simulé de cette GMAO).

## 2. ISO 27001

Contrôles effectivement en place dans ce dépôt, alignés sur le SMSI déjà décrit au Bloc 1 :

- **Authentification** : toutes les routes sauf `/sante` exigent une clé API (`X-API-Key`), vérifiée
  côté serveur (`app/main.py::verifier_cle_api`).
- **Pas de secret en dur** : `.env.example` documente la variable attendue (`API_KEY`), jamais commitée.
- **Conteneurisation durcie** : utilisateur non-root (`useradd --create-home appuser`) dans les deux
  `Dockerfile` (API et interface), `HEALTHCHECK` explicite pour la supervision d'infrastructure.
- **Traçabilité** : chaque ticket porte un horodatage UTC ISO 8601 et un identifiant unique (`uuid4`).
- **Tests de non-régression** (`tests/test_regression_modele.py`) : empêche un modèle moins performant
  d'être déployé silencieusement — cohérent avec l'exigence de fiabilité/traçabilité IATF 16949 évoquée
  au Bloc 1, appliquée ici techniquement.

## 3. IA éthique

### Explicabilité (SHAP)

`app/explicabilite.py` calcule, à la demande, les facteurs ayant le plus contribué à une prédiction —
`TreeExplainer` pour le modèle d'anomalie, `KernelExplainer` pour le modèle de survie (même choix que
dans les notebooks du dépôt solution-IA, `TreeExplainer` n'étant pas compatible avec
`RandomSurvivalForest`). Exposé via `GET /tickets/{id}/explication` et affiché dans l'interface de
supervision (`ui/app.py`, section "Facteurs déclencheurs") sous forme de tableau **et** de graphique —
un technicien voit *pourquoi* une alerte a été levée, pas seulement qu'elle l'a été.

Les deux explainers sont construits une seule fois au démarrage du service (`app/main.py`, `lifespan`) :
leur construction est lente (~13 secondes, mesuré) mais ce coût est absorbé une fois au démarrage du
conteneur plutôt que de retarder la première consultation d'un technicien.

### Non-discrimination entre équipes et postes

Garantie **structurelle**, pas une règle ajoutée après coup : les modèles n'ont jamais accès à une
variable d'équipe, de poste ou d'opérateur (voir la liste `colonnes_features` des deux modèles — capteurs
et attributs machine uniquement). Il ne peut donc pas y avoir de discrimination entre équipes ou postes
puisque cette information n'existe nulle part dans le pipeline de décision.

### Respect de la vie privée

Découle directement du point 1 : aucune donnée personnelle ne transite par ce système.

## 4. Accessibilité de l'interface de supervision

L'interface de décision du technicien (consultation d'une alerte, lecture des facteurs SHAP, validation
ou clôture d'un ticket) est `ui/app.py`, une application Streamlit qui consomme l'API par HTTP.

Choix retenus :
- **Uniquement des composants natifs Streamlit**, aucun HTML/JS injecté (`unsafe_allow_html`) qui
  casserait leur gestion clavier/ARIA native.
- **Label explicite sur chaque champ** (aucun label vide ou masqué).
- **Jamais la couleur seule** : la criticité et le statut sont toujours donnés en texte + icône
  (`⚠️ CRITICITÉ ÉLEVÉE`, `✅ Criticité standard`...).
- **L'information critique en texte simple** (`st.metric`/`st.write`), les tableaux interactifs
  (`st.dataframe`) ne servant qu'en complément : ils ont des limites documentées de compatibilité avec
  les lecteurs d'écran côté Streamlit, on ne s'appuie donc pas sur eux pour l'information dont dépend une
  décision (garde-fou de validation humaine notamment).

**Limite assumée** : Streamlit ne donne pas un contrôle complet sur l'ARIA/l'ordre de focus ; il n'y a
pas eu d'audit avec un lecteur d'écran réel. C'est un choix honnête plutôt qu'une conformité WCAG 2.1 AA
certifiée — cohérent avec le reste de ce projet (voir par exemple la note sur les limites du PSI dans
`app/derive.py`).

## Vérification

- Tests automatisés : `tests/test_explicabilite.py`, `tests/test_ui.py` (fonctionnels, contre la vraie
  API et les vrais modèles — voir leur docstring pour les deux bugs réels qu'ils ont permis de trouver).
- Vérification manuelle : parcours complet réel (création d'alerte → validation → clôture → suivi de
  performance) piloté via `streamlit.testing.v1.AppTest` contre l'API réelle, dérive injectée
  volontairement (×5 sur `vibration_*`) et correctement détectée par `/monitoring/derive`.
