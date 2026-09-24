# AutoMeca Systems — Maintenance prédictive IoT

AutoMeca Systems conçoit des équipements de freinage pour l'industrie
automobile. Ce projet met en place une plateforme de données pour la
maintenance prédictive de son parc de machines de production : capteurs
IoT en atelier, modélisation et stockage des données, pipelines
d'ingestion, et déploiement d'un modèle prédictif de panne.

Le projet est organisé en dépôts indépendants, un par domaine :

| Dépôt | Contenu |
|---|---|
| [Bloc2-AutoMeca-Maintenance-Predictive-IoT-Architecture-Data](https://github.com/<user>/Bloc2-AutoMeca-Maintenance-Predictive-IoT-Architecture-Data) | Architecture de données : diagramme Edge/Cloud, modèle conceptuel, star schema, dictionnaire de données |
| [Bloc3-AutoMeca-Maintenance-Predictive-IoT-Pipeline-Data](https://github.com/<user>/Bloc3-AutoMeca-Maintenance-Predictive-IoT-Pipeline-Data) | Pipelines d'ingestion et de transformation des données (ELT) |
| [Bloc4-AutoMeca-Maintenance-Predictive-IoT-Solution-IA](https://github.com/<user>/Bloc4-AutoMeca-Maintenance-Predictive-IoT-Solution-IA) | Modèles de maintenance prédictive (entraînement) |
| [Bloc4-AutoMeca-Maintenance-Predictive-IoT-CICD](https://github.com/<user>/Bloc4-AutoMeca-Maintenance-Predictive-IoT-CICD) | Intégration et déploiement continus |

---

## Ce dépôt : Bloc4-AutoMeca-Maintenance-Predictive-IoT-CICD

Service de prédiction et d'intégration GMAO : reçoit des mesures
capteur, calcule un niveau de criticité à partir des deux modèles
entraînés dans [Bloc4-...-Solution-IA](https://github.com/<user>/Bloc4-AutoMeca-Maintenance-Predictive-IoT-Solution-IA), et crée automatiquement un
ticket de maintenance — avec un garde-fou de validation humaine
obligatoire pour les criticités élevées.

> [!IMPORTANT]
> **Garde-fou "human oversight"** (exigé par le sujet) : un ticket de criticité élevée n'est **jamais** exécuté automatiquement. Il reste `en_attente_validation_technicien_senior` jusqu'à l'appel explicite de `POST /tickets/{id}/valider` par un technicien senior. L'automatisation (ce service, le pipeline CI/CD) ne concerne que la création et le routage du ticket — jamais la décision opérationnelle qui en découle.

## Structure

```
Bloc4-AutoMeca-Maintenance-Predictive-IoT-CICD/
├── .github/workflows/
│   └── ci.yml                 # tests -> conteneurisation (API + UI) -> deploiement (differe, voir README)
├── app/
│   ├── criticite.py      # logique de criticite (anomalie + RUL), degradation gracieuse
│   ├── derive.py            # detection de derive (PSI), suivi statistique des variables d'entree
│   ├── explicabilite.py       # facteurs declencheurs (SHAP), a la demande
│   ├── performance.py        # precision / taux de fausses alertes / latence en production
│   └── main.py             # API FastAPI : tickets GMAO, authentification, garde-fou, monitoring
├── ui/                       # interface de supervision (Streamlit) — voir CONFORMITE.md, section 4
│   ├── app.py
│   ├── requirements.txt
│   └── Dockerfile
├── models/                  # copies des modeles du depot solution-IA (Git LFS, voir Prerequis)
│   ├── isolation_forest.joblib
│   ├── random_survival_forest.joblib
│   ├── metriques.json              # metriques + id du run MLflow d'origine (petit fichier, PAS en LFS) pour les tests de non-regression
│   └── reference_distribution.json # distribution des features a l'entrainement (petit fichier, PAS en LFS) pour la derive
├── tests/
│   ├── test_criticite.py       # 8 tests, modeles mockes
│   ├── test_main.py             # 20 tests, API + garde-fou + monitoring, modeles mockes
│   ├── test_derive.py            # 9 tests, detection de derive (PSI)
│   ├── test_explicabilite.py      # 4 tests, tri/selection des facteurs SHAP (explainers mockes)
│   ├── test_performance.py        # 5 tests, calcul precision/faux positifs/latence
│   ├── test_ui.py                  # 3 tests fonctionnels : vraie API + vrais modeles, interface pilotee via AppTest
│   └── test_regression_modele.py # 3 tests, lit models/metriques.json (pas les .joblib)
├── Dockerfile                  # API — construit et verifie (image demarre, /sante repond)
├── docker-compose.yml           # lance API + interface ensemble (local/demo)
├── CONFORMITE.md                # RGPD, ISO 27001, IA ethique, accessibilite — voir sujet Bloc 4
├── .dockerignore
├── .env.example
├── .gitattributes            # Git LFS : *.joblib
├── .gitignore
├── requirements.txt
└── README.md
```

> [!NOTE]
> `models/*.joblib` est suivi via **Git LFS** (le modèle de survie seul fait 186 Mo — les forêts de survie stockent la courbe de survie complète à chaque feuille). Ce sont des **copies** des modèles produits par les notebooks du dépôt [Bloc4-...-Solution-IA](https://github.com/<user>/Bloc4-AutoMeca-Maintenance-Predictive-IoT-Solution-IA) — pas une dépendance technique entre dépôts (même principe qu'au Bloc 3), un point de passage explicite entre entraînement et déploiement. **Prérequis** : `git lfs install` avant de cloner, sinon `models/*.joblib` reste un pointeur texte au lieu du vrai fichier.
>
> **Git LFS n'est pas un registre de modèles** : c'est un mécanisme de stockage de gros fichiers, rien de plus — il ne sait rien des paramètres, métriques ou runs qui ont produit ce fichier. Le vrai registre de modèles (suivi de tous les essais du grid search, versions, cycle de vie) est **MLflow**, dans le dépôt solution-IA. `metriques.json` référence l'ID du run MLflow exact dont chaque modèle est issu (`mlflow_run_id`), pour la traçabilité — sans dépendance technique à MLflow depuis ce dépôt (même principe que le reste : un artefact de sortie explicite, pas un accès direct au tracking store).

## Démarrage local

```bash
cp .env.example .env   # renseigner API_KEY
docker compose up --build
```

- API : http://localhost:8000/docs (documentation interactive Swagger)
- Interface de supervision : http://localhost:8501

Le démarrage du service API prend ~20-30 secondes (préchargement des modèles et des explainers SHAP,
voir `app/main.py`) avant que `/sante` et l'interface ne répondent.

## Stack technique

- 🐍 **FastAPI / Pydantic** — service de prédiction et de tickets
- 🌲 **scikit-learn / scikit-survival** — chargement des modèles entraînés
- 🔍 **SHAP** — explicabilité des alertes (facteurs déclencheurs)
- 🖥️ **Streamlit** — interface de supervision (`ui/`)
- 📦 **Git LFS** — versionnage des modèles (gros fichiers binaires)
- 🧪 **pytest** — tests unitaires et fonctionnels

## Contenu

- **`app/criticite.py`** — combine les deux modèles : criticité **élevée** si le modèle d'anomalie détecte une anomalie (`IsolationForest.predict() == -1`, seuil intégré au modèle) **ou** si la probabilité de panne sous 7 jours dépasse 0,335 (seuil calculé sur les données de validation réelles du modèle RUL — 85ᵉ percentile, ~15 % des observations flaguées, ~38 % des vraies pannes couvertes ; voir le notebook `02_prediction_rul.ipynb` du dépôt solution-IA). **Dégradation gracieuse** : une mesure de capteur manquante est imputée et signalée (`degrade=True`), sauf si plus de 50 % des mesures manquent — dans ce cas la prédiction n'est plus fiable et une exception explicite est levée plutôt qu'un résultat silencieusement faux.
- **`app/derive.py`** — détection de dérive (*data drift*, exigence explicite du sujet) : compare la distribution des variables reçues (fenêtre glissante des 200 dernières requêtes) à leur distribution réelle au moment de l'entraînement (`models/reference_distribution.json`, généré depuis les mêmes données et le même code que les notebooks du dépôt solution-IA). Indicateur : **PSI** (Population Stability Index), standard MLOps — deciles de référence pour les variables continues (capteurs), proportions exactes par valeur pour les variables à faible cardinalité (indicatrices de modèle, `nb_erreurs_7j`). Cette distinction a été ajoutée après une vérification manuelle contre le service réel : les déciles seuls arrondissaient une proportion réelle de 14,76 % à 10 %/20 %, gonflant artificiellement le PSI sur des indicatrices de modèle pourtant non dérivées (voir le module `app/derive.py` pour le détail). Une dérive confirmée ne déclenche pas de réentraînement automatique (jeu d'entraînement de plusieurs Go, hors de portée d'une CI gratuite) mais un signal explicite (`GET /monitoring/derive`) à destination de l'équipe data, qui relance alors les notebooks versionnés et déterministes (`random_state=42`) du dépôt solution-IA — reproductibilité satisfaite sans orchestrateur dédié pour un réentraînement occasionnel.
- **`app/explicabilite.py`** — facteurs déclencheurs d'une alerte (méthodes SHAP, exigence "IA éthique" du sujet) : `TreeExplainer` pour le modèle d'anomalie, `KernelExplainer` pour le modèle de survie (même choix que dans les notebooks — `TreeExplainer` n'est pas compatible avec `RandomSurvivalForest`). Les deux explainers sont construits une seule fois au démarrage du service (coût mesuré ~13s) plutôt qu'à la première consultation d'un technicien. Exposé via `GET /tickets/{id}/explication`.
- **`app/performance.py`** — précision et taux de fausses alertes **mesurés en production** (pas l'AUC/C-index hors ligne des notebooks) : à partir des tickets clôturés par un technicien avec un résultat réel (`POST /tickets/{id}/cloturer`), plus la latence moyenne du service. Mesure volontairement différente d'un taux de faux positifs classique : en conditions réelles, seuls les tickets effectivement créés sont observables, jamais les vrais négatifs.
- **`app/main.py`** — API authentifiée (`X-API-Key`) : `POST /predictions/ticket` crée un ticket GMAO (simulé — ce projet n'a pas de vrai système GMAO à intégrer) et le route selon la criticité ; `GET /tickets` liste les tickets (filtrables par statut) ; `POST /tickets/{id}/valider` implémente le garde-fou de validation humaine ; `POST /tickets/{id}/cloturer` enregistre le résultat réel (panne confirmée / fausse alerte) ; `GET /tickets/{id}/explication` renvoie les facteurs SHAP ; `GET /monitoring/derive` et `GET /monitoring/performance` exposent le suivi de dérive et de performance ; `GET /sante` (sans authentification) pour le monitoring d'infrastructure. Précharge modèles, référence de dérive et explainers SHAP au démarrage (`lifespan`), pas à la première requête.
- **`ui/app.py`** — interface de supervision (Streamlit, exigence "accessibilité" du sujet — voir `CONFORMITE.md` section 4) : liste et filtre des tickets, détail d'un ticket (criticité, probabilité, facteurs SHAP en tableau + graphique, mesures brutes), garde-fou de validation humaine, clôture avec résultat réel, tableau de bord dérive/performance. Consomme l'API du même dépôt par HTTP uniquement (pas de dépendance technique directe).
- **`docker-compose.yml`** — lance l'API et l'interface ensemble pour un usage local ou une démonstration.
- **`tests/`** — 55 tests. `test_criticite.py`, `test_main.py`, `test_derive.py`, `test_explicabilite.py`, `test_performance.py` : modèles/référence mockés. `test_ui.py` : **fonctionnels**, démarrent la vraie API avec les vrais modèles et pilotent l'interface réelle via `streamlit.testing.v1.AppTest` — ont trouvé deux bugs réels invisibles aux tests mockés (voir leur docstring) : un ticket nouvellement créé n'apparaissait pas dans la liste sans action supplémentaire (ordre d'exécution des onglets dans le script), et un message de confirmation était effacé par un `st.rerun()` avant d'être jamais affiché. Vérifié aussi manuellement contre le service réel : sur 200 mesures réelles issues de périodes saines, 7 % de faux positifs de criticité (cohérent avec `contamination=0.05`) ; injection contrôlée d'une dérive réelle sur `vibration_*` (×5) correctement détectée (PSI ≈ 8) pendant que les variables non affectées restent stables.
