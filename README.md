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
├── app/
│   ├── criticite.py      # logique de criticite (anomalie + RUL), degradation gracieuse
│   └── main.py             # API FastAPI : tickets GMAO, authentification, garde-fou
├── models/                  # copies des modeles du depot solution-IA (Git LFS, voir Prerequis)
│   ├── isolation_forest.joblib
│   └── random_survival_forest.joblib
├── tests/
│   ├── test_criticite.py    # 8 tests, modeles mockes
│   └── test_main.py          # 8 tests, API + garde-fou, modeles mockes
├── .env.example
├── .gitattributes            # Git LFS : *.joblib
├── .gitignore
├── requirements.txt
└── README.md
```

> [!NOTE]
> `models/*.joblib` est suivi via **Git LFS** (le modèle de survie seul fait 186 Mo — les forêts de survie stockent la courbe de survie complète à chaque feuille). Ce sont des **copies** des modèles produits par les notebooks du dépôt [Bloc4-...-Solution-IA](https://github.com/<user>/Bloc4-AutoMeca-Maintenance-Predictive-IoT-Solution-IA) — pas une dépendance technique entre dépôts (même principe qu'au Bloc 3), un point de passage explicite entre entraînement et déploiement. **Prérequis** : `git lfs install` avant de cloner, sinon `models/*.joblib` reste un pointeur texte au lieu du vrai fichier.

## Stack technique

- 🐍 **FastAPI / Pydantic** — service de prédiction et de tickets
- 🌲 **scikit-learn / scikit-survival** — chargement des modèles entraînés
- 📦 **Git LFS** — versionnage des modèles (gros fichiers binaires)
- 🧪 **pytest** — tests unitaires (modèles mockés)

## Contenu

- **`app/criticite.py`** — combine les deux modèles : criticité **élevée** si le modèle d'anomalie détecte une dérive (`IsolationForest.predict() == -1`, seuil intégré au modèle) **ou** si la probabilité de panne sous 7 jours dépasse 0,335 (seuil calculé sur les données de validation réelles du modèle RUL — 85ᵉ percentile, ~15 % des observations flaguées, ~38 % des vraies pannes couvertes ; voir le notebook `02_prediction_rul.ipynb` du dépôt solution-IA). **Dégradation gracieuse** : une mesure de capteur manquante est imputée et signalée (`degrade=True`), sauf si plus de 50 % des mesures manquent — dans ce cas la prédiction n'est plus fiable et une exception explicite est levée plutôt qu'un résultat silencieusement faux.
- **`app/main.py`** — API authentifiée (`X-API-Key`) : `POST /predictions/ticket` crée un ticket GMAO (simulé — ce projet n'a pas de vrai système GMAO à intégrer) et le route selon la criticité ; `POST /tickets/{id}/valider` implémente le garde-fou de validation humaine ; `GET /sante` (sans authentification) pour le monitoring d'infrastructure.
- **`tests/`** — 16 tests, modèles mockés. Le module a aussi été vérifié manuellement contre les vrais modèles (voir le journal de développement) : sur 200 mesures réelles issues de périodes saines, 7 % de faux positifs — cohérent avec le paramètre `contamination=0.05` du modèle d'anomalie.
