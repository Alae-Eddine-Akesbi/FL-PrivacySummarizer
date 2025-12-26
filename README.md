# 🚀 Federated Privacy-Preserving Summarization Platform

## 📋 Vue d'ensemble

Plateforme industrielle de résumé de documents longs utilisant le Federated Learning pour préserver la confidentialité des données. Trois départements (Santé, Finance, Juridique) collaborent pour entraîner un modèle sans partager leurs données sensibles.

## 🏗️ Architecture du Projet

```
project/
├── README.md                           # Documentation principale
├── requirements.txt                    # Dépendances Python
├── docker-compose.yml                  # Orchestration des services
├── .env.example                        # Variables d'environnement template
│
├── configs/                            # Configuration centralisée
│   ├── __init__.py
│   ├── model_config.py                 # Configuration du modèle LED
│   ├── federated_config.py             # Paramètres FedProx & FL
│   └── kafka_config.py                 # Configuration Kafka
│
├── data_ingestion/                     # Phase 1: Ingestion Kafka
│   ├── __init__.py
│   ├── producer.py                     # Single Producer robuste
│   ├── topic_manager.py                # Gestion des topics Kafka
│   └── data_loader.py                  # Chargement des datasets
│
├── federated_learning/                 # Cœur du système FL
│   ├── __init__.py
│   ├── flower_client.py                # FlowerClient avec FedProx + LoRA
│   ├── flower_server.py                # Stratégie d'agrégation personnalisée
│   ├── fedprox_optimizer.py            # Implémentation FedProx
│   └── lora_manager.py                 # Gestion PEFT/LoRA
│
├── model/                              # Gestion du modèle
│   ├── __init__.py
│   ├── led_summarizer.py               # Wrapper pour LED avec global_attention
│   ├── model_loader.py                 # Chargement + Quantification 4-bit
│   └── tokenizer_utils.py              # Utilitaires de tokenization
│
├── evaluation/                         # Métriques et évaluation
│   ├── __init__.py
│   ├── metrics.py                      # ROUGE + BERTScore
│   ├── aggregator.py                   # Agrégation des métriques globales
│   └── evaluator.py                    # Pipeline d'évaluation
│
├── utils/                              # Utilitaires transversaux
│   ├── __init__.py
│   ├── checkpoint_manager.py           # Sauvegarde/Reprise LoRA
│   ├── kafka_offset_manager.py         # Gestion des offsets Kafka
│   ├── logger.py                       # Logging structuré
│   └── helpers.py                      # Fonctions auxiliaires
│
├── monitoring/                         # Dashboard & Visualisation
│   ├── __init__.py
│   ├── streamlit_dashboard.py          # Interface Streamlit
│   ├── metrics_collector.py            # Collecte des métriques
│   └── visualization.py                # Génération des graphiques
│
├── inference/                          # Phase 2: Inférence temps réel
│   ├── __init__.py
│   ├── kafka_consumer.py               # Consumer pour inférence
│   └── inference_pipeline.py           # Pipeline de résumé
│
├── docker/                             # Dockerfiles et scripts
│   ├── Dockerfile.client               # Image pour Flower Client
│   ├── Dockerfile.server               # Image pour Flower Server
│   ├── Dockerfile.producer             # Image pour Producer
│   ├── Dockerfile.dashboard            # Image pour Dashboard
│   └── entrypoint.sh                   # Script d'initialisation
│
├── scripts/                            # Scripts d'automatisation
│   ├── setup_kafka_topics.sh           # Création des topics
│   ├── run_training.sh                 # Lancement de l'entraînement
│   └── test_inference.sh               # Test du pipeline d'inférence
│
├── notebooks/                          # Analyse pédagogique
│   └── analysis_and_theory.ipynb       # Notebook Jupyter complet
│
└── tests/                              # Tests unitaires
    ├── __init__.py
    ├── test_producer.py
    ├── test_client.py
    └── test_metrics.py
```

## 🎯 Les 11 Piliers Techniques

### 1️⃣ Modèle: LED Large Book Summary
- **Modèle**: `pszemraj/led-large-book-summary`
- **Global Attention**: Gestion automatique du masque d'attention sur `<s>`
- **Capacité**: Textes jusqu'à 16,384 tokens

### 2️⃣ Algorithmes: FedProx + LoRA
- **FedProx**: Terme de pénalité proximale µ = 0.01
- **PEFT/LoRA**: Adapters rank=16, alpha=32
- **Optimisation**: Réduction de 99% des paramètres entraînables

### 3️⃣ Infrastructure Streaming (Kafka)
- **Phase 1 (Fine-tuning)**: 3 topics dédiés (health, finance, legal)
- **Phase 2 (Inférence)**: Pipeline temps réel
- **Résilience**: Gestion des offsets + replay

### 4️⃣ Dynamique de Training
- **Steps fixes**: 50 pas par round
- **Équilibrage**: Distribution uniforme entre clients
- **Convergence**: 10 rounds globaux

### 5️⃣ Datasets (20k lignes/client)
| Client | Dataset | Topic Kafka |
|--------|---------|-------------|
| Santé | `ccdv/pubmed-summarization` | `health-documents` |
| Finance | `mrSoul7766/ECTSum` | `finance-documents` |
| Juridique | `FiscalNote/billsum` | `legal-documents` |

### 6️⃣ Évaluation
- **Métriques**: ROUGE-1, ROUGE-2, ROUGE-L, BERTScore
- **Calcul**: Local (par client) + Agrégation globale
- **Fréquence**: Après chaque round

### 7️⃣ Résilience
- **Checkpoints LoRA**: Sauvegarde après chaque round
- **Offsets Kafka**: Commit automatique post-traitement
- **Reprise**: Récupération complète de l'état

### 8️⃣ Monitoring
- **Dashboard Streamlit**: Courbes de loss en temps réel
- **Interface de test**: Résumé interactif
- **Métriques**: Visualisation des performances

### 9️⃣ Orchestration Docker
```
Services:
├── zookeeper          # Coordination Kafka
├── kafka              # Message broker
├── flower-server      # Serveur Flower
├── health-client      # Client Santé
├── finance-client     # Client Finance
├── legal-client       # Client Juridique
├── producer           # Ingestion des données
└── dashboard          # Interface Streamlit
```

### 🔟 Ingestion Intelligente
- **Single Producer**: Un seul point d'entrée
- **Routage**: Distribution vers 3 topics selon le type
- **Robustesse**: Retry automatique + gestion d'erreurs

### 1️⃣1️⃣ Approche Hybride
- **Production**: Scripts `.py` modulaires
- **Pédagogie**: Notebook `.ipynb` avec théorie

## 🚀 Démarrage Rapide

### Prérequis
```bash
- Docker & Docker Compose
- Python 3.9+
- CUDA 11.8+ (pour GPU)
- 16GB RAM minimum
```

### Installation

1. **Cloner et configurer**
```bash
cd project
cp .env.example .env
# Éditer .env avec vos paramètres
```

2. **Lancer l'infrastructure**
```bash
docker-compose up -d
```

3. **Vérifier les logs**
```bash
docker-compose logs -f flower-server
```

4. **Accéder au dashboard**
```
http://localhost:8501
```

## 📊 Utilisation

### Phase 1: Fine-tuning Fédéré

1. **Ingestion des données**
```bash
docker-compose exec producer python data_ingestion/producer.py
```

2. **Lancement de l'entraînement**
```bash
docker-compose exec flower-server python federated_learning/flower_server.py
```

3. **Monitoring**
- Dashboard: http://localhost:8501
- Flower UI: http://localhost:8080

### Phase 2: Inférence Temps Réel

```bash
docker-compose exec dashboard python inference/inference_pipeline.py
```

## 🔧 Configuration

### Variables d'environnement (.env)

```bash
# Kafka
KAFKA_BOOTSTRAP_SERVERS=kafka:9092
KAFKA_HEALTH_TOPIC=health-documents
KAFKA_FINANCE_TOPIC=finance-documents
KAFKA_LEGAL_TOPIC=legal-documents

# Flower
FLOWER_SERVER_ADDRESS=flower-server:8080
NUM_ROUNDS=10
STEPS_PER_ROUND=50

# Model
MODEL_NAME=pszemraj/led-large-book-summary
MAX_INPUT_LENGTH=8192
MAX_TARGET_LENGTH=512
LOAD_IN_4BIT=true

# FedProx
FEDPROX_MU=0.01
LEARNING_RATE=2e-5

# LoRA
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.05
```

## 📈 Métriques et Performance

### Attendues après convergence:
- **ROUGE-1**: ~0.45
- **ROUGE-2**: ~0.22
- **ROUGE-L**: ~0.38
- **BERTScore F1**: ~0.85

### Optimisations VRAM:
- **Quantification 4-bit**: ~8GB VRAM par client
- **Gradient Checkpointing**: Activé
- **LoRA**: 0.5% des paramètres

## 🛡️ Sécurité et Confidentialité

- ✅ **Aucun partage de données brutes**
- ✅ **Agrégation sécurisée des gradients**
- ✅ **Isolation des clients (Docker)**
- ✅ **Chiffrement des communications (TLS possible)**

## 📚 Documentation Technique

Voir le notebook `notebooks/analysis_and_theory.ipynb` pour:
- Explication théorique du Federated Learning
- Détails sur FedProx vs FedAvg
- Analyse des résultats
- Visualisations avancées

## 🤝 Contribution

Ce projet suit les standards:
- **Type Hints**: Obligatoires
- **Docstrings**: Style Google
- **Tests**: Coverage > 80%
- **Linting**: Black + Flake8

## 📄 Licence

MIT License - Voir LICENSE file

## 👥 Contact

Pour toute question ou support, contacter l'équipe AI Solutions Architecture.

---

**Version**: 1.0.0  
**Dernière mise à jour**: Décembre 2024  
**Status**: Production Ready ✅
