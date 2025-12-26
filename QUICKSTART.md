# 🚀 Guide de Démarrage Rapide

Ce guide vous permet de démarrer rapidement avec la plateforme de Federated Summarization.

## 📋 Prérequis

### Matériel Recommandé
- **CPU**: 8+ cores
- **RAM**: 16GB minimum (32GB recommandé)
- **GPU**: 3x NVIDIA GPUs avec 8GB+ VRAM (ou utiliser CPU)
- **Stockage**: 50GB disponible

### Logiciels Requis
- Docker & Docker Compose
- Python 3.9+
- CUDA 11.8+ (pour GPU)

## 🔧 Installation

### 1. Cloner le projet

```bash
cd project
```

### 2. Configuration des variables d'environnement

```bash
cp .env.example .env
```

Éditez `.env` selon vos besoins:
- Ajustez les adresses des serveurs
- Configurez les paramètres GPU
- Modifiez les hyperparamètres si nécessaire

### 3. Vérifier Docker

```bash
docker --version
docker-compose --version
```

## 🏃 Démarrage

### Option 1: Démarrage Complet (Recommandé)

Lance tous les services:

```bash
docker-compose up -d
```

Vérifier les logs:

```bash
docker-compose logs -f
```

### Option 2: Démarrage Progressif

1. **Infrastructure Kafka**
```bash
docker-compose up -d zookeeper kafka
sleep 10  # Attendre que Kafka soit prêt
```

2. **Ingestion des Données**
```bash
docker-compose up producer
# Attendre la fin de l'ingestion
```

3. **Serveur et Clients Flower**
```bash
docker-compose up -d flower-server
sleep 5
docker-compose up -d health-client finance-client legal-client
```

4. **Dashboard**
```bash
docker-compose up -d dashboard
```

## 📊 Accès aux Services

| Service | URL | Description |
|---------|-----|-------------|
| Dashboard | http://localhost:8501 | Interface Streamlit |
| Flower Server | http://localhost:8080 | API Flower (si disponible) |
| Kafka | localhost:9092 | Broker Kafka |

## 🔍 Surveillance

### Voir les Logs

```bash
# Tous les services
docker-compose logs -f

# Service spécifique
docker-compose logs -f flower-server
docker-compose logs -f health-client
```

### Vérifier le Statut

```bash
docker-compose ps
```

### Statistiques des Conteneurs

```bash
docker stats
```

## 📈 Suivi de l'Entraînement

### Dashboard Streamlit

1. Ouvrir http://localhost:8501
2. Onglet "Training Progress" pour voir:
   - Courbes de loss en temps réel
   - Métriques par client
   - Progression des rounds

### Logs du Serveur

```bash
docker-compose logs -f flower-server | grep "Round"
```

Vous verrez:
```
Round 1: Aggregating results from 3 clients
Round 1: Aggregated Loss = 2.1543
...
```

## 🧪 Test de l'Inférence

### Via le Dashboard

1. Aller sur l'onglet "Test Inference"
2. Coller un document
3. Cliquer sur "Générer le Résumé"

### Via Script Python

```bash
docker-compose exec dashboard python inference/inference_pipeline.py
```

## 🛑 Arrêter les Services

### Arrêt Simple

```bash
docker-compose stop
```

### Arrêt avec Nettoyage

```bash
docker-compose down
```

### Arrêt et Suppression des Volumes

⚠️ **ATTENTION**: Ceci supprimera tous les checkpoints et données!

```bash
docker-compose down -v
```

## 🔧 Dépannage

### Problème: Kafka ne démarre pas

**Solution**:
```bash
docker-compose down
docker volume rm project_kafka-data project_zookeeper-data
docker-compose up -d zookeeper kafka
```

### Problème: Clients ne se connectent pas

**Solution**:
1. Vérifier que le serveur est bien démarré:
```bash
docker-compose logs flower-server
```

2. Redémarrer les clients:
```bash
docker-compose restart health-client finance-client legal-client
```

### Problème: Mémoire GPU insuffisante

**Solution 1**: Activer la quantification 4-bit dans `.env`:
```bash
LOAD_IN_4BIT=true
```

**Solution 2**: Réduire le batch size:
```bash
BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=8
```

**Solution 3**: Utiliser CPU (plus lent):
Retirer la section `deploy.resources` dans `docker-compose.yml`

### Problème: Producer échoue à charger les datasets

**Solution**:
```bash
# Vérifier la connexion internet
docker-compose exec producer ping -c 3 huggingface.co

# Augmenter le timeout
docker-compose restart producer
```

## 📚 Ressources Supplémentaires

### Documentation Complète

Voir [README.md](README.md) pour:
- Architecture détaillée
- Explication des 11 piliers
- Configuration avancée

### Notebook Pédagogique

Ouvrir `notebooks/analysis_and_theory.ipynb` pour:
- Théorie du Federated Learning
- Explication de FedProx et LoRA
- Analyse des résultats
- Visualisations interactives

### Structure du Code

```
project/
├── configs/          # Configuration centralisée
├── data_ingestion/   # Producer Kafka
├── federated_learning/ # Clients & Serveur Flower
├── model/            # LED model wrapper
├── evaluation/       # Métriques ROUGE & BERTScore
├── monitoring/       # Dashboard Streamlit
├── inference/        # Pipeline d'inférence
└── utils/            # Checkpoints, offsets, logging
```

## 💡 Conseils Pro

### Performance

1. **GPU**: Utilisez des GPUs avec 16GB+ VRAM pour de meilleures performances
2. **Batch Size**: Augmentez si vous avez plus de VRAM
3. **LoRA Rank**: Augmentez à 32 pour plus de capacité (plus lent)

### Production

1. **Monitoring**: Ajoutez Prometheus/Grafana
2. **Logging**: Configurez un système centralisé (ELK)
3. **Backup**: Sauvegardez régulièrement `/app/checkpoints`
4. **Security**: Activez TLS pour Kafka et Flower

### Expérimentation

1. **Hyperparamètres**: Modifiez dans `.env`
2. **Datasets**: Changez les datasets dans `docker-compose.yml`
3. **Rounds**: Augmentez `NUM_ROUNDS` pour plus d'entraînement
4. **FedProx µ**: Ajustez selon l'hétérogénéité des données

## ❓ Support

Pour toute question ou problème:
1. Vérifier les logs: `docker-compose logs -f`
2. Consulter le README.md
3. Examiner le notebook pédagogique
4. Vérifier les issues GitHub (si applicable)

---

**Happy Federated Learning! 🎉**
