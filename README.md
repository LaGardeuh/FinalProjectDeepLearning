# Prédiction de Complétion de Cours - TensorFlow

## Description du Projet

Ce projet implémente un réseau de neurones profond (Deep Neural Network) multi-tâches utilisant **TensorFlow/Keras** pour prédire la complétion de cours en ligne. Le modèle résout simultanément :

1. **Régression multi-sorties** : Prédire 4 variables continues
   - `Project_Grade` : Note du projet (0-100)
   - `Quiz_Score_Avg` : Score moyen des quiz (0-100)
   - `Progress_Percentage` : Pourcentage de progression (0-100)
   - `Satisfaction_Rating` : Note de satisfaction (1-5)

2. **Classification binaire** : Prédire si l'étudiant complète le cours
   - `Completed` : Oui (1) / Non (0)

## Architecture du Modèle

### Architecture Multi-Tâches (Multi-Task Learning)

```
Input (n_features)
    ↓
[Shared Layers] - Tronc commun
    ├─ Dense(256) + BatchNorm + Dropout
    ├─ Dense(128) + BatchNorm + Dropout
    ├─ Dense(64) + BatchNorm + Dropout
    └─ Dense(32) + BatchNorm + Dropout
    ↓
    ├─────────────────┬─────────────────┐
    ↓                 ↓                 ↓
[Regression Branch] [Classification Branch]
    ↓                 ↓
Dense(64,32)     Dense(64,32)
    ↓                 ↓
Output(4)        Output(1)
sigmoid          sigmoid
```

### Caractéristiques Techniques

- **Activation** : ReLU pour les couches cachées, Sigmoid pour les sorties
- **Régularisation** : 
  - Dropout (30%)
  - L2 Regularization (0.001)
  - Batch Normalization
- **Optimiseur** : Adam (learning rate = 0.001)
- **Loss Functions** :
  - Régression : Mean Squared Error (MSE)
  - Classification : Binary Cross-Entropy
- **Callbacks** :
  - Early Stopping (patience=15)
  - Reduce Learning Rate on Plateau (patience=10)
  - Model Checkpoint (sauvegarde du meilleur modèle)

## Structure du Projet

```
course-completion-prediction/
│
├── config.py                 # Configuration et constantes
├── preprocessing.py          # Prétraitement des données
├── model_tensorflow.py       # Modèle TensorFlow/Keras
├── evaluation.py             # Évaluation et visualisation
├── main.py                   # Script principal
├── requirements.txt          # Dépendances Python
├── README.md                 # Documentation
│
├── data/
│   └── Course_Completion_Prediction.csv
│
└── outputs/
    ├── models/               # Modèles sauvegardés
    ├── plots/                # Graphiques générés
    └── metrics_report_*.txt  # Rapports d'évaluation
```

## Installation et Utilisation

### 1. Prérequis

- Python 3.8 ou supérieur
- pip

### 2. Installation des dépendances

```bash
pip install -r requirements.txt
```

### 3. Exécution du pipeline complet

```bash
python main.py
```

### Options de ligne de commande

```bash
# Avec des paramètres personnalisés
python main.py --epochs 150 --batch-size 256 --data-path /chemin/vers/data.csv
```

## Pipeline de Données

### 1. Prétraitement

Le module `preprocessing.py` effectue :

- **Vérification de qualité** : Détection des valeurs manquantes et doublons
- **Encodage** : LabelEncoder pour les variables catégorielles
- **Normalisation** : 
  - StandardScaler pour les features
  - Min-Max normalization (0-1) pour les targets de régression
- **Split** : 80% train / 20% test avec stratification

### 2. Features Utilisées

**Variables numériques** (19 features) :
- Démographiques : Age
- Cours : Course_Duration_Days, Instructor_Rating
- Engagement : Login_Frequency, Average_Session_Duration_Min, Video_Completion_Rate
- Interaction : Discussion_Participation, Peer_Interaction_Score
- Activité : Time_Spent_Hours, Days_Since_Last_Login, Rewatch_Count
- Performance : Quiz_Attempts, Assignments_Missed
- Paiement : Payment_Amount, App_Usage_Percentage
- Support : Reminder_Emails_Clicked, Support_Tickets_Raised, Notifications_Checked

**Variables catégorielles** (12 features) :
- Gender, Education_Level, Employment_Status
- City, Device_Type, Internet_Connection_Quality
- Course_Name, Category, Course_Level
- Payment_Mode, Fee_Paid, Discount_Used

**Total : 31 features**

### 3. Variables Exclues

- Student_ID, Name (identifiants)
- Course_ID, Enrollment_Date (métadonnées)
- Assignments_Submitted (risque de data leakage)
- Les 5 variables cibles

## 📈 Évaluation

### Métriques de Régression

Pour chaque target :
- **MAE** (Mean Absolute Error)
- **MSE** (Mean Squared Error)
- **RMSE** (Root Mean Squared Error)
- **R²** (Coefficient de détermination)

### Métriques de Classification

- **Accuracy** : Taux de prédictions correctes
- **Precision** : Proportion de vrais positifs parmi les positifs prédits
- **Recall** : Proportion de vrais positifs identifiés
- **F1-Score** : Moyenne harmonique de Precision et Recall
- **AUC-ROC** : Aire sous la courbe ROC

### Visualisations Générées

1. **training_history.png** : Courbes d'apprentissage (loss, métriques)
2. **regression_predictions.png** : Scatter plots prédictions vs réalité
3. **regression_residuals.png** : Analyse des résidus
4. **classification_results.png** : Matrice de confusion, courbe ROC, distribution des probabilités

## 🎯 Justifications des Choix Techniques

### Pourquoi un Modèle Multi-Tâches ?

1. **Partage de représentations** : Les features communes (engagement, démographie) sont pertinentes pour les deux tâches
2. **Régularisation implicite** : L'apprentissage simultané réduit le surapprentissage
3. **Efficacité** : Un seul modèle au lieu de 5 modèles séparés
4. **Cohérence** : Les prédictions sont liées (un étudiant avec de bonnes notes a plus de chances de compléter)

### Architecture en Détail

**Tronc commun (4 couches)** :
- Extrait des features générales utiles aux deux tâches
- Profondeur suffisante pour capturer des patterns complexes
- BatchNorm stabilise l'apprentissage
- Dropout évite le surapprentissage

**Branches spécialisées (2 couches chacune)** :
- Permet d'apprendre des features spécifiques à chaque tâche
- Couches plus petites (64→32) car elles affinent les représentations

**Activations** :
- ReLU : Standard pour les couches cachées, évite le gradient vanishing
- Sigmoid (sorties) : Approprié pour régression normalisée [0,1] et classification binaire

### Hyperparamètres

**Choix du Learning Rate (0.001)** :
- Valeur standard pour Adam
- Équilibre entre vitesse de convergence et stabilité
- ReduceLROnPlateau ajuste automatiquement si nécessaire

**Dropout (30%)** :
- Taux modéré pour réseau profond
- Prévient le co-adaptation des neurones
- Balance régularisation et capacité d'apprentissage

**Batch Size (128)** :
- Compromis entre :
  - Stabilité du gradient (batches plus grands)
  - Généralisation (batches plus petits)
  - Performance computationnelle

**Early Stopping (patience=15)** :
- 15 époques sans amélioration avant arrêt
- Empêche le surapprentissage
- Économise du temps de calcul

### Normalisation des Données

**StandardScaler pour les features** :
- Centre les données (moyenne=0, variance=1)
- Crucial pour la convergence des réseaux de neurones
- Évite que certaines features dominent

**Min-Max [0,1] pour les targets de régression** :
- Facilite l'apprentissage avec activation sigmoid
- Homogénéise les échelles différentes (0-100 vs 1-5)
- Améliore la stabilité numérique

## 🔧 Code de Qualité

### Standards Respectés

- **PEP-8** : Formatage du code Python
- **Type hints** : Annotations de types pour clarté
- **Docstrings** : Documentation complète de chaque fonction/classe
- **Modularité** : Séparation en modules logiques
- **Commentaires** : Explications des choix techniques

### Bonnes Pratiques

- Gestion des erreurs
- Logging informatif
- Reproductibilité (random_state=42)
- Séparation train/validation/test
- Callbacks pour monitoring

## 📝 Résultats Attendus

Le modèle génère automatiquement :

1. **Modèles sauvegardés** :
   - `best_model_tensorflow.h5` : Meilleur modèle pendant l'entraînement
   - `final_model_tensorflow.h5` : Modèle final

2. **Rapport de métriques** :
   - Fichier texte avec toutes les métriques
   - Configuration du modèle
   - Timestamp

3. **Visualisations** :
   - 4 graphiques PNG détaillés
   - Haute résolution (DPI=100)

## 🎓 Contexte Académique

**Projet** : Implémentation de réseaux de neurones pour régression + classification  
**Framework** : TensorFlow/Keras  
**Date** : Décembre 2024  
**Objectifs** :
- ✅ Implémentation d'un MLP/DNN avec TensorFlow
- ✅ Prétraitement rigoureux des données
- ✅ Évaluation et optimisation des modèles
- ✅ Code de qualité, documenté et modulaire
- ✅ Justification de toutes les décisions techniques

## 📚 Références

- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [Keras Guide](https://keras.io/guides/)
- [Multi-Task Learning](https://en.wikipedia.org/wiki/Multi-task_learning)
- [Deep Learning Book](https://www.deeplearningbook.org/)

## 👤 Auteur

**Keralo**  
Étudiant en Computer Science/Engineering - ESAIP  
Spécialisation : Intelligence Artificielle

---

*Ce projet démontre une compréhension approfondie des réseaux de neurones, du prétraitement de données, et des bonnes pratiques de développement en Deep Learning.*