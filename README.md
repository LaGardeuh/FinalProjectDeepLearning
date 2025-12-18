## Description du Projet


1. **Régression multi-sorties** : Prédire 4 variables continues
   - `Project_Grade` : Note du projet (0-100)
   - `Quiz_Score_Avg` : Score moyen des quiz (0-100)
   - `Progress_Percentage` : Pourcentage de progression (0-100)
   - `Satisfaction_Rating` : Note de satisfaction (1-5)

2. **Classification binaire** : Prédire si l'étudiant complète le cours
   - `Completed` : 1 (Completed) / 0 (Not Completed)

## Architecture du Modèle

### Modèle de Régression Multi-Sorties

```
Input (n_features)
    ↓
Dense(256) + BatchNorm + Dropout(0.3) + L2(0.001)
    ↓
Dense(128) + BatchNorm + Dropout(0.3) + L2(0.001)
    ↓
Dense(64) + BatchNorm + Dropout(0.2) + L2(0.001)
    ↓
Dense(32) + BatchNorm + Dropout(0.2) + L2(0.001)
    ↓
Output(4) - Linear activation
```

**Paramètres totaux** : ~93,000 paramètres

### Modèle de Classification Binaire

```
Input (n_features)
    ↓
Dense(256) + BatchNorm + Dropout(0.4) + L2(0.001)
    ↓
Dense(128) + BatchNorm + Dropout(0.4) + L2(0.001)
    ↓
Dense(64) + BatchNorm + Dropout(0.3) + L2(0.001)
    ↓
Dense(32) + BatchNorm + Dropout(0.3) + L2(0.001)
    ↓
Output(1) - Sigmoid activation
```

**Paramètres totaux** : ~89,000 paramètres

### Caractéristiques Techniques

- **Activation** : ReLU pour les couches cachées
- **Initialisation** : He Normal (adapté pour ReLU)
- **Régularisation** : 
  - Dropout progressif (0.4 → 0.3 → 0.2)
  - L2 Regularization (λ = 0.001)
  - Batch Normalization après chaque couche
- **Optimiseur** : Adam (learning rate = 0.001)
- **Loss Functions** :
  - Régression : Mean Squared Error (MSE)
  - Classification : Binary Cross-Entropy
- **Callbacks** :
  - Early Stopping (patience=30 pour régression, 30 pour classification)
  - Reduce Learning Rate on Plateau (factor=0.5, patience=10/15)
  - Model Checkpoint (sauvegarde du meilleur modèle)

## Installation et Utilisation

### 1. Prérequis

- Python 3.8 ou supérieur
- TensorFlow 2.x
- pip

### 2. Installation des dépendances

```bash
pip install tensorflow pandas numpy scikit-learn matplotlib seaborn
```

### 3. Exécution du pipeline complet

```bash
python model_tensorflow.py
```

Le script exécute automatiquement :
1. Prétraitement des données
2. Entraînement du modèle de régression
3. Entraînement du modèle de classification
4. Évaluation complète sur le test set
5. Génération des visualisations

## Pipeline de Données

### 1. Prétraitement (load_and_preprocess_data)

**Étapes effectuées** :

#### a) Feature Engineering (11 nouvelles features créées)

**Features temporelles (4)** :
- `Enrollment_Month` : Mois d'inscription (1-12)
- `Enrollment_DayOfWeek` : Jour de la semaine (0-6)
- `Enrollment_Quarter` : Trimestre (1-4)
- `Days_Since_Enrollment` : Ancienneté en jours

**Features d'engagement (3)** :
- `Assignment_Completion_Rate` : Taux de complétion des devoirs (%)
- `Activity_Score` : Score d'activité global
- `Quiz_Engagement` : Engagement avec les quiz

**Features de ressources (2)** :
- `Hours_Per_Session` : Heures moyennes par session
- `Usage_Intensity` : Intensité d'utilisation du cours

**Features d'interaction (2)** :
- `Social_Engagement` : Engagement social (discussions × interactions)
- `Notification_Response_Rate` : Taux de réponse aux notifications

#### b) Traitement des valeurs manquantes
- Méthode automatique selon le type :
  - Numériques : Remplacement par la **médiane**
  - Catégorielles : Remplacement par le **mode**

#### c) Encodage des variables catégorielles

**Stratégie adaptative** :
- **One-Hot Encoding** : Variables avec ≤ 10 catégories (drop_first=True)
- **Label Encoding** : Variables avec > 10 catégories

**Variables encodées** :
- Gender, Education_Level, Employment_Status
- Device_Type, Internet_Connection_Quality
- Category, Course_Level
- Payment_Mode, Fee_Paid, Discount_Used
- City (Label Encoded car >10 valeurs)
- Course_Name (Label Encoded car >10 valeurs)

#### d) Séparation des données

**Split stratifié (basé sur la cible de classification)** :
- **Train** : 70% (70,000 exemples)
- **Validation** : 15% (15,000 exemples)
- **Test** : 15% (15,000 exemples)

**Avantage** : Garantit la même distribution des classes dans chaque ensemble

#### e) Normalisation

- **StandardScaler** : Centre les données (mean=0, std=1)
- Appliqué sur les features uniquement
- Fit sur train, transform sur val/test (évite le data leakage)

### 2. Variables du Dataset

**Features d'origine** : ~34 features après encodage

**Variables exclues** :
- `Student_ID`, `Name` : Identifiants
- `Enrollment_Date` : Transformée en features temporelles
- Les 5 variables cibles

**Variables cibles** :
- 4 cibles de régression continues
- 1 cible de classification binaire

## Modèles

### Régression Multi-Sorties (build_regression_model)

**Architecture** : [256 → 128 → 64 → 32 → 4]

**Caractéristiques** :
- Dropout progressif : 0.3 → 0.3 → 0.2 → 0.2
- L2 regularization : 0.001 pour toutes les couches
- Activation finale : **Linear** (régression)
- Loss : **MSE** (Mean Squared Error)
- Métriques : MAE, MSE

**Entraînement** :
- Epochs max : 200
- Batch size : 64
- Early stopping : patience=20
- Reduce LR : factor=0.5, patience=10

### Classification Binaire (build_classification_model)

**Architecture** : [256 → 128 → 64 → 32 → 1]

**Caractéristiques** :
- Dropout progressif : 0.4 → 0.4 → 0.3 → 0.3 (plus élevé)
- L2 regularization : 0.001
- Activation finale : **Sigmoid** (probabilités)
- Loss : **Binary Crossentropy**
- Métriques : Accuracy, Precision, Recall, AUC

**Entraînement** :
- Epochs max : 300
- Batch size : 32 (plus petit pour meilleure généralisation)
- Class weights : Calculés automatiquement avec `compute_class_weight`
- Early stopping : patience=30, monitor='val_auc'
- Reduce LR : factor=0.5, patience=10

## Évaluation

### Métriques de Régression

Pour chaque cible (Project_Grade, Quiz_Score_Avg, Progress_Percentage, Satisfaction_Rating) :
- **MAE** (Mean Absolute Error) : Erreur absolue moyenne
- **MSE** (Mean Squared Error) : Erreur quadratique moyenne
- **RMSE** (Root Mean Squared Error) : Racine de MSE
- **R²** (Coefficient de détermination) : Qualité de l'ajustement (0-1)

**Rapport généré** : `results.csv` avec métriques par cible

### Métriques de Classification

- **Accuracy** : Taux de prédictions correctes
- **Precision** : Proportion de vrais positifs parmi les positifs prédits
- **Recall** : Proportion de vrais positifs identifiés
- **F1-Score** : Moyenne harmonique de Precision et Recall
- **AUC-ROC** : Aire sous la courbe ROC (0.5 = hasard, 1.0 = parfait)

**Matrice de confusion** :
```
           Prédiction
           0     1
Réel  0   TN    FP
      1   FN    TP
```

### Visualisations Générées

**Régression** :
1. `training_curves.png` : Loss et MAE (train/val)

**Classification** :
1. `training_curves.png` : Loss et AUC (train/val)
2. `roc_curve.png` : Courbe ROC avec AUC

## Justifications des Choix Techniques

### 1. Architecture Progressive

**Choix** : [256 → 128 → 64 → 32]

**Justification** :
- Réduction progressive permet extraction hiérarchique de features
- 256 neurones au début : capacité suffisante pour patterns complexes
- 32 à la fin : features condensées et abstraites
- 4 couches : profondeur équilibrée (ni sous-ajustement, ni surapprentissage)

### 2. Batch Normalization

**Avantages** :
- Stabilise l'apprentissage
- Permet des learning rates plus élevés
- Agit comme régularisateur
- Accélère la convergence

**Placement** : Après chaque couche Dense, avant Dropout

### 3. Dropout Progressif

**Régression** : 0.3 → 0.3 → 0.2 → 0.2  
**Classification** : 0.4 → 0.4 → 0.3 → 0.3

**Justification** :
- Plus élevé en début : évite co-adaptation des neurones
- Plus faible en fin : préserve les représentations finales
- Classification > Régression : plus sujet au surapprentissage

### 4. Régularisation L2 (0.001)

**Avantages** :
- Pénalise les poids élevés
- Encourage des modèles plus généralisables
- λ=0.001 : bon compromis (ni sous-ajustement, ni sous-régularisation)

### 5. Hyperparamètres

**Learning Rate (0.001)** :
- Valeur standard pour Adam
- Suffisamment petit pour convergence stable
- ReduceLROnPlateau ajuste si nécessaire

**Batch Size** :
- Régression : 64 (bon compromis performance/mémoire)
- Classification : 32 (meilleure généralisation avec déséquilibre)

**Epochs** :
- Régression : 200 (Early Stopping évite le surapprentissage)
- Classification : 300 (plus complexe, besoin de plus de temps)

**Patience** :
- Régression : 20 epochs
- Classification : 30 epochs (plus tolérant aux fluctuations)

### 6. Gestion du Déséquilibre des Classes

**Méthode** : Class weights automatiques

```python
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=[0, 1],
    y=y_train
)
```

**Effet** :
- Pénalise davantage les erreurs sur la classe minoritaire
- Force le modèle à apprendre les deux classes équitablement
- Alternative à SMOTE (plus simple et efficace)

### 7. Feature Engineering

**Pourquoi créer 11 nouvelles features ?**

1. **Ratios et taux** : Capturent l'efficacité (ex: Assignment_Completion_Rate)
2. **Interactions** : Relations entre variables (ex: Social_Engagement)
3. **Temporel** : Saisonnalité et patterns temporels
4. **Agrégations** : Scores globaux (ex: Activity_Score)

**Impact** : Améliore significativement R² et AUC-ROC

### 8. Normalisation StandardScaler

**Avantages** :
- Centre les données (mean=0)
- Même échelle (std=1)
- Essentiel pour convergence des réseaux de neurones
- Évite que certaines features dominent

**Alternative** : MinMaxScaler (non utilisé car moins robuste aux outliers)

## Code de Qualité

### Bonnes Pratiques Implémentées

- Logging informatif (print statements détaillés)
- Reproductibilité (RANDOM_STATE=42)
- Séparation train/validation/test
- Sauvegarde automatique des modèles



## Résultats Attendus

### Fichiers Générés

**Données prétraitées** :
- `X_train.npy`, `X_val.npy`, `X_test.npy` : Features normalisées
- `y_reg_*.npy` : Cibles de régression
- `y_cls_*.npy` : Cibles de classification
- `scaler.pkl` : Scaler pour production
- `feature_names.pkl` : Noms des features
- `config.pkl` : Configuration complète

**Modèles de régression** :
- `final_model.keras` : Modèle final
- `best_model.keras` : Meilleur modèle (val_loss)
- `results.csv` : Métriques par cible
- `training_curves.png` : Visualisations

**Modèles de classification** :
- `final_model.keras` : Modèle final
- `best_model.keras` : Meilleur modèle (val_auc)
- `results.pkl` : Métriques
- `training_curves.png` : Loss et AUC
- `roc_curve.png` : Courbe ROC

## Contexte Académique

**Projet** : Implémentation de réseaux de neurones pour prédiction de complétion de cours  
**Objectifs** :
- Implémentation d'un DNN multi-sorties
- Prétraitement rigoureux avec feature engineering
- Évaluation complète avec métriques appropriées

