
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pickle
import os
from datetime import datetime

# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score
)
from sklearn.utils.class_weight import compute_class_weight

# TensorFlow et Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks, regularizers
from tensorflow.keras.optimizers import Adam

warnings.filterwarnings('ignore')

# Configuration
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

print(f"Random state : {RANDOM_STATE}")

def load_and_preprocess_data(filepath):
    """
    Charge et prépare les données pour l'entraînement.

    Args:
        filepath: Chemin vers le fichier CSV

    Returns:
        Données prétraitées prêtes pour l'entraînement
    """

    print("\n" + "=" * 80)
    print(" ÉTAPE 1: CHARGEMENT ET EXPLORATION")
    print("=" * 80)

    # Chargement
    df = pd.read_csv(filepath)
    print(f"\nDataset chargé : {df.shape[0]:,} lignes × {df.shape[1]} colonnes")

    # Définition des variables
    ID_COLS = ['Student_ID', 'Name']
    REGRESSION_TARGETS = [
        'Project_Grade',
        'Quiz_Score_Avg',
        'Progress_Percentage',
        'Satisfaction_Rating'
    ]
    CLASSIFICATION_TARGET = 'Completed'
    ALL_TARGETS = REGRESSION_TARGETS + [CLASSIFICATION_TARGET]
    FEATURE_COLS = [col for col in df.columns if col not in ID_COLS + ALL_TARGETS]

    print(f"\n{len(FEATURE_COLS)} features identifiées")
    print(f"{len(REGRESSION_TARGETS)} cibles de régression")
    print(f"1 cible de classification")

    print("\n" + "=" * 80)
    print(" ÉTAPE 2: FEATURE ENGINEERING")
    print("=" * 80)

    df_processed = df.copy()

    # Traitement de la date d'inscription
    if 'Enrollment_Date' in df_processed.columns:
        print("\n1. Traitement des dates...")
        df_processed['Enrollment_Date'] = pd.to_datetime(
            df_processed['Enrollment_Date'],
            format='%d-%m-%Y'
        )
        df_processed['Enrollment_Month'] = df_processed['Enrollment_Date'].dt.month
        df_processed['Enrollment_DayOfWeek'] = df_processed['Enrollment_Date'].dt.dayofweek
        df_processed['Enrollment_Quarter'] = df_processed['Enrollment_Date'].dt.quarter
        reference_date = df_processed['Enrollment_Date'].max()
        df_processed['Days_Since_Enrollment'] = (
                reference_date - df_processed['Enrollment_Date']
        ).dt.days
        df_processed = df_processed.drop(columns=['Enrollment_Date'])
        print("   4 nouvelles features temporelles créées")

    # Features d'engagement
    print("\n2. Création de features d'engagement...")
    df_processed['Assignment_Completion_Rate'] = (
                                                         df_processed['Assignments_Submitted'] /
                                                         (df_processed['Assignments_Submitted'] + df_processed[
                                                             'Assignments_Missed'])
                                                 ) * 100
    df_processed['Activity_Score'] = (
            df_processed['Login_Frequency'] *
            df_processed['Average_Session_Duration_Min'] *
            df_processed['Video_Completion_Rate'] / 100
    )
    df_processed['Quiz_Engagement'] = (
            df_processed['Quiz_Attempts'] * df_processed['Quiz_Score_Avg']
    )
    print("3 features d'engagement créées")

    # Features de ressources
    print("\n3. Création de features de ressources...")
    df_processed['Hours_Per_Session'] = (
            df_processed['Time_Spent_Hours'] /
            df_processed['Login_Frequency'].replace(0, 1)
    )
    df_processed['Usage_Intensity'] = (
            df_processed['Time_Spent_Hours'] /
            df_processed['Course_Duration_Days'].replace(0, 1)
    )
    print("2 features de ressources créées")

    # Features d'interaction
    print("\n4. Création de features d'interaction...")
    df_processed['Social_Engagement'] = (
            df_processed['Discussion_Participation'] *
            df_processed['Peer_Interaction_Score']
    )
    df_processed['Notification_Response_Rate'] = (
            df_processed['Reminder_Emails_Clicked'] /
            df_processed['Notifications_Checked'].replace(0, 1)
    )
    print("2 features d'interaction créées")

    # Gestion des valeurs infinies
    df_processed = df_processed.replace([np.inf, -np.inf], np.nan)

    print(f"\nTotal: 11 nouvelles features créées")
    print(f"Nouvelle forme: {df_processed.shape}")

    # ========================================================================
    print("\n" + "=" * 80)
    print(" ÉTAPE 3: TRAITEMENT DES VALEURS MANQUANTES")
    print("=" * 80)

    missing_count = df_processed.isnull().sum().sum()
    if missing_count > 0:
        print(f"\n{missing_count} valeurs manquantes détectées")
        print("   Application des stratégies de remplissage...")

        for col in df_processed.columns:
            if df_processed[col].isnull().sum() > 0:
                if df_processed[col].dtype in ['int64', 'float64']:
                    df_processed[col].fillna(df_processed[col].median(), inplace=True)
                else:
                    df_processed[col].fillna(df_processed[col].mode()[0], inplace=True)

        print(f"Toutes les valeurs manquantes traitées")

    # ========================================================================
    print("\n" + "=" * 80)
    print(" ÉTAPE 4: ENCODAGE DES VARIABLES")
    print("=" * 80)

    # Identification des features catégorielles
    categorical_features = df_processed.select_dtypes(include=['object']).columns.tolist()
    categorical_features = [
        col for col in categorical_features
        if col not in ID_COLS + ALL_TARGETS
    ]

    # Stratégie d'encodage
    low_cardinality = [col for col in categorical_features if df_processed[col].nunique() <= 10]
    high_cardinality = [col for col in categorical_features if df_processed[col].nunique() > 10]

    print(f"\n1. One-Hot Encoding: {len(low_cardinality)} variables")
    if low_cardinality:
        df_processed = pd.get_dummies(
            df_processed,
            columns=low_cardinality,
            prefix=low_cardinality,
            drop_first=True
        )
        print("One-Hot Encoding appliqué")

    print(f"\n2. Label Encoding: {len(high_cardinality)} variables")
    label_encoders = {}
    if high_cardinality:
        for col in high_cardinality:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col])
            label_encoders[col] = le
        print("Label Encoding appliqué")

    # Encodage de la cible de classification
    print("\n3. Encodage de la cible de classification...")
    df_processed[CLASSIFICATION_TARGET] = (
        df_processed[CLASSIFICATION_TARGET].map({'Completed': 1, 'Not Completed': 0})
    )
    print("'Completed' → 1, 'Not Completed' → 0")

    # ========================================================================
    print("\n" + "=" * 80)
    print(" ÉTAPE 5: SÉPARATION DES DONNÉES")
    print("=" * 80)

    # Séparation features/cibles
    feature_columns = [
        col for col in df_processed.columns
        if col not in ID_COLS + ALL_TARGETS
    ]

    X = df_processed[feature_columns].copy()
    y_regression = df_processed[REGRESSION_TARGETS].copy()
    y_classification = df_processed[CLASSIFICATION_TARGET].copy()

    print(f"\nDimensions:")
    print(f"Features: {X.shape}")
    print(f"Cibles régression: {y_regression.shape}")
    print(f"Cible classification: {y_classification.shape}")

    # Division stratifiée: 70% train, 15% val, 15% test
    X_train, X_temp, y_reg_train, y_reg_temp, y_cls_train, y_cls_temp = train_test_split(
        X, y_regression, y_classification,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=y_classification
    )

    X_val, X_test, y_reg_val, y_reg_test, y_cls_val, y_cls_test = train_test_split(
        X_temp, y_reg_temp, y_cls_temp,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=y_cls_temp
    )

    print(f"\nRépartition:")
    print(f"Train: {len(X_train):,} ({len(X_train) / len(X) * 100:.1f}%)")
    print(f"Val:   {len(X_val):,} ({len(X_val) / len(X) * 100:.1f}%)")
    print(f"Test:  {len(X_test):,} ({len(X_test) / len(X) * 100:.1f}%)")

    # ========================================================================
    print("\n" + "=" * 80)
    print(" ÉTAPE 6: NORMALISATION")
    print("=" * 80)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    print("\nStandardScaler appliqué")
    print(f"Moyenne (train): {X_train_scaled.mean():.6f}")
    print(f"Écart-type (train): {X_train_scaled.std():.6f}")

    # ========================================================================
    print("\n" + "=" * 80)
    print(" ÉTAPE 7: SAUVEGARDE")
    print("=" * 80)

    save_dir = 'tensorflow_preprocessed_data'
    os.makedirs(save_dir, exist_ok=True)

    # Sauvegarde des données
    np.save(f'{save_dir}/X_train.npy', X_train_scaled)
    np.save(f'{save_dir}/y_reg_train.npy', y_reg_train.values)
    np.save(f'{save_dir}/y_cls_train.npy', y_cls_train.values)
    np.save(f'{save_dir}/X_val.npy', X_val_scaled)
    np.save(f'{save_dir}/y_reg_val.npy', y_reg_val.values)
    np.save(f'{save_dir}/y_cls_val.npy', y_cls_val.values)
    np.save(f'{save_dir}/X_test.npy', X_test_scaled)
    np.save(f'{save_dir}/y_reg_test.npy', y_reg_test.values)
    np.save(f'{save_dir}/y_cls_test.npy', y_cls_test.values)

    # Sauvegarde des objets
    with open(f'{save_dir}/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    feature_names = X_train.columns.tolist()
    with open(f'{save_dir}/feature_names.pkl', 'wb') as f:
        pickle.dump(feature_names, f)

    if label_encoders:
        with open(f'{save_dir}/label_encoders.pkl', 'wb') as f:
            pickle.dump(label_encoders, f)

    # Configuration
    config = {
        'n_features': len(feature_names),
        'n_regression_targets': len(REGRESSION_TARGETS),
        'regression_targets': REGRESSION_TARGETS,
        'classification_target': CLASSIFICATION_TARGET,
        'train_size': len(X_train),
        'val_size': len(X_val),
        'test_size': len(X_test),
        'random_state': RANDOM_STATE,
        'preprocessing_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    with open(f'{save_dir}/config.pkl', 'wb') as f:
        pickle.dump(config, f)

    print(f"\ Données sauvegardées dans: {save_dir}/")
    print("Prétraitement terminé avec succès!")

    return {
        'X_train': X_train_scaled,
        'y_reg_train': y_reg_train.values,
        'y_cls_train': y_cls_train.values,
        'X_val': X_val_scaled,
        'y_reg_val': y_reg_val.values,
        'y_cls_val': y_cls_val.values,
        'X_test': X_test_scaled,
        'y_reg_test': y_reg_test.values,
        'y_cls_test': y_cls_test.values,
        'config': config,
        'feature_names': feature_names
    }


# ============================================================================
# SECTION 3: MODÈLE DE RÉGRESSION MULTI-SORTIES
# ============================================================================

def build_regression_model(input_dim, n_outputs,
                           hidden_layers=[256, 128, 64, 32],
                           dropout_rate=0.3,
                           l2_reg=0.001,
                           learning_rate=0.001):
    """
    Construction d'un modèle DNN pour la régression multi-sorties.

    Args:
        input_dim: Nombre de features d'entrée
        n_outputs: Nombre de sorties (cibles de régression)
        hidden_layers: Liste des tailles de couches cachées
        dropout_rate: Taux de dropout
        l2_reg: Coefficient de régularisation L2
        learning_rate: Taux d'apprentissage

    Returns:
        Modèle Keras compilé
    """

    inputs = keras.Input(shape=(input_dim,), name='input_features')

    # Première couche
    x = layers.Dense(
        hidden_layers[0],
        activation='relu',
        kernel_regularizer=regularizers.l2(l2_reg),
        kernel_initializer='he_normal',
        name='dense_1'
    )(inputs)
    x = layers.BatchNormalization(name='batch_norm_1')(x)
    x = layers.Dropout(dropout_rate, name='dropout_1')(x)

    # Couches cachées
    for i, units in enumerate(hidden_layers[1:], start=2):
        x = layers.Dense(
            units,
            activation='relu',
            kernel_regularizer=regularizers.l2(l2_reg),
            kernel_initializer='he_normal',
            name=f'dense_{i}'
        )(x)
        x = layers.BatchNormalization(name=f'batch_norm_{i}')(x)
        x = layers.Dropout(dropout_rate, name=f'dropout_{i}')(x)

    # Couche de sortie
    outputs = layers.Dense(n_outputs, activation='linear', name='regression_output')(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name='Regression_DNN')

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae', 'mse']
    )

    return model


def train_regression_model(data):
    """
    Entraînement du modèle de régression.

    Args:
        data: Dictionnaire contenant les données prétraitées

    Returns:
        Modèle entraîné et historique
    """

    print("\n" + "=" * 80)
    print(" RÉGRESSION MULTI-SORTIES")
    print("=" * 80)

    X_train = data['X_train']
    y_reg_train = data['y_reg_train']
    X_val = data['X_val']
    y_reg_val = data['y_reg_val']
    X_test = data['X_test']
    y_reg_test = data['y_reg_test']
    config = data['config']

    print(f"\n1. Construction du modèle...")
    model = build_regression_model(
        input_dim=X_train.shape[1],
        n_outputs=y_reg_train.shape[1]
    )
    print(f"Architecture: [256, 128, 64, 32]")
    print(f"Paramètres: {model.count_params():,}")

    # Callbacks
    model_dir = 'tensorflow_regression_models'
    os.makedirs(model_dir, exist_ok=True)

    callback_list = [
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-7,
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            filepath=f'{model_dir}/best_model.keras',
            monitor='val_loss',
            save_best_only=True,
            verbose=0
        )
    ]

    print(f"\n2. Entraînement...")
    print(f"Train: {len(X_train):,} exemples")
    print(f"Val:   {len(X_val):,} exemples")

    history = model.fit(
        X_train, y_reg_train,
        validation_data=(X_val, y_reg_val),
        epochs=200,
        batch_size=64,
        callbacks=callback_list,
        verbose=0
    )

    epochs_trained = len(history.history['loss'])
    min_val_loss = min(history.history['val_loss'])
    print(f"\nEntraînement terminé: {epochs_trained} epochs")
    print(f"Meilleur val_loss: {min_val_loss:.4f}")

    # Évaluation
    print(f"\n3. Évaluation sur le test set...")
    y_pred_test = model.predict(X_test, verbose=0)

    results = []
    for i, target in enumerate(config['regression_targets']):
        y_true = y_reg_test[:, i]
        y_pred = y_pred_test[:, i]

        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        results.append({
            'Target': target,
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2
        })

    results_df = pd.DataFrame(results)

    print("\n" + "=" * 80)
    print(" RÉSULTATS - RÉGRESSION")
    print("=" * 80)
    print(results_df.to_string(index=False))
    print(f"\n  R² moyen:  {results_df['R²'].mean():.4f}")
    print(f"  MAE moyen: {results_df['MAE'].mean():.4f}")

    # Sauvegarde
    model.save(f'{model_dir}/final_model.keras')
    results_df.to_csv(f'{model_dir}/results.csv', index=False)

    # Visualisation
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Régression - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Train MAE')
    plt.plot(history.history['val_mae'], label='Val MAE')
    plt.title('Régression - MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{model_dir}/training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nModèle sauvegardé: {model_dir}/final_model.keras")
    print(f"Graphiques sauvegardés: {model_dir}/training_curves.png")

    return model, history, results_df


# ============================================================================
# SECTION 4: MODÈLE DE CLASSIFICATION BINAIRE
# ============================================================================

def build_classification_model(input_dim,
                               hidden_layers=[512, 256, 128, 64, 32],
                               dropout_rate=0.5,
                               l2_reg=0.001,
                               learning_rate=0.001):
    """
    Construction d'un modèle DNN pour la classification binaire.

    Args:
        input_dim: Nombre de features d'entrée
        hidden_layers: Liste des tailles de couches cachées
        dropout_rate: Taux de dropout
        l2_reg: Coefficient de régularisation L2
        learning_rate: Taux d'apprentissage

    Returns:
        Modèle Keras compilé
    """

    inputs = keras.Input(shape=(input_dim,), name='input_features')

    # Première couche
    x = layers.Dense(
        hidden_layers[0],
        activation='relu',
        kernel_regularizer=regularizers.l2(l2_reg),
        kernel_initializer='he_normal',
        name='dense_1'
    )(inputs)
    x = layers.BatchNormalization(name='batch_norm_1')(x)
    x = layers.Dropout(dropout_rate, name='dropout_1')(x)

    # Couches cachées
    for i, units in enumerate(hidden_layers[1:], start=2):
        x = layers.Dense(
            units,
            activation='relu',
            kernel_regularizer=regularizers.l2(l2_reg),
            kernel_initializer='he_normal',
            name=f'dense_{i}'
        )(x)
        x = layers.BatchNormalization(name=f'batch_norm_{i}')(x)
        x = layers.Dropout(dropout_rate, name=f'dropout_{i}')(x)

    # Couche de sortie (Sigmoid pour classification binaire)
    outputs = layers.Dense(1, activation='sigmoid', name='classification_output')(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name='Classification_DNN')

    metrics = [
        'accuracy',
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
        keras.metrics.AUC(name='auc')
    ]

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=metrics
    )

    return model


def train_classification_model(data):
    """
    Entraînement du modèle de classification.

    Args:
        data: Dictionnaire contenant les données prétraitées

    Returns:
        Modèle entraîné et historique
    """

    print("\n" + "=" * 80)
    print("CLASSIFICATION BINAIRE")
    print("=" * 80)

    X_train = data['X_train']
    y_cls_train = data['y_cls_train']
    X_val = data['X_val']
    y_cls_val = data['y_cls_val']
    X_test = data['X_test']
    y_cls_test = data['y_cls_test']

    # Analyse du déséquilibre
    unique, counts = np.unique(y_cls_train, return_counts=True)
    print(f"\nDistribution des classes (Train):")
    for cls, count in zip(unique, counts):
        cls_name = "Completed" if cls == 1 else "Not Completed"
        pct = (count / len(y_cls_train)) * 100
        print(f"  {cls_name:15s}: {count:6,} ({pct:.1f}%)")

    # Calcul des poids de classe
    class_weights_array = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_cls_train),
        y=y_cls_train
    )
    class_weights = {i: weight for i, weight in enumerate(class_weights_array)}
    print(f"\nPoids de classe: {class_weights}")

    print(f"\n1. Construction du modèle...")
    model = build_classification_model(input_dim=X_train.shape[1])
    print(f"Architecture: [256, 128, 64, 32]")
    print(f"Paramètres: {model.count_params():,}")

    # Callbacks
    model_dir = 'tensorflow_classification_models'
    os.makedirs(model_dir, exist_ok=True)

    callback_list = [
        callbacks.EarlyStopping(
            monitor='val_auc',
            patience=30,
            restore_best_weights=True,
            verbose=1,
            mode='max'
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-7,
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            filepath=f'{model_dir}/best_model.keras',
            monitor='val_auc',
            save_best_only=True,
            verbose=0,
            mode='max'
        )
    ]

    print(f"\n2. Entraînement...")
    print(f"Train: {len(X_train):,} exemples")
    print(f"Val:   {len(X_val):,} exemples")

    history = model.fit(
        X_train, y_cls_train,
        validation_data=(X_val, y_cls_val),
        epochs=300,
        batch_size=32,
        class_weight=class_weights,
        callbacks=callback_list,
        verbose=1
    )

    epochs_trained = len(history.history['loss'])
    max_val_auc = max(history.history['val_auc'])
    print(f"\nEntraînement terminé: {epochs_trained} epochs")
    print(f"Meilleur val_auc: {max_val_auc:.4f}")

    # Évaluation
    print(f"\n3. Évaluation sur le test set...")
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred_classes = (y_pred_proba > 0.5).astype(int).flatten()

    accuracy = accuracy_score(y_cls_test, y_pred_classes)
    precision = precision_score(y_cls_test, y_pred_classes, zero_division=0)
    recall = recall_score(y_cls_test, y_pred_classes, zero_division=0)
    f1 = f1_score(y_cls_test, y_pred_classes, zero_division=0)
    auc_roc = roc_auc_score(y_cls_test, y_pred_proba)

    print("\n" + "=" * 80)
    print("RÉSULTATS - CLASSIFICATION")
    print("=" * 80)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"AUC-ROC:   {auc_roc:.4f}")

    # Matrice de confusion
    cm = confusion_matrix(y_cls_test, y_pred_classes)
    print(f"\nMatrice de confusion:")
    print(f"  TN: {cm[0, 0]:6,}  FP: {cm[0, 1]:6,}")
    print(f"  FN: {cm[1, 0]:6,}  TP: {cm[1, 1]:6,}")

    # Sauvegarde
    model.save(f'{model_dir}/final_model.keras')

    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc_roc': auc_roc
    }

    with open(f'{model_dir}/results.pkl', 'wb') as f:
        pickle.dump(results, f)

    # Visualisation
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(history.history['loss'], label='Train Loss')
    axes[0].plot(history.history['val_loss'], label='Val Loss')
    axes[0].set_title('Classification - Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Binary Crossentropy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # AUC
    axes[1].plot(history.history['auc'], label='Train AUC')
    axes[1].plot(history.history['val_auc'], label='Val AUC')
    axes[1].set_title('Classification - AUC-ROC')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('AUC')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{model_dir}/training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Courbe ROC
    fpr, tpr, _ = roc_curve(y_cls_test, y_pred_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {auc_roc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Courbe ROC')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{model_dir}/roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nModèle sauvegardé: {model_dir}/final_model.keras")
    print(f"Graphiques sauvegardés: {model_dir}/")

    return model, history, results


# ============================================================================
# SECTION 5: FONCTION PRINCIPALE
# ============================================================================

def main():
    """
    Fonction principale pour exécuter le pipeline complet.
    """

    print("\n" + "=" * 80)
    print(" DÉMARRAGE DU PIPELINE TENSORFLOW")
    print("=" * 80)

    # Étape 1: Prétraitement
    try:
        data = load_and_preprocess_data('../data/Course_Completion_Prediction.csv')
    except FileNotFoundError:
        print("\nERREUR: Fichier 'Course_Completion_Prediction.csv' non trouvé")
        return

    # Étape 2: Régression
    print("\n" + "=" * 80)
    print("PHASE 2: RÉGRESSION MULTI-SORTIES")
    print("=" * 80)
    model_reg, history_reg, results_reg = train_regression_model(data)

    # Étape 3: Classification
    print("\n" + "=" * 80)
    print("PHASE 3: CLASSIFICATION BINAIRE")
    print("=" * 80)
    model_cls, history_cls, results_cls = train_classification_model(data)

    # Rapport final
    print("\n" + "=" * 80)
    print("RAPPORT FINAL")
    print("=" * 80)

    print("\nRÉGRESSION MULTI-SORTIES:")
    print(f"R² moyen:  {results_reg['R²'].mean():.4f}")
    print(f"MAE moyen: {results_reg['MAE'].mean():.4f}")
    print(f"RMSE moyen: {results_reg['RMSE'].mean():.4f}")

    print("\nCLASSIFICATION BINAIRE:")
    print(f"Accuracy:  {results_cls['accuracy']:.4f}")
    print(f"Precision: {results_cls['precision']:.4f}")
    print(f"Recall:    {results_cls['recall']:.4f}")
    print(f"F1-Score:  {results_cls['f1_score']:.4f}")
    print(f"AUC-ROC:   {results_cls['auc_roc']:.4f}")

    print("\nFichiers générés:")
    print("tensorflow_preprocessed_data/")
    print("tensorflow_regression_models/")
    print("tensorflow_classification_models/")


if __name__ == "__main__":
    main()