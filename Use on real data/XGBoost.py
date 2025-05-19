# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 16:55:48 2025

@author: Dominik
"""

from sklearn import _config
# Konfiguration anzeigen
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import xgboost as xgb
import loadH5datei
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from itertools import product
from sklearn.utils import shuffle

# **Shuffling der Daten**
def shuffle_data(X, y):
    """
    Zufälliges Mischen der Daten und Labels.
    
    Args:
        X (np.ndarray): Feature-Matrix.
        y (np.ndarray): Labels.
    
    Returns:
        np.ndarray, np.ndarray: Gemischte Feature-Matrix und Labels.
    """
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    return X[indices], y[indices]


# Informationen über die verwendete BLAS-Bibliothek
print("NumPy BLAS-Konfiguration:")
np.__config__.show()

# =============================
# 1) Datenaufbereitung
# =============================
good_data,bad_data = loadH5datei.load_h5_data("Data/With_peakFinder_RandomSclaer_Lifetime_SameDetcetor_Pure_Cupper.h5")


print("Shape von good_data:", good_data.shape)
print("Shape von bad_data:", bad_data.shape)

# Filtere Daten (z. B. für Spalte 2)
col_idx = 2
mask = good_data[:, col_idx] >= 0
good_data_clean = good_data[mask]

# Daten auf gleiche Größe beschränken
min_size = min(good_data_clean.shape[0], bad_data.shape[0])
good_data_aligned = good_data[:min_size]
bad_data_aligned = bad_data[:min_size]

# Arrays zusammenfügen und Labels erstellen
X = np.vstack([good_data_aligned, bad_data_aligned])
y = np.concatenate([np.zeros(len(good_data_aligned)), np.ones(len(bad_data_aligned))])

X, y = shuffle_data(X, y)

# Train/Test-Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


column = ["amplitude","rise_time","fall_time","pulse_width","ratio_RT_FT","area","difference_crossing"]
# =============================
# 2) Hyperparameter-Schleife mit Fortschrittsanzeige
# =============================
param_grid = {
    'n_estimators': [100, 500, 1000,2000],
    'max_depth': [3, 5, 7, 9, 11],
    'learning_rate': [0.01, 0.1, 0.2, 0.005],
    'subsample': [0.6, 0.8, 1.0, 0.4]
}

# Alle Parameterkombinationen erzeugen
param_combinations = list(product(
    param_grid['n_estimators'],
    param_grid['max_depth'],
    param_grid['learning_rate'],
    param_grid['subsample']
))

results = []

# Fortschrittsanzeige mit tqdm
with tqdm(total=len(param_combinations), desc="GridSearch Progress") as pbar:
    for n_estimators, max_depth, learning_rate, subsample in param_combinations:
        # Modell erstellen mit GPU-Beschleunigung
        model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            reg_alpha=1,  # L1-Regularisierung
            reg_lambda=1,  # L2-Regularisierung
            random_state=42
        )
        
        # Training
        model.fit(X_train, y_train)
        
        # Testgenauigkeit berechnen
        y_test_pred = model.predict(X_test)
        test_acc = accuracy_score(y_test, y_test_pred)
        
        # Ergebnisse speichern
        results.append({
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'subsample': subsample,
            'test_acc': test_acc
        })
        pbar.update(1)

# Ergebnisse in DataFrame umwandeln
results_df = pd.DataFrame(results)
best_result = results_df.loc[results_df['test_acc'].idxmax()]
print("Beste Parameter:", best_result)
print(f"Beste Test Accuracy: {best_result['test_acc']:.2f}")

# =============================
# 3) Visualisierungen
# =============================

# Confusion Matrix
best_model = xgb.XGBClassifier( 
    n_estimators=int(best_result['n_estimators']),
    max_depth=int(best_result['max_depth']),
    learning_rate=float(best_result['learning_rate']),
    subsample=float(best_result['subsample']),
    random_state=42
)
best_model.fit(X_train, y_train)
y_test_pred = best_model.predict(X_test)
best_model.save_model("Models/Scaler_and_Peakfinder_xgboost_model.json")  # Speichert als JSON

cm = confusion_matrix(y_test, y_test_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["GUT", "SCHLECHT"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()

# Feature-Wichtigkeiten
feature_importances = best_model.feature_importances_
plt.figure(figsize=(8, 6))
plt.bar(range(len(feature_importances)), feature_importances, tick_label=column)
plt.xlabel("Features")
plt.ylabel("Wichtigkeit")
plt.title("Feature Importances")
plt.show()

# ROC-Kurve
y_test_prob = best_model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_test_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC-Kurve")
plt.legend()
plt.show()

