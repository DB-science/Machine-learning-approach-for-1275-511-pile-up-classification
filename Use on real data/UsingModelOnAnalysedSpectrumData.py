# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 16:05:36 2025

@author: Dr. Dominik Boras
"""
import h5py
import xgboost as xgb
import numpy as np
from tqdm import tqdm

# 1. Modell laden
model = xgb.XGBClassifier()
model.load_model("Models/Scaler_and_Peakfinder_xgboost_model.json")

# 2. HDF5-Datei mit Features öffnen (Batch-Verarbeitung)
input_h5 = "StreamAnalysis/PureCupper_AnalysisWithNanAdder.h5"
output_h5 = "StreamAnalysis/Classification_PureCupper_AnalysisWithNanAdder.h5"

BATCH_SIZE = 10_000  # Anzahl Zeilen pro Batch

with h5py.File(input_h5, "r") as h5f:
    features_dataset = h5f["PulseA"]
    column = ["amplitude","rise_time","fall_time","pulse_width","ratio_RT_FT","area","difference_crossing"]
    #column_names = [col.decode() for col in h5f["columns"][:]]

    num_samples, num_features = features_dataset.shape
    print(f" Lade {num_samples} Zeilen mit {num_features} Features.")

    # 3. HDF5-Datei für die Vorhersagen anlegen
    with h5py.File(output_h5, "w") as out_h5:
        predictions_dset = out_h5.create_dataset("predictions", (num_samples,), dtype="i8")

        # 4. Verarbeitung in Batches mit NaN-Prüfung
        with tqdm(total=num_samples, desc="Predicting Pulses", unit="pulse") as pbar:
            for start in range(0, num_samples, BATCH_SIZE):
                end = min(start + BATCH_SIZE, num_samples)  # Ende des Batches sicherstellen
                batch_data = features_dataset[start:end, :]  # Lade Batch

                # ** Schritt 1: Prüfe auf NaN-Werte in den Features**
                nan_mask = np.isnan(batch_data).any(axis=1)  # True = hat NaNs
                predictions = np.full(batch_data.shape[0],fill_value =1, dtype="i8")  # Standard: Alle als schlecht (1) Erzeugt ein nparry aus einsen und nur die rows mit keinem nan werden predicted und haben die möglichkeit auf null zu gehen
                predictions[nan_mask] = 2 #Zeilen mit NaN bekommen die Nummer 2
                
                # ** Schritt 2: Nur Zeilen ohne NaN durch das Modell laufen lassen**
                valid_indices = ~nan_mask  # True = Zeilen ohne NaN
                if np.any(valid_indices):  # Falls es gültige Zeilen gibt
                    valid_data = batch_data[valid_indices]  # Lade nur gültige Zeilen
                    predictions[valid_indices] = model.predict(valid_data)  # Model-Predictions

                # ** Schritt 3: Speichere die Predictions**
                predictions_dset[start:end] = predictions
                pbar.update(end - start)

print(f" Fertig! Vorhersagen gespeichert in {output_h5}")
