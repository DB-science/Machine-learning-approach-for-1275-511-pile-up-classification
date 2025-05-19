# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 14:32:31 2025

@author: Dominik
"""
""

import h5py
import numpy as np

# Datei mit den Features laden
input_h5 = "NewAnalysisWithNanAdder.h5"

with h5py.File(input_h5, "r") as h5f:
    features_dataset = h5f["PulseA"][:]  # Lade das gesamte Dataset als NumPy-Array

# Prüfen, welche Zeilen eine 0 oder NaN enthalten
contains_nan = np.isnan(features_dataset).any(axis=1)  # Zeilen mit mindestens einem NaN
contains_zero = (features_dataset == 0).any(axis=1)  # Zeilen mit mindestens einer 0

# Zeilen zählen
num_nan_rows = np.sum(contains_nan)
num_zero_rows = np.sum(contains_zero)
num_both = np.sum(contains_nan | contains_zero)  # Zeilen mit 0 oder NaN

amplitude_column_index = 0

valid_amplitudes_mask = features_dataset[:, amplitude_column_index]>-100
num_valid_amplitudes = np.sum(valid_amplitudes_mask)


print(f" Anzahl der Rows mit Amplitude >-100mV: {num_valid_amplitudes}")
# Ergebnisse ausgeben
print(f" Anzahl der Rows mit mindestens einer NaN: {num_nan_rows}")
print(f" Anzahl der Rows mit mindestens einer 0: {num_zero_rows}")
print(f" Gesamtzahl der Rows mit 0 oder NaN: {num_both}")
print(f" Gesamtanzahl der Zeilen im Dataset: {features_dataset.shape[0]}")
