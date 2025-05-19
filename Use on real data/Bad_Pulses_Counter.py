# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 14:00:09 2025

@author: Dominik
"""


import h5py
import numpy as np

# Datei mit den Vorhersagen laden
output_h5 = "StreamAnalysis/Classification_PureCupper_AnalysisWithNanAdder.h5"

with h5py.File(output_h5, "r") as h5f:
    predictions = h5f["predictions"][:]  # Lade alle Vorhersagen als NumPy-Array

# Anzahl der Einsen (schlechte Pulse) und Nullen (gute Pulse) zählen
num_ones = np.sum(predictions == 1)
num_zeros = np.sum(predictions == 0)
num_twos = np.sum(predictions == 2)

Frequency = num_ones / (num_ones+num_zeros) *100
# Ergebnis ausgeben
print(f" Anzahl der schlechten Pulse (1): {num_ones}")
print(f" Anzahl der guten Pulse (0): {num_zeros}")
print(f" Anzahl der NaNs (2): {num_twos}")
print(f" Gesamtzahl der Pulse: {len(predictions)}")
print(f" Anteil an schlechtenn Pulsen im Stream: {Frequency}")
