# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 16:54:21 2025

@author: Dominik
"""
import h5py
import numpy as np

def load_h5_data(h5_filename):
    """
    Lädt die gespeicherten Puls-Analyse-Daten aus einer HDF5-Datei.

    Args:
        h5_filename (str): Pfad zur HDF5-Datei.

    Returns:
        tuple: (Pulse A Daten, Kombinierte Pulse Daten)
    """
    print(f" Lade Daten aus {h5_filename}...")

    with h5py.File(h5_filename, "r") as h5f:
        pulse_a_data = np.array(h5f["PulseA"])  # Einzelpuls A
        pulse_combined_data = np.array(h5f["PulseCombined"])  # Doppel-Pulse

    print(" Daten erfolgreich geladen!")
    return pulse_a_data, pulse_combined_data
