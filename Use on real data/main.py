# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 10:44:15 2025

@author: Dr. Dominik Boras
"""

from GenerateDataStructureDRS4 import *
import h5py
import numpy as np


def process_pulses(file_path, file_path_2, max_pairs=100, positive_polarity=False, h5_filename="pulse_analysis.h5"):
    """
    Liest Pulspaare einzeln aus der Datei, erzeugt Double-Detection Pulse und speichert die Ergebnisse direkt.
    Vermeidet hohen Speicherverbrauch.

    Args:
        file_path (str): Pfad zur Binärdatei.
        max_pairs (int): Maximale Anzahl an Pulspaaren, die geladen werden.
        positive_polarity (bool): Gibt an, ob die Pulse positive oder negative Polarität haben.
        h5_filename (str): Name der Ausgabedatei für die HDF5-Speicherung.
    """
    print(" Starte speicherschonende Verarbeitung der Pulsdaten...")

    # Datei einlesen, aber nicht alles in den RAM laden!
    processor = PulseStreamProcessor(file_path)
    processor.read_header()  # Header einlesen

    num_cells = processor.header["number_of_cells"]  # Zellanzahl für Pulse
    analyzer = PulseAnalyzer(positive_polarity=positive_polarity)
    double_detector = create_double_detection(numberOfdoubleDetection=1,)  # 1 Double-Detection pro Schritt

    # Erstelle HDF5-Datei mit Platzhaltern für effizientes Schreiben
    with h5py.File(h5_filename, "w") as h5f:
        dset_A = h5f.create_dataset("PulseA", shape=(max_pairs, 7), dtype="f8")
        dset_combined = h5f.create_dataset("PulseCombined", shape=(max_pairs, 7), dtype="f8")

        print(f" HDF5-Datei {h5_filename} erstellt.")

        # 2️⃣ Lese und verarbeite Pulse paarweise
        with open(file_path, "rb") as file, open(file_path_2, "rb") as file2:
            file.seek(32)  # Header überspringen
            file2.seek(32)  # Header überspringen
            for i in range(max_pairs):
                print(f" Verarbeite Pulspaar {i+1}/{max_pairs}...")

                # Lese Puls A
                time_A, voltage_A = processor.read_pulse(file, num_cells)
                if len(time_A) == 0 or len(voltage_A) == 0:
                    print(" Datei zu Ende oder beschädigt – Abbruch!")
                    break
                
                _ = processor.read_pulse(file, num_cells)
                if len(time_A) == 0 or len(voltage_A) == 0:
                    print(" Datei zu Ende oder beschädigt – Abbruch!")
                    break

                # Lese Puls B
                time_B, voltage_B = processor.read_pulse(file2, num_cells)
                if len(time_B) == 0 or len(voltage_B) == 0:
                    print(" Datei zu Ende oder beschädigt – Abbruch!")
                    break
                
                # Lese Puls B
                _ = processor.read_pulse(file2, num_cells)
                if len(time_B) == 0 or len(voltage_B) == 0:
                    print(" Datei zu Ende oder beschädigt – Abbruch!")
                    break
                processor.plot_pulse(time_A,voltage_A)
                processor.plot_pulse(time_B,voltage_B)
                
                # Pulse A analysieren
                results_A = analyzer.analyze_pulse(time_A, voltage_A, remove_spikes=True)
                dset_A[i] = list(results_A.values())  # Direkt in die Datei schreiben

                # Double-Detection Pulse mit Peak-Alignment erzeugen
                time_combined, voltage_combined = double_detector.combine_pulses( 
                    np.array([time_A]), np.array([voltage_A]),
                    np.array([time_B]), np.array([voltage_B])
                )

                # Kombinierten Puls analysieren
                results_combined = analyzer.analyze_pulse(time_combined[0], voltage_combined[0], remove_spikes=True)
                dset_combined[i] = list(results_combined.values())  # Direkt in die Datei schreiben
                
                
                # **Speicher freigeben**
                del time_A, voltage_A, time_B, voltage_B, time_combined, voltage_combined, results_A, results_combined
                
                
                print(f" Pulspaar {i+1} gespeichert.")

    print(f" Verarbeitung abgeschlossen! Daten gespeichert in {h5_filename}")


if __name__ == "__main__":
    # **Hauptprogramm**
    FILE_PATH = "/Start_Pure_Cu_90Grad_06032025__truePulses.drs4DataStream"
    FILE_PATH2 = "/Stop_Pure_Cu_90Grad_07032025__falsePulses__truePulses.drs4DataStream"
    
    MAX_PAIRS = 250000  # Maximale Anzahl an Pulspaaren
    POSITIVE_POLARITY = False  # Polarität der Pulse
    H5_FILENAME = "Data/Test.h5"  # Name der HDF5-Datei

    process_pulses(FILE_PATH,FILE_PATH2, MAX_PAIRS, POSITIVE_POLARITY, H5_FILENAME)
