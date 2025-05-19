
"""
Created on Mon Jan 27 10:44:15 2025

@author: Dr. Dominik Boras
"""
# -*- coding: utf-8 -*-
"""
Pulsverarbeitung ohne max_pairs-Limit & automatische NaN-Werte für hohe Amplituden
"""

from GenerateDataStructureDRS4 import *
import h5py
import numpy as np


def process_pulses(file_path, positive_polarity=False, h5_filename="pulse_analysis.h5", amplitude_threshold=500.0):
    """
    Liest Pulspaare einzeln aus der Datei und speichert die Ergebnisse direkt, ohne alles in den RAM zu laden.
    Falls die Amplitude eines Pulses den Schwellenwert überschreitet, werden automatisch NaN-Werte gespeichert.

    Args:
        file_path (str): Pfad zur Binärdatei.
        positive_polarity (bool): Gibt an, ob die Pulse positive oder negative Polarität haben.
        h5_filename (str): Name der Ausgabedatei für die HDF5-Speicherung.
        amplitude_threshold (float): Wenn die Amplitude diesen Wert überschreitet, wird der Puls als `NaN` gespeichert.
    """
    print(" Starte speicherschonende Verarbeitung der Pulsdaten...")

    # Datei einlesen, aber nicht alles in den RAM laden!
    processor = PulseStreamProcessor(file_path)
    processor.read_header()  # Header einlesen

    num_cells = processor.header["number_of_cells"]  # Zellanzahl für Pulse
    analyzer = PulseAnalyzer(positive_polarity=positive_polarity)

    # Erstelle HDF5-Datei mit dynamisch wachsenden Platzhaltern
    with h5py.File(h5_filename, "w") as h5f:
        dset_A = h5f.create_dataset("PulseA", shape=(0, 7), maxshape=(None, 7), dtype="f8")

        print(f" HDF5-Datei {h5_filename} erstellt.")

        #  Lese und verarbeite Pulse, bis das Ende der Datei erreicht ist
        with open(file_path, "rb") as file:
            file.seek(32)  # Header überspringen
            pulse_index = 0  # Zähler für gespeicherte Pulse

            while True:
                print(f" Verarbeite Pulspaar {pulse_index + 1}...")

                # Lese Puls A
                time_A, voltage_A = processor.read_pulse(file, num_cells)
                if len(time_A) == 0 or len(voltage_A) == 0:
                    print(" Datei zu Ende oder beschädigt – Abbruch!")
                    break

                _ = processor.read_pulse(file, num_cells)  # Lese zweiten Puls, aber ignoriere ihn

                # **Amplitude prüfen & ggf. NaN-Werte schreiben**
                amplitude = np.max(voltage_A) if positive_polarity else np.min(voltage_A)

                if amplitude > amplitude_threshold:
                    print(f" Amplitude {amplitude:.2f} mV überschreitet Threshold ({amplitude_threshold} mV) → NaN gespeichert!")
                    results_A = {key: np.nan for key in analyzer.analyze_pulse(time_A, voltage_A, remove_spikes=True).keys()}
                else:
                    results_A = analyzer.analyze_pulse(time_A, voltage_A, remove_spikes=True)

                # **Dynamisch Dataset erweitern und speichern**
                dset_A.resize((pulse_index + 1, 7))
                dset_A[pulse_index] = list(results_A.values())

                pulse_index += 1

                # **Speicher freigeben**
                del time_A, voltage_A, results_A

    print(f" Verarbeitung abgeschlossen! {pulse_index} Pulse gespeichert in {h5_filename}")


if __name__ == "__main__":
    # **Hauptprogramm**
    FILE_PATH = "/PureKupfer_ohne_Bleiabschirmung.drs4DataStream"

    POSITIVE_POLARITY = False  # Polarität der Pulse
    H5_FILENAME = "StreamAnalysis/PureCupper_AnalysisWithNanAdder.h5"  # Name der HDF5-Datei
    AMPLITUDE_THRESHOLD = -99.0  # Schwellenwert für Amplituden (in mV)

    process_pulses(FILE_PATH, POSITIVE_POLARITY, H5_FILENAME, AMPLITUDE_THRESHOLD)
