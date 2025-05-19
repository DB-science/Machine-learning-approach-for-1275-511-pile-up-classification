# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 16:41:14 2025

@author: Dominik
"""

import h5py
import numpy as np
import struct
from tqdm import tqdm
from GenerateDataStructureDRS4 import *

# Eingangsdateien
input_bin_file = "/PureKupfer_ohne_Bleiabschirmung.drs4DataStream"
classification_h5 = "s/Classification_PureCupper_AnalysisWithNanAdder.h5"  # Enthält 0 (gut) oder 1 (schlecht) oder 2 (NaN werte inhalten)
output_bin_file = "/KI_FilteredStream_PureKupfer_ohne_Bleiabschirmung.drs4DataStream"

# Initialisiere den Stream-Processor
processor = PulseStreamProcessor(input_bin_file)
processor.read_header()  # Header einlesen

# Lade Klassifikationen
with h5py.File(classification_h5, "r") as h5f:
    classifications = h5f["predictions"][:]  # Array mit 0 und 1 (gleiche Reihenfolge wie Pulse)

num_pulses = len(classifications)
print(f"🔍 Lade {num_pulses} Klassifikationen aus {classification_h5}")

# Neue Binärdatei für gefilterte Pulse erstellen
with open(input_bin_file, "rb") as infile, open(output_bin_file, "wb") as outfile:
    infile.seek(0)  # Lese originalen Header
    header = infile.read(32)  # Die ersten 32 Bytes sind der Header
    outfile.write(header)  # Header in die neue Datei schreiben

    # Pulse sequenziell lesen & schreiben
    with tqdm(total=num_pulses, desc="Filtering Pulses", unit="pulse") as pbar:
        for i in range(num_pulses):
            # **Ersten Puls (A) lesen**
            time_a, voltage_a = processor.read_pulse(infile, processor.header["number_of_cells"])
            if len(time_a) == 0 or len(voltage_a) == 0:
                break  # Datei zu Ende

            # **Zweiten Puls (B) lesen**
            time_b, voltage_b = processor.read_pulse(infile, processor.header["number_of_cells"])
            if len(time_b) == 0 or len(voltage_b) == 0:
                break  # Datei zu Ende

            # **Falls Klassifikation = 1 → Puls verwerfen**
            if classifications[i] in {1,2}:
                pbar.update(1)
                continue
            
            
            try:
                # **Falls Klassifikation = 0 → Puls speichern**
                # Konvertiere Daten zurück in Binärformat und schreibe sie in die neue Datei
                for t in time_a:
                    outfile.write(struct.pack('f', t))
                for v in voltage_a:
                    outfile.write(struct.pack('f', v))
    
                for t in time_b:
                    outfile.write(struct.pack('f', t))
                for v in voltage_b:
                    outfile.write(struct.pack('f', v))
                
                outfile.flush() #Sicherheitsmaßnahme für schnelles Speichern
                
            except Exception as e:
                print(f" Fehler beim Speichern von Puls {i}: {e}")
            pbar.update(1)

print(f" Fertig! Die neue Datei mit guten Pulsen wurde gespeichert: {output_bin_file}")
