# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 15:54:05 2025

@author: Dr. Dominik Boras
"""


import numpy as np
import struct
import matplotlib.pyplot as plt
from scipy.interpolate import Akima1DInterpolator
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from typing import List, Optional

class PulseStreamProcessor:
    
    def __init__(self, file_path):
        """
        Initialisiert den Prozessor mit dem Pfad zur Binärdatei und der Puls-Polarität.
        
        Args:
            file_path (str): Pfad zur Binärdatei.
            positive_polarity (bool): True für positive Puls-Polarität, False für negative Polarität.
        """
        self.file_path = file_path
        self.header = {}
        self.a_pulses = None
        self.b_pulses = None

    def read_header(self):
        """
        Liest den Header der Datei und speichert die Header-Daten als Klassenattribut.
        """
        with open(self.file_path, "rb") as file:
            __ = struct.unpack('i', file.read(4))[0]  # unused
            __ = struct.unpack('i', file.read(4))[0]  # unused
            
            sweep_in_nanoseconds = struct.unpack('d', file.read(8))[0]
            frequency_in_ghz = struct.unpack('d', file.read(8))[0]
            number_of_cells = struct.unpack('i', file.read(4))[0]
            __ = struct.unpack('i', file.read(4))[0]  # padding (32 bytes total)
        
        self.header = {
            "number_of_cells": number_of_cells,
            "sweep_in_ns": sweep_in_nanoseconds,
            "frequency_in_ghz": frequency_in_ghz,
        }

    def read_pulse(self, file, number_of_cells):
        """
        Liest die Zeit- und Spannungsdaten eines einzelnen Pulses aus der Datei.
        """
        time = np.zeros(number_of_cells)
        volt = np.zeros(number_of_cells)

        for i in range(number_of_cells):
            byte_chunk = file.read(4)
            if not byte_chunk:
                return np.zeros(0), np.zeros(0)
            time[i] = struct.unpack('f', byte_chunk)[0]

        for i in range(number_of_cells):
            byte_chunk = file.read(4)
            if not byte_chunk:
                return np.zeros(0), np.zeros(0)
            volt[i] = struct.unpack('f', byte_chunk)[0]

        return time, volt

    def read_stream(self, max_pairs=None):
        """
        Liest den Puls-Stream und speichert die A- und B-Pulse als Klassenattribute.
        """
        if not self.header:
            self.read_header()

        number_of_cells = self.header["number_of_cells"]
        a_pulses_time = []
        a_pulses_voltage = []
        b_pulses_time = []
        b_pulses_voltage = []

        with open(self.file_path, "rb") as file:
            file.seek(32)  # Überspringe den Header
            pair_count = 0

            while True:
                if max_pairs is not None and pair_count >= max_pairs:
                    break

                time_a, volt_a = self.read_pulse(file, number_of_cells)
                if len(time_a) == 0 or len(volt_a) == 0:
                    break
                a_pulses_time.append(time_a)
                a_pulses_voltage.append(volt_a)

                time_b, volt_b = self.read_pulse(file, number_of_cells)
                if len(time_b) == 0 or len(volt_b) == 0:
                    break
                b_pulses_time.append(time_b)
                b_pulses_voltage.append(volt_b)

                pair_count += 1

        self.a_pulses = np.array(list(zip(a_pulses_time, a_pulses_voltage)), dtype=[('Time', 'f4', number_of_cells), ('Voltage', 'f4', number_of_cells)])
        self.b_pulses = np.array(list(zip(b_pulses_time, b_pulses_voltage)), dtype=[('Time', 'f4', number_of_cells), ('Voltage', 'f4', number_of_cells)])

    def plot_pulse(self, time, voltage, title="Pulsdarstellung"):
        """
        Stellt einen einzelnen Puls dar.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(time, voltage, label="Puls", linewidth=1.5)
        plt.title(title, fontsize=16)
        plt.xlabel("Zeit [ns]", fontsize=14)
        plt.ylabel("Spannung [mV]", fontsize=14)
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend(fontsize=12) 
        plt.tight_layout()
        plt.show()
        plt.tight_layout()
    
    
    def akima_interpolate(self, time, voltage):
        """
        Führt eine Akima-Interpolation durch und gibt interpolierte Werte zurück.
        
        Die Anzahl der Interpolationspunkte wird basierend auf der Anzahl der Zellen (number_of_cells) berechnet.
        
        Args:
            time (np.ndarray): Zeitwerte des Pulses.
            voltage (np.ndarray): Spannungswerte des Pulses.
        
        Returns:
            interp_time (np.ndarray): Interpolierte Zeitpunkte.
            interp_voltage (np.ndarray): Interpolierte Spannungswerte.
        """
        if not self.header or "number_of_cells" not in self.header:
            raise ValueError("Header muss gelesen werden, um die Anzahl der Zellen zu bestimmen.")
        
        # Anzahl der Interpolationspunkte: 10x number_of_cells
        num_points = self.header["number_of_cells"] * 20
    
        # Akima-Interpolation durchführen
        akima = Akima1DInterpolator(time, voltage)
        interp_time = np.linspace(time.min(), time.max(), num_points)
        interp_voltage = akima(interp_time)
        return interp_time, interp_voltage

class PulseAnalyzer:
   
    def __init__(self, positive_polarity=True):
        """
        Initialisiert den Analyzer mit der Polarität der Pulse.

        Args:
            positive_polarity (bool): True für positive Polarität, False für negative Polarität.
        """
        self.positive_polarity = positive_polarity
        self.i=0

    def calculate_amplitude(self, voltage):
        """
        Berechnet die Amplitude eines Pulses.
        """
        if self.positive_polarity:
            return np.max(voltage)
        else:
            return np.min(voltage)

    
    def calculate_rise_time(self, time, voltage, amplitude, trim_percentage=0.05, debug_plot=False):
        """
        Berechnet die Rise Time (Anstiegszeit) eines Pulses und visualisiert Pulse mit NaN-Werten.
    
        Args:
            time (np.ndarray): Zeitwerte des Pulses.
            voltage (np.ndarray): Spannungswerte des Pulses.
            amplitude (float): Amplitude des Pulses.
            trim_percentage (float): Anteil, der vorne und hinten abgeschnitten wird.
            debug_plot (bool): Falls True, werden problematische Pulse geplottet.
    
        Returns:
            float: Rise Time (Zeit von 10% bis 90% der Amplitude) oder np.nan, falls ungültig.
        """
        # Trimmen des Signals
        num_points = len(voltage)
        trim_count = int(num_points * trim_percentage)
    
        if trim_count * 2 >= num_points:
            return np.nan  # Zu wenig Daten nach Trimmen, Rückgabe von NaN
    
        time_trimmed = time[trim_count:-trim_count]
        voltage_trimmed = voltage[trim_count:-trim_count]
    
        # Indizes für Anstieg finden
        if self.positive_polarity:
            rise_start_indices = np.where(voltage_trimmed >= 0.1 * amplitude)[0]
            rise_end_indices = np.where(voltage_trimmed >= 0.9 * amplitude)[0]
        else:
            rise_start_indices = np.where(voltage_trimmed <= 0.1 * amplitude)[0]
            rise_end_indices = np.where(voltage_trimmed <= 0.9 * amplitude)[0]
            
    
        # Fehlerbehandlung: Falls keine gültigen Indizes gefunden wurden
        if rise_start_indices.size == 0 or rise_end_indices.size == 0:
            if rise_start_indices.size == 0:
                print("rise_start_problem")
            if rise_end_indices.size == 0:
                print("rise_end_problem")
            if debug_plot:
                self.plot_problematic_pulse(time_trimmed, voltage_trimmed, "NaN Rise Time Detected")
            return np.nan  
    
        # Sicherstellen, dass die Indizes in richtiger Reihenfolge kommen
        rise_start = rise_start_indices[0]
        rise_end = rise_end_indices[0]
    
        if rise_end <= rise_start:
            if debug_plot:
                self.plot_problematic_pulse(time_trimmed, voltage_trimmed, "Invalid Rise Time Sequence")
                print("rise_start is equal or bigger then rise_end")
            return np.nan  
    
        return time_trimmed[rise_end] - time_trimmed[rise_start]

    def calculate_fall_time(self, time, voltage, amplitude, trim_percentage=0.05, debug_plot=False):
        """
        Berechnet die Fall Time (Abfallzeit) eines Pulses.
    
        Args:
            time (np.ndarray): Zeitwerte des Pulses.
            voltage (np.ndarray): Spannungswerte des Pulses.
            amplitude (float): Amplitude des Pulses.
            trim_percentage (float): Anteil, der vorne und hinten abgeschnitten wird.
    
        Returns:
            float: Fall Time (Zeit von 90% bis 10% der Amplitude) oder np.nan, falls ungültig.
        """
        # Sicherheitsabfrage für die Amplitude
        if np.isnan(amplitude):
            print("⚠️ Amplitude ist 0 oder NaN! → Kann keine Fall Time berechnen.")
            return np.nan  
    
        # Trimmen des Signals
        num_points = len(voltage)
        trim_count = int(num_points * trim_percentage)
    
        if trim_count * 2 >= num_points:
            print(" Zu wenig Daten nach Trimmen!")
            return np.nan  
    
        time_trimmed = time[trim_count:-trim_count]
        voltage_trimmed = voltage[trim_count:-trim_count]
    
        # Indizes für Abfall finden (von rechts nach links!)
        if self.positive_polarity:
            fall_start_indices = np.where(voltage_trimmed >= 0.9 * amplitude)[0]
            fall_end_indices = np.where(voltage_trimmed >= 0.1 * amplitude)[0]
        else:
            fall_start_indices = np.where(voltage_trimmed <= 0.9 * amplitude)[0]
            fall_end_indices = np.where(voltage_trimmed <= 0.1 * amplitude)[0]
    
        # Fehlerbehandlung: Falls keine gültigen Indizes gefunden wurden
        if fall_start_indices.size == 0 or fall_end_indices.size == 0:
            if fall_start_indices.size == 0:
                print(" fall_start_problem: Kein Startpunkt für den Abfall gefunden.")
            if fall_end_indices.size == 0:
                print(" fall_end_problem: Kein Endpunkt für den Abfall gefunden.")
            if debug_plot:
                self.plot_problematic_pulse(time_trimmed, voltage_trimmed, "NaN Fall Time Detected")
            return np.nan  
    
        # Richtige Reihenfolge: von rechts nach links suchen!
        fall_start = fall_start_indices[-1]  # Letztes Vorkommen von 90% Amplitude
        fall_end = fall_end_indices[-1]  # Erstes Vorkommen von 10% Amplitude
    
        # Sicherstellen, dass die Indizes gültig sind
        if fall_end <= fall_start:
            print(" fall_start ist größer oder gleich fall_end! → Ungültige Reihenfolge.")
            if debug_plot:
                self.plot_problematic_pulse(time_trimmed, voltage_trimmed, "Invalid Fall Time Sequence")
            return np.nan  
    
        return time_trimmed[fall_end] - time_trimmed[fall_start]


    def calculate_pulse_width(self, time, voltage, amplitude):
        """
        Berechnet die Pulsbreite auf halber Höhe.
        """
        if self.positive_polarity:
            rise_start_indices = np.where(voltage >= 0.1 * amplitude)[0]
            fall_end_indices = np.where(voltage <= 0.1 * amplitude)[0]
        else:
            rise_start_indices = np.where(voltage <= 0.1 * amplitude)[0]
            fall_end_indices = np.where(voltage >= 0.1 * amplitude)[0]
    
        # Fehlerbehandlung
        if rise_start_indices.size == 0 or fall_end_indices.size == 0:
            return np.nan  
    
        return time[fall_end_indices[-1]] - time[rise_start_indices[0]]
    
    @staticmethod
    def smooth_signal_savgol(voltage, window_length=11, polyorder=2):
        """
        Glättet das Signal mit einem Savitzky-Golay-Filter.
    
        Args:
            voltage (np.ndarray): Die Spannungswerte des Pulses.
            window_length (int): Länge des Glättungsfensters (ungerade Zahl).
            polyorder (int): Grad des Polynoms für die Anpassung.
    
        Returns:
            np.ndarray: Geglättetes Signal.
        """
        return savgol_filter(voltage, window_length=window_length, polyorder=polyorder)
    
    @staticmethod
    def smooth_signal(voltage, window_size=5):
        """
        Glättet das Signal mit einem Moving-Average-Filter.
    
        Args:
            voltage (np.ndarray): Die Spannungswerte des Pulses.
            window_size (int): Fenstergröße für die Mittelung.
    
        Returns:
            np.ndarray: Geglättetes Signal.
        """
        return np.convolve(voltage, np.ones(window_size) / window_size, mode='same')


    def calculate_derivative(self, time, voltage, amplitude, smooth=True, method='savgol', window_size=11):
        """
        Berechnet die numerische Ableitung des Pulses und findet Null-Durchgänge.
    
        Args:
            time (np.ndarray): Zeitwerte des Pulses.
            voltage (np.ndarray): Spannungswerte des Pulses.
            amplitude (float): Amplitude des Pulses.
            smooth (bool): Ob das Signal geglättet werden soll.
            method (str): Glättungsmethode ('moving_average' oder 'savgol').
            window_size (int): Fenstergröße für die Glättung.
    
        Returns:
            float: Differenz zwischen den ersten beiden Null-Durchgängen oder np.nan, falls nicht vorhanden.
        """
        if smooth:
            if method == 'moving_average':
                voltage = self.smooth_signal(voltage, window_size=window_size)
            elif method == 'savgol':
                voltage = self.smooth_signal_savgol(voltage, window_length=window_size, polyorder=2)
    
        # Numerische Ableitung berechnen
        derivative = np.gradient(voltage, time)
    
        # Referenzspannung basierend auf dem Durchschnitt der ersten 100 Datenpunkte
        baseline = np.mean(voltage[10:100])
    
        # Schwellenwerte basierend auf der Polarität
        if self.positive_polarity:
            lower_threshold = baseline + 0.1 * amplitude         
        else:
            lower_threshold = baseline - 0.1 * amplitude       
    
        # Relevanter Bereich: nur Werte zwischen den Schwellenwerten (Flanken)
        relevant_indices = np.where((voltage > lower_threshold) )[0] \
            if self.positive_polarity else \
            np.where((voltage < lower_threshold) )[0]
    
        #Sicherstellen, dass relevante Werte vorhanden sind
        if len(relevant_indices) < 2:
            return np.nan  # Zu wenige Datenpunkte zum Berechnen
    
        # Null-Durchgänge in der Ableitung finden (nur in relevanten Bereichen)
        sign_changes = np.where(np.diff(np.sign(derivative[relevant_indices])))[0]
    
        # Sicherstellen, dass mindestens zwei Null-Durchgänge existieren
        if len(sign_changes) < 2:
            return np.nan  
    
        # Interpolierte Zeitpunkte der Null-Durchgänge
        zero_crossings = []
        for idx in sign_changes:
            t1, t2 = time[relevant_indices[idx]], time[relevant_indices[idx + 1]]
            v1, v2 = derivative[relevant_indices[idx]], derivative[relevant_indices[idx + 1]]
            zero_crossing = t1 - (v1 * (t2 - t1)) / (v2 - v1)
            zero_crossings.append(zero_crossing)
    
        # Sicherstellen, dass mindestens zwei Null-Durchgänge existieren
        if len(zero_crossings) < 2:
            return np.nan  
    
        rise_value = zero_crossings[0]
        fall_value = zero_crossings[1]
        ratio_crossing = fall_value - rise_value
    
        return ratio_crossing
    
    def calculate_area(self, time, voltage, amplitude):
        """
        Berechnet die Fläche unter der Puls-Kurve zwischen 10% Anstieg und 10% Abfall der Amplitude.
    
        Args:
            time (np.ndarray): Zeitwerte des Pulses.
            voltage (np.ndarray): Spannungswerte des Pulses.
            amplitude (float): Amplitude des Pulses.
    
        Returns:
            float: Fläche unter der Kurve (integral der Spannung) oder np.nan, falls Werte fehlen.
        """
        if self.positive_polarity:
            rise_start_indices = np.where(voltage >= 0.1 * amplitude)[0]
            fall_end_indices = np.where(voltage <= 0.1 * amplitude)[0]
        else:
            rise_start_indices = np.where(voltage <= 0.1 * amplitude)[0]
            fall_end_indices = np.where(voltage >= 0.1 * amplitude)[0]
    
        # Sicherstellen, dass Indizes existieren
        if rise_start_indices.size == 0 or fall_end_indices.size == 0:
            return np.nan  # Kein gültiger Bereich gefunden
    
        rise_start = rise_start_indices[0]
        fall_end = fall_end_indices[-1]
    
        # Sicherstellen, dass `fall_end` nach `rise_start` kommt
        if fall_end <= rise_start:
            return np.nan  # Ungültiger Bereich
    
        # Fläche unter der Kurve berechnen (numerische Integration)
        area = np.trapz(voltage[rise_start:fall_end], time[rise_start:fall_end])
    
        return area
    
    def trim_edges(self, voltage, time, trim_percentage=0.05):
        """
        Entfernt Spikes an den Rändern eines Pulses.
        """
        num_points = len(voltage)
        trim_count = int(num_points * trim_percentage)
    
        # Falls das Trimmen zu einem leeren Array führen würde, abbrechen
        if trim_count * 2 >= num_points:
            return time, voltage  
    
        return time[trim_count:-trim_count], voltage[trim_count:-trim_count] 
    
    def analyze_pulse(self, time, voltage, remove_spikes = True):
        """
        Analysiert den Puls und berechnet verschiedene Parameter.

        Args:
            time (np.ndarray): Zeitwerte des Pulses.
            voltage (np.ndarray): Spannungswerte des Pulses.

        Returns:
            dict: Enthält die berechneten Parameter.
        """
        if remove_spikes:
            time, voltage = self.trim_edges(voltage, time)
    
        amplitude = self.calculate_amplitude(voltage)
        rise_time = self.calculate_rise_time(time, voltage, amplitude)
        fall_time = self.calculate_fall_time(time, voltage, amplitude)
        pulse_width = self.calculate_pulse_width(time, voltage, amplitude)
        ratio_RT_FT = rise_time / fall_time if fall_time != 0 and not np.isnan(fall_time) else np.nan
        area = self.calculate_area(time, voltage, amplitude)
        differenceCrossing = self.calculate_derivative(time, voltage, amplitude)
    
        return {
            "amplitude": amplitude,
            "rise_time": rise_time,
            "fall_time": fall_time,
            "pulse_width": pulse_width,
            "ratio_RT_FT": ratio_RT_FT,
            "area": area,
            "difference_crossing": differenceCrossing,
        }
    
    def plot_problematic_pulse(self, time, voltage, title="Problematischer Puls"):
        """
        Plottet Pulse, bei denen eine fehlerhafte Berechnung erkannt wurde.
    
        Args:
            time (np.ndarray): Zeitwerte des Pulses.
            voltage (np.ndarray): Spannungswerte des Pulses.
            title (str): Titel des Plots.
        """
        plt.figure(figsize=(10, 5))
        plt.plot(time, voltage, label="Puls", linewidth=1.5)
        plt.title(title, fontsize=14, color="red")
        plt.xlabel("Zeit [ns]", fontsize=12)
        plt.ylabel("Spannung [mV]", fontsize=12)
        plt.axhline(0, color="gray", linestyle="--", linewidth=1)
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend()
        plt.show()

class create_double_detection:
    def __init__(self, numberOfdoubleDetection=10000,sigma = 0.09, components: int = 3, decay: List[float] = [0.164, 0.395, 3.344, 0],intensity: List[float] = [0.88, 0.115, 0.005, 0], debug=False):
        """
        Initialisiert die Anzahl der Double Detections.

        Args:
            numberOfdoubleDetection: Anzahl der Double Detections, die erzeugt werden sollen.
            sigma: Standardabweichung für die zufällige Verschiebung von Puls B.
        """
        self.numberOfdoubleDetection = numberOfdoubleDetection
        self.sigma = sigma
        self.components = components
        self.decay = decay
        self.intensity = intensity
        self.debug = debug
        
        
    def gaussian_distribution(self):
        """
        Erzeugt eine Normalverteilung für zufällige Verschiebungen.

        Returns:
            np.ndarray: Array mit `self.numberOfdoubleDetection` zufälligen Verschiebungen.
        """
        return np.random.normal(0, self.sigma, self.numberOfdoubleDetection)
    
    def lifetime_array(self
        ) -> Optional[np.ndarray]:
        """
        Generate an array of positron lifetimes based on specified decay components and intensities.
    
        Parameters:
            size (int): Number of lifetimes to generate.
            components (int): Number of components to use (1 to 4).
            decay (List[float]): Decay times for each component.
            intensity (List[float]): Intensities for each component.
    
        Returns:
            np.ndarray: Array of generated lifetimes, or None if the intensities do not sum to 1.
        """
        # Ensure valid input
        if self.components < 1 or self.components > len(self.decay):
            raise ValueError("Invalid number of components specified.")
        if not np.isclose(sum(self.intensity[:self.components]), 1.0):
            print("Intensities don't sum up to 1")
            return None
    
        # Generate random uniform values
        r = np.random.uniform(0, 1,self.numberOfdoubleDetection)
    
        # Allocate the lifetime array
        lifetimes = np.zeros(self.numberOfdoubleDetection)
    
        # Determine which decay component applies to each random value
        cumulative_intensity = np.cumsum(self.intensity[:self.components])
        for i, (decay_time, cum_intensity) in enumerate(zip(self.decay[:self.components], cumulative_intensity)):
            mask = (r < cum_intensity) & (lifetimes == 0)
            lifetimes[mask] = np.random.exponential(decay_time, size=np.sum(mask))
    
        return lifetimes
    
    def find_peak_index(self, voltage):
        """
        Findet den Index des Peaks (maximaler Wert) im Spannungsverlauf.

        Args:
            voltage (np.ndarray): Spannungswerte des Pulses.

        Returns:
            int: Index des Peak-Werts.
        """
        return np.argmax(voltage)  # Gibt den Index des maximalen Werts zurück

        
        def shift_pulse(self, time, voltage, shift):
            """
            Verschiebt einen Puls entlang der Zeitachse.

            Args:
                time (np.ndarray): Zeitwerte des Pulses.
                voltage (np.ndarray): Spannungswerte des Pulses.
                shift (float): Verschiebung in ns.

            Returns:
                tuple: Verschobene Zeit- und Spannungswerte.
            """
            shifted_time = time + shift  # Zeitachse um "shift" verschieben
            return shifted_time, voltage
        
        def combine_pulses(self, time_A_all, voltage_A_all, time_B_all, voltage_B_all):
            """
            Erstellt `numberOfdoubleDetection` überlagerte Pulse mit zufälliger Verschiebung von Puls B.
            Es wird für jedes Pulspaar die passende Zeitachse von Puls A verwendet.
        
            Args:
                time_A_all (np.ndarray): Zeitwerte aller Puls A (Shape: (N, num_samples)).
                voltage_A_all (np.ndarray): Spannungswerte aller Puls A (Shape: (N, num_samples)).
                time_B_all (np.ndarray): Zeitwerte aller Puls B (Shape: (N, num_samples)).
                voltage_B_all (np.ndarray): Spannungswerte aller Puls B (Shape: (N, num_samples)).
        
            Returns:
                tuple:
                    - `np.ndarray` mit Shape `(numberOfdoubleDetection, num_samples)` für die individuellen Zeitachsen.
                    - `np.ndarray` mit Shape `(numberOfdoubleDetection, num_samples)` für die summierten Spannungswerte.
            """
            num_pulses, num_samples = time_A_all.shape  # Anzahl der Pulse und Länge der Zeitachse
        
            # 1. Zufällige Verschiebungen für alle Pulse erzeugen
            shift_lifetime = self.lifetime_array() +self.gaussian_distribution() # Shape: (numberOfdoubleDetection,)
        
            # 2. Platzhalter für Zeit- und Spannungswerte
            combined_time = np.zeros((num_pulses, num_samples))
            combined_voltage = np.zeros((num_pulses, num_samples))
        
            # Verarbeitung für jeden Puls individuell
            for i in range(num_pulses):
                # Originale Pulse extrahieren
                time_A = time_A_all[i]  
                voltage_A = voltage_A_all[i]
                time_B = time_B_all[i]
                voltage_B = voltage_B_all[i]
    
                # **Peak finden**
                peak_index_A = self.find_peak_index(voltage_A)
                peak_index_B = self.find_peak_index(voltage_B)
    
                peak_time_A = time_A[peak_index_A]
                peak_time_B = time_B[peak_index_B]
    
                # **Verschiebung berechnen**
                shift_value = peak_time_A - peak_time_B  # So dass Peak B mit Peak A übereinstimmt
                shifted_time_B = time_B + shift_value + shift_lifetime
                
                # **Skalierungsfaktor für Puls B generieren**
                scale_factor = np.random.uniform(0.1, 1.0)  # Zufälliger Faktor zwischen 0.1 und 1.0
                voltage_B_scaled = voltage_B * scale_factor  # Skalierung von B
                
                # **Interpolieren von B auf die Zeitachse von A**
                interp_B = interp1d(shifted_time_B, voltage_B_scaled, kind="linear", fill_value="extrapolate")
                aligned_voltage_B = interp_B(time_A)
    
                # **NaN-Werte durch 0 ersetzen (Fehlende Werte verhindern)**
                aligned_voltage_B = np.nan_to_num(aligned_voltage_B, nan=0.0)
    
                # **Pulse A und B addieren**
                combined_time[i] = time_A  # Zeitachse bleibt gleich
                combined_voltage[i] = voltage_A + aligned_voltage_B
    
                # **DEBUGGING**
                if self.debug and i < 5:  # Nur die ersten 5 Pulse debuggen
                    print(f" Pulse {i}: Peak von A bei {peak_time_A:.2f} ns, Peak von B bei {peak_time_B:.2f} ns (Shift: {shift_value:.2f} ns)")
                    print(f" Peak-Index A: {peak_index_A}, Peak-Index B: {peak_index_B}")
                    print(f" Min/Max Voltage A: {voltage_A.min():.3f} / {voltage_A.max():.3f}")
                    print(f" Min/Max Voltage B (nach Shift): {aligned_voltage_B.min():.3f} / {aligned_voltage_B.max():.3f}\n")
    
            return combined_time, combined_voltage
        
    def shift_pulse(self, time, voltage, shift):
        """
        Verschiebt einen Puls entlang der Zeitachse.

        Args:
            time (np.ndarray): Zeitwerte des Pulses.
            voltage (np.ndarray): Spannungswerte des Pulses.
            shift (float): Verschiebung in ns.

        Returns:
            tuple: Verschobene Zeit- und Spannungswerte.
        """
        shifted_time = time + shift  # Zeitachse um "shift" verschieben
        return shifted_time, voltage
    
    def combine_pulses(self, time_A_all, voltage_A_all, time_B_all, voltage_B_all):
        """
        Erstellt `numberOfdoubleDetection` überlagerte Pulse mit zufälliger Verschiebung von Puls B.
        Es wird für jedes Pulspaar die passende Zeitachse von Puls A verwendet.
    
        Args:
            time_A_all (np.ndarray): Zeitwerte aller Puls A (Shape: (N, num_samples)).
            voltage_A_all (np.ndarray): Spannungswerte aller Puls A (Shape: (N, num_samples)).
            time_B_all (np.ndarray): Zeitwerte aller Puls B (Shape: (N, num_samples)).
            voltage_B_all (np.ndarray): Spannungswerte aller Puls B (Shape: (N, num_samples)).
    
        Returns:
            tuple:
                - `np.ndarray` mit Shape `(numberOfdoubleDetection, num_samples)` für die individuellen Zeitachsen.
                - `np.ndarray` mit Shape `(numberOfdoubleDetection, num_samples)` für die summierten Spannungswerte.
        """
        num_pulses, num_samples = time_A_all.shape  # Anzahl der Pulse und Länge der Zeitachse
    
        # 1. Zufällige Verschiebungen für alle Pulse erzeugen
        shift_values = 0.0# self.lifetime_array() +self.gaussian_distribution() # Shape: (numberOfdoubleDetection,)
    
        # 2. Platzhalter für Zeit- und Spannungswerte
        combined_time = np.zeros((num_pulses, num_samples))
        combined_voltage = np.zeros((num_pulses, num_samples))
    
        # 3. Berechnung für jeden Puls individuell
        for i in range(num_pulses):
            # Richtiges Zeit- und Spannungsprofil für Puls A und B holen
            time_A = time_A_all[i]  # Die Zeitachse von Puls A
            voltage_A = voltage_A_all[i]
            time_B = time_B_all[i]
            voltage_B = voltage_B_all[i]
    
            # Verschiebung anwenden
            shift_value = 0.0#shift_values[i]
            shifted_time_B = time_B + shift_value
    
            # Interpolation von B auf Zeitachse von A
            interp_B = interp1d(shifted_time_B, voltage_B, kind="linear", fill_value="extrapolate")
            aligned_voltage_B = interp_B(time_A)
    
            # 4. Pulse A und B addieren
            combined_time[i] = time_A  # Zeitachse von A beibehalten
            combined_voltage[i] = voltage_A + aligned_voltage_B
    
        return combined_time, combined_voltage
