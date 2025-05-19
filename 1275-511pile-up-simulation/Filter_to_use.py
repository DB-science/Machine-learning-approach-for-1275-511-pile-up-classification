#*************************************************************************************************
#** Copyright (c) 2025 Dr. Dominik Boras. All rights reserved.
#** 
#** Redistribution and use in source and binary forms, with or without modification, 
#** are permitted provided that the following conditions are met:
#**
#** 1. Redistributions of source code must retain the above copyright notice
#**    this list of conditions and the following disclaimer.
#**
#** 2. Redistributions in binary form must reproduce the above copyright notice, 
#**    this list of conditions and the following disclaimer in the documentation 
#**    and/or other materials provided with the distribution.
#**
#** 3. Neither the name of the copyright holder "Danny Petschke" nor the names of its  
#**    contributors may be used to endorse or promote products derived from this software  
#**    without specific prior written permission.
#**
#**
#** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS 
#** OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF 
#** MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE 
#** COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, 
#** EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
#** SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) 
#** HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR 
#** TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, 
#** EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#**
#** Contact: dominik.boras@uni-wuerzburg.de
#*************************************************************************************************
import numpy as np
from sklearn.linear_model import LinearRegression
from numba import jit

@jit(nopython=True)
def dervitive1(t, pulse):
    dt = t[1] - t[0]
    return np.diff(pulse) / dt

@jit(nopython=True)
def dervitive2(t, pulse):
    dt = t[1] - t[0]
    diff_signal = np.diff(pulse) / dt
    return np.diff(diff_signal) / dt


@jit(nopython=True)
def zero_crossing(data, t):
    zero_crossings = []
    for i in range(len(data) - 1):
        if data[i] * data[i + 1] < 0:
            t_zero = t[i] - (data[i] * (t[i + 1] - t[i])) / (data[i + 1] - data[i])
            zero_crossings.append(t_zero)
    return zero_crossings


@jit(nopython=True)
def find_devitation_1_and_2_difference(xs, y_akima, y_akima_GT):
    diff_signal = dervitive1(xs, y_akima)
    diff_signal_GT = dervitive1(xs, y_akima_GT)

    second_diff_signal = dervitive2(xs, y_akima)
    second_diff_signal_GT = dervitive2(xs, y_akima_GT)

    zero_crossings = zero_crossing(second_diff_signal, xs)
    zero_crossings_GT = zero_crossing(second_diff_signal_GT, xs)

    zero_crossings_first = zero_crossing(diff_signal, xs)
    zero_crossings_first_GT = zero_crossing(diff_signal_GT, xs)

    if len(zero_crossings) < 2 or len(zero_crossings_first) < 1 or len(zero_crossings_GT) < 2 or len(zero_crossings_first_GT) < 1:
        # Rückgabe von 0.1-Werten bei Fehler
        return np.array([0.1] * 9)

    peak_to_rise = zero_crossings[0] - zero_crossings_first[0]
    peak_to_down = zero_crossings_first[0] - zero_crossings[1]

    peak_to_rise_GT = zero_crossings_GT[0] - zero_crossings_first_GT[0]
    peak_to_down_GT = zero_crossings_first_GT[0] - zero_crossings_GT[1]

    difference_crossing = zero_crossings[1] - zero_crossings[0]
    difference_crossing_GT = zero_crossings_GT[1] - zero_crossings_GT[0]

    return np.array([
        zero_crossings_first[0], peak_to_rise, peak_to_down, difference_crossing,
        zero_crossings_first_GT[0], peak_to_rise_GT, peak_to_down_GT, difference_crossing_GT
    ])

def linearLimt_filter_use( area, amplitude, area_GT, amplitude_GT,shift):
    i = 0
    i_GT = 0
    # Erstellen des Dictionaries für maximale y-Werte
    unique_amplitude_GT, max_area_GT = np.unique(amplitude_GT, return_index=True)
    max_y_for_x = {x: area_GT[idx] for x, idx in zip(unique_amplitude_GT, max_area_GT)}

    # Daten für Regression vorbereiten
    x_vals = np.array(list(max_y_for_x.keys())).reshape(-1, 1)
    y_vals = np.array(list(max_y_for_x.values()))

    # Lineare Regression
    model = LinearRegression()
    model.fit(x_vals, y_vals)

    # Regressionsgerade berechnen
    x_value = np.linspace(min(x_vals), max(amplitude), 1000).reshape(-1, 1)
    y_shift = shift
    y_value = model.predict(x_value) + y_shift

    # Überprüfen, ob Punkte über der Grenze liegen
    y_limits = model.predict(np.array(amplitude).reshape(-1, 1)) + y_shift
    i = np.sum(np.array(area) < y_limits)

    y_limits_GT = model.predict(np.array(amplitude_GT).reshape(-1, 1)) + y_shift
    i_GT = np.sum(np.array(area_GT) < y_limits_GT)

    return x_value.tolist(), y_value.tolist(), i, i_GT

def linearLimt_lower_filter_use(area, amplitude, area_GT, amplitude_GT):
    
    i = 0
    i_GT = 0

    # Überprüfen, ob die Eingaben nicht leer sind
    if len(area_GT) == 0 or len(amplitude_GT) == 0:
        raise ValueError("area_GT und amplitude_GT dürfen nicht leer sein.")

    # Median und Standardabweichung berechnen
    y_median = np.median(area_GT)
    # Median Absolute Deviation (MAD)
    mad = np.median(np.abs(area_GT - y_median))
    y_shift = y_median - 3 * mad

    # Schwellenlinie berechnen (konstante Linie)
    x_value = np.linspace(min(amplitude_GT), max(amplitude), 1000)
    y_value = np.full_like(x_value, y_shift)

    # Überprüfen, ob Punkte unterhalb der Schwelle liegen
    i = np.sum(np.array(area) < y_shift)
    i_GT = np.sum(np.array(area_GT) < y_shift)

    return x_value.tolist(), y_value.tolist(), i, i_GT

def find_10percent_crossings(signal):
    # 1) Minimum (negativer Peak) bestimmen
    min_val = np.min(signal)
    
    # 2) 10%-Schwelle definieren
    threshold = 0.1 * min_val  # Bsp.: min_val = -100 -> threshold = -10
    
    # 3) Index-Übergänge suchen
    # Wir betrachten dazu ein bool-Array, das True ist, wo signal > threshold,
    # und False, wo signal <= threshold.
    above_threshold = signal > threshold
    
    # a) Leading Edge: erster Index, an dem das Signal von >threshold (True) 
    #    auf <=threshold (False) wechselt => Übergang True -> False
    #    In diff: True->False bedeutet -1
    diff_array = np.diff(above_threshold.astype(int))
    indices_leading = np.where(diff_array == -1)[0]
    idx_leading_10pct = indices_leading[0] if len(indices_leading) > 0 else 0
    
    # b) Trailing Edge: erster Index, an dem das Signal von <=threshold (False)
    #    auf >threshold (True) wechselt => Übergang False -> True
    #    In diff: False->True bedeutet +1
    indices_trailing = np.where(diff_array == 1)[0]
    idx_trailing_10pct = indices_trailing[0] if len(indices_trailing) > 0 else 0
    
    return idx_leading_10pct, idx_trailing_10pct
