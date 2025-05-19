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
from scipy.interpolate import interp1d
from scipy.interpolate import Akima1DInterpolator
import traceback
from typing import Tuple,List, Optional
from sklearn.linear_model import LinearRegression
from scipy.stats import norm

m_e_c2 = 511


# Log-normal Verteilung berechnen
def pmt_pulse_new(amplitude, t, t_risefull, pulse_width):
    
    if np.any(t <= 0) or np.any(t_risefull <= 0):
        raise ValueError("Alle Werte in t und t_risefull müssen größer als 0 sein.")
    if np.any(pulse_width <= 0):
        raise ValueError("pulse_width muss größer als 0 sein.")
    
    pmt_pulse_out = amplitude * np.exp(-0.5 * ((np.log(t / t_risefull))**2) / (pulse_width**2))
   
    return pmt_pulse_out

def pmt_pulse_shifted_interp(amplitude, t, t_risefull, pulse_width, shift):
    pulse = amplitude * np.exp(-0.5 * ((np.log(t / t_risefull))**2) / (pulse_width**2))
    shifted_time = t - shift  # Neue Zeitachse
    interp_func = interp1d(t, pulse, bounds_error=False, fill_value=0)  # Interpolationsfunktion
    shifted_pulse = interp_func(shifted_time)  # Verschobenes Signal
    return shifted_pulse

def add_white_noise(signal, noise_level):
    noise = np.random.normal(0, noise_level, size=len(signal))
    return signal + noise

def add_shot_noise(signal, noise_scale):
    noise = np.random.poisson(signal * noise_scale) - (signal * noise_scale)
    return signal + noise

def complet_pulse(amplitude1,amplitude2, t, t_risefull, pulse_width, shift, noise_level, noise_scale):
    if np.any(t <= 0) or np.any(t_risefull <= 0):
        raise ValueError("Alle Werte in t und t_risefull müssen größer als 0 sein.")
    if np.any(pulse_width <= 0):
        raise ValueError("pulse_width muss größer als 0 sein.")
    
    pmt_pulse1 = amplitude1 * np.exp(-0.5 * ((np.log(t / t_risefull))**2) / (pulse_width**2))
    pmt_pulse2 = amplitude2 * np.exp(-0.5 * ((np.log(t / t_risefull))**2) / (pulse_width**2))
    shifted_time = t - shift  # Neue Zeitachse
    interp_func = interp1d(t, pmt_pulse2, bounds_error=False, fill_value=0)  # Interpolationsfunktion
    shifted_pulse = interp_func(shifted_time)
    
    pmt_pulse1_noise = add_shot_noise(pmt_pulse1, noise_scale)
    pmt_pulse1_noise = add_white_noise(pmt_pulse1_noise, noise_level)
    
    
    shifted_pulse_noise = add_shot_noise(shifted_pulse, noise_scale)
    shifted_pulse_noise = add_white_noise(shifted_pulse_noise, noise_level)
    
    
    pmt_result_GT= np.array(pmt_pulse1) + np.array(shifted_pulse)
    pmt_result_noise= np.array(pmt_pulse1_noise) + np.array(shifted_pulse_noise)
    
    return  pmt_result_noise, pmt_result_GT, pmt_pulse1_noise
    
def dervitive1(t, pulse):
    
    # Differenzieren
    dt = t[1] - t[0]
    diff_signal = np.diff(pulse) / dt
    
    return diff_signal

def dervitive2(t, pulse):
    
    # Differenzieren
    dt = t[1] - t[0]
    diff_signal = np.diff(pulse) / dt
    diff_signal2= np.diff(diff_signal) / dt
    
    return diff_signal2

def zero_crossing(Data,t):
    # Nulldurchgänge der zweiten Ableitung finden
    zero_crossings = []
    for i in range(len(Data) - 1):
        if Data[i] * Data[i + 1] < 0:  # Vorzeichenwechsel
            # Lineare Interpolation für genaueren Nulldurchgang
            t_zero = Data[i] - (Data[i] * (t[i + 1] - t[i])) / (
                Data[i + 1] - Data[i]
            )
            zero_crossings.append(t_zero)
            
    return zero_crossings

def generate_random_from_gaussian(num_samples, mean1, std1, weight1, mean2=None, std2=None, weight2=None):
    """
    Generiert Zufallszahlen basierend auf einer Verteilung mit bis zu zwei Gaußkurven.

    Parameters:
        num_samples (int): Anzahl der zu generierenden Zufallszahlen.
        mean1 (float): Mittelwert der ersten Gaußkurve.
        std1 (float): Standardabweichung der ersten Gaußkurve.
        weight1 (float): Gewicht der ersten Gaußkurve (0 bis 1).
        mean2 (float): Mittelwert der zweiten Gaußkurve (optional).
        std2 (float): Standardabweichung der zweiten Gaußkurve (optional).
        weight2 (float): Gewicht der zweiten Gaußkurve (optional, 0 bis 1).

    Returns:
        np.ndarray: Array der generierten Zufallszahlen.
    """
    if weight1 < 0 or (weight2 is not None and weight2 < 0):
        raise ValueError("Intensities müssen nicht-negativ sein.")
    if weight2 is not None:
        if not np.isclose(weight1 + weight2, 1.0):
            raise ValueError("Die Summe der Intensities muss 1 betragen.")
    else:
        if not np.isclose(weight1, 1.0):
            raise ValueError("Das Intensities der ersten Gaußkurve muss 1 betragen, wenn keine zweite angegeben ist.")

    # Zufallszahlen für die erste Gaußkurve
    num_samples1 = int(num_samples * weight1)
    samples1 = np.random.normal(loc=mean1, scale=std1, size=num_samples1)

    if mean2 is not None and std2 is not None and weight2 is not None:
        # Zufallszahlen für die zweite Gaußkurve
        num_samples2 = num_samples - num_samples1
        samples2 = np.random.normal(loc=mean2, scale=std2, size=num_samples2)
        # Kombiniere die beiden Datensätze
        samples = np.concatenate([samples1, samples2])
    else:
        samples = samples1

    # Zufällige Permutation für bessere Mischung
    np.random.shuffle(samples)
    return samples

def lifetime_array(
    size: int,
    components: int = 3,
    decay: List[float] = [164e-12, 395e-12, 2750e-12, 0],
    intensity: List[float] = [0.885, 0.11, 0.005, 0]
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
    if components < 1 or components > len(decay):
        raise ValueError("Invalid number of components specified.")
    if not np.isclose(sum(intensity[:components]), 1.0):
        print("Intensities don't sum up to 1")
        return None

    # Generate random uniform values
    r = np.random.uniform(0, 1, size)

    # Allocate the lifetime array
    lifetimes = np.zeros(size)

    # Determine which decay component applies to each random value
    cumulative_intensity = np.cumsum(intensity[:components])
    for i, (decay_time, cum_intensity) in enumerate(zip(decay[:components], cumulative_intensity)):
        mask = (r < cum_intensity) & (lifetimes == 0)
        lifetimes[mask] = np.random.exponential(decay_time, size=np.sum(mask))

    return lifetimes

def FourierAnalysis (y_input, x_input, samplingrate):
    # Frequenzachse korrekt berechnen
    frequencies = np.fft.fftfreq(len(y_input), d=1/samplingrate)  # Sampling-Rate = 1280 Hz
    fft_result = np.fft.fft(y_input)
    # Nur positive Frequenzen
    positive_frequencies = frequencies[:len(frequencies)//2]
    fft_result_positive = np.abs(fft_result[:len(fft_result)//2])
    fft_result_positive[1:] *= 2
    
    return positive_frequencies, fft_result_positive
   
# Zielverteilung: Rechteck von x=0 bis x=1
def custom_distribution(x):
    return x**2  # Beispielverteilung f(x) = x²

# Rejection Sampling
def rejection_sampling(size, xmin, xmax, max_prob):
    samples = []
    while len(samples) < size:
        x = np.random.uniform(xmin, xmax)  # Zufälliger x-Wert
        y = np.random.uniform(0, max_prob)  # Zufälliger y-Wert
        if y <= custom_distribution(x):  # Akzeptiere den Punkt, falls er unter der Kurve liegt
            samples.append(x)
    return np.array(samples)    

# Compton-Formel für Energie des gestreuten Photons
def compton_energy(E, theta):
    return E / (1 + (E / m_e_c2) * (1 - np.cos(theta)))

# Energieübertragung an Elektron
def energy_transfer(E, theta):
    return E - compton_energy(E, theta)

# Detektorantwort (Gauß-Funktion)
def detector_response(E, E_det, sigma):
    return np.exp(-0.5 * ((E - E_det) / sigma)**2) / (np.sqrt(2 * np.pi) * sigma)

# Winkelverteilung (isotrop)
def angular_distribution(theta):
    return 0.5 * np.sin(theta)

# Pulshöhenspektrum
def pulse_height_spectrum(E, E_det, sigma, num_points=10000):
    thetas = np.linspace(0, np.pi, num_points)  # Winkel von 0 bis π
    d_theta = thetas[1] - thetas[0]  # Winkel-Schrittweite
    spectrum = np.zeros_like(E_det)

    for i, E_d in enumerate(E_det):
        E_e = energy_transfer(E, thetas)
        spectrum[i] = np.sum(
            angular_distribution(thetas) * detector_response(E_e, E_d, sigma) * d_theta
        )
    pdf = spectrum / np.sum(spectrum)  # Normierte PDF
    cdf = np.cumsum(pdf)  # Kumulative Verteilungsfunktion (CDF)
    cdf /= cdf[-1]  # Normiere die CDF
    return spectrum, cdf

def generate_random_values(cdf: np.ndarray,scale:int, size: int) -> np.ndarray:
    """
    Generiert ein Array von Zufallszahlen basierend auf einer CDF.

    Parameters:
    cdf (np.ndarray): Kumulative Verteilungsfunktion (CDF).
    size (int): Anzahl der Zufallszahlen, die generiert werden sollen.

    Returns:
    np.ndarray: Array der generierten Zufallszahlen.
    """
    # Dynamisch E_det basierend auf der Länge von cdf erstellen
    E_det = np.linspace(0, scale, len(cdf))  # Energieachse von 0 bis 500 keV
    
    # Überprüfen, ob die CDF normalisiert ist
    if not np.isclose(cdf[-1], 1.0):
        cdf = cdf / cdf[-1]  # Normierung auf 1
    
    # Zufallswerte zwischen 0 und 1 generieren
    random_cdf_values = np.random.uniform(0, 1, size)
    
    # Interpolation der Zufallswerte basierend auf der CDF
    random_values = np.interp(random_cdf_values, cdf, E_det)

    return random_values

def find_devitation_1_and_2_difference(xs, y_akima,y_akima_GT ):
    try:
        #GT nur start Pulse
        diff_signal = dervitive1(xs, y_akima)
        diff_signal_GT = dervitive1(xs, y_akima_GT)
        
        # Zweite Ableitung berechnen
        second_diff_signal = dervitive2(xs, y_akima)
        second_diff_signal_GT = dervitive2(xs, y_akima_GT)
        
   
        zero_crossings = zero_crossing(second_diff_signal, xs[:-2])
        zero_crossings_GT = zero_crossing(second_diff_signal_GT, xs[:-2])
        
        zero_crossings_first = zero_crossing(diff_signal, xs[:-1])
        zero_crossings_first_GT = zero_crossing(diff_signal_GT, xs[:-1])

        # Schutz vor leeren Listen
        if not zero_crossings or not zero_crossings_first or not zero_crossings_GT or not zero_crossings_first_GT:
            print("Warnung: Keine Nulldurchgänge gefunden")
            return 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1

        first_crossing = zero_crossings_first[0]
        first_crossing_GT = zero_crossings_first_GT[0]
        
        peak_to_rise = zero_crossings[0] - zero_crossings_first[0]
        peak_to_down = zero_crossings_first[0] - zero_crossings[1]
        
        peak_to_rise_GT = zero_crossings_GT[0] - zero_crossings_first_GT[0]
        peak_to_down_GT = zero_crossings_first_GT[0] - zero_crossings_GT[1]
        
        difference_crossing = zero_crossings[1] - zero_crossings[0]
        difference_crossing_GT = zero_crossings_GT[1] - zero_crossings_GT[0]
        
        GT_differnce = difference_crossing - difference_crossing_GT
        
        return first_crossing, peak_to_rise, peak_to_down, difference_crossing,first_crossing_GT, peak_to_rise_GT ,peak_to_down_GT, difference_crossing_GT,GT_differnce
    
    
    except IndexError as e:
        print(f"IndexError in find_devitation_1_and_2_difference: {e}")
        return 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1

def process_doubleDetection(i,pulse_params, digitizer_params, amplitude, amplitude_511,lifetime):
    
    try:
        
        #for corrupted coincidence
     
        amplitude = amplitude
        amplitude_511 = amplitude_511
        true_PHS_1275= amplitude
        true_PHS_511 = amplitude_511
                
                
        amplitude_Value_511_GT = amplitude_511
    
        lifetime_value = lifetime
            
        # Log-normal Verteilung berechnen
        before_noiese_pmt_pulse = pmt_pulse_new(amplitude, digitizer_params["t"], digitizer_params["t_risefull"], digitizer_params["pulse_width"])
            
        before_noiese_pmt_pulse511 = pmt_pulse_shifted_interp(amplitude_511, digitizer_params["t"], digitizer_params["t_risefull"], digitizer_params["pulse_width"],lifetime_value)
            
        pmt_pulse = add_shot_noise(before_noiese_pmt_pulse, digitizer_params["noise_scale"])    
        pmt_pulse = add_white_noise(pmt_pulse, digitizer_params["noise_level"])
        
        pmt_pulse511 =  add_shot_noise(before_noiese_pmt_pulse511, digitizer_params["noise_scale"])    
        pmt_pulse511 =  add_white_noise(pmt_pulse511, digitizer_params["noise_level"])
                
        if len(pmt_pulse) != len(pmt_pulse511):
            raise ValueError("pmt_pulse und pmt_pulse511 haben unterschiedliche Längen.")
    
        pmt_result= np.array(pmt_pulse) + np.array(pmt_pulse511)
           
        # Akima-Interpolation
        # Interpolationspunkte
        xs = np.linspace(min(digitizer_params["t"]), max(digitizer_params["t"]), num=1280)
        y_akima = Akima1DInterpolator(digitizer_params["t"], pmt_result)(xs)
        y_akima_GT = Akima1DInterpolator(digitizer_params["t"], pmt_pulse)(xs)
        
        # 30 % der maximalen Amplitude für Pile Up 
        pmt_amplitude = np.max(y_akima)
        threshold = (digitizer_params["CF_level"]/100) * pmt_amplitude
        
        # 30 % der maximalen Amplitude für GT
        pmt_amplitude_GT = np.max(y_akima_GT)
        threshold_GT = (digitizer_params["CF_level"]/100) * pmt_amplitude_GT
        
        # Finde den ersten Index, bei dem das Signal den Schwellwert überschreitet
        index = np.where(y_akima >= threshold)[0][0]
        index_GT = np.where(y_akima_GT >= threshold_GT)[0][0]
        
        total_sum = np.sum(pmt_result)
        total_sum_GT = np.sum(pmt_pulse)
            
                       
        first_crossing, peak_to_rise, peak_to_down, difference_crossing,first_crossing_GT, peak_to_rise_GT ,peak_to_down_GT, difference_crossing_GT,GT_differnce = find_devitation_1_and_2_difference(xs, y_akima, y_akima_GT)
    
        
        return {            
            "differenceCrossing": difference_crossing,
            "amplitude_Value": pmt_amplitude,
            "differenceCrossing_GT": difference_crossing_GT,
            "amplitude_Value_GT": pmt_amplitude_GT,
            "Peak_to_rise":peak_to_rise,
            "peak_to_down":peak_to_down,
            "peak_to_rise_GT":peak_to_rise_GT,
            "peak_to_down_GT": peak_to_down_GT,
            "Ratio_Zero_crossing": peak_to_rise/peak_to_down,
            "Ratio_Zero_crossing_GT": peak_to_rise_GT/peak_to_down_GT,
            "GT_differnce":GT_differnce,
            "total_sum":total_sum,
            "total_sum_GT":total_sum_GT,
            "lifetime_value":lifetime_value,
            "true_PHS_1275":true_PHS_1275,
            "true_PHS_511": true_PHS_511,
            "amplitude_Value_511_GT": amplitude_Value_511_GT,
            "index":index,
            "index_GT":index_GT
            }
    except Exception as e:
        print(f"Fehler in Funktion für Index {i}: {e}")
        traceback.print_exc()  # Gibt den vollständigen Stacktrace aus
        return None

def process_pulses(amplitude1, amplitude2, t, t_risefull, pulse_width, shift, noise_level, noise_scale):
    pulse1 = pmt_pulse_new(amplitude1, t, t_risefull, pulse_width)
    pulse2 = pmt_pulse_shifted_interp(amplitude2, t, t_risefull, pulse_width, shift)
    
    pulse1_noise = add_shot_noise(pulse1, noise_scale)
    pulse1_noise = add_white_noise(pulse1_noise, noise_level)
    
    pulse2_noise = add_shot_noise(pulse2, noise_scale)
    pulse2_noise = add_white_noise(pulse2_noise, noise_level)
    
    return pulse1_noise, pulse2_noise
    
def pileUpCreation(i,pulse_params, digitizer_params, random_values_uneven,random_values_even, amplitude, amplitude_511, amplitude_2, amplitude_511_2):
    try:
        szenario = 0

        rd_value = random_values_uneven * 4  # Skalieren, um den Bereich 0 bis 4 zu erhalten
        shift_value = random_values_even * digitizer_params["pulse_length"]
        if (rd_value<=1):
            pmt_pulse1275_noise, pmt_pulse511_noise = process_pulses(
                amplitude, amplitude_511, digitizer_params["t"], digitizer_params["t_risefull"],
                digitizer_params["pulse_width"], shift_value, digitizer_params["noise_level"], digitizer_params["noise_scale"]
            )
            pmt_result = np.array(pmt_pulse1275_noise) + np.array(pmt_pulse511_noise)
            Pulse_1 = pmt_pulse1275_noise
            szenario = 1            
        elif (rd_value<=2):
            
            pmt_pulse511_noise,pmt_pulse1275_noise = process_pulses(
                 amplitude_511, amplitude, digitizer_params["t"], digitizer_params["t_risefull"],
                digitizer_params["pulse_width"], shift_value, digitizer_params["noise_level"], digitizer_params["noise_scale"]
            )
            pmt_result = np.array(pmt_pulse1275_noise) + np.array(pmt_pulse511_noise)
            Pulse_1 = pmt_pulse511_noise
            szenario = 2   
        elif (rd_value<=3):
            pmt_pulse511_noise,pmt_pulse511_noise_2 = process_pulses(
                 amplitude_511, amplitude_511_2, digitizer_params["t"], digitizer_params["t_risefull"],
                digitizer_params["pulse_width"], shift_value, digitizer_params["noise_level"], digitizer_params["noise_scale"]
            )
            pmt_result = np.array(pmt_pulse511_noise) + np.array(pmt_pulse511_noise_2)
            Pulse_1 = pmt_pulse511_noise
            szenario = 3            
        elif (rd_value<=4):
            pmt_pulse1275_noise,pmt_pulse1275_noise_2 = process_pulses(
                 amplitude, amplitude_2, digitizer_params["t"], digitizer_params["t_risefull"],
                digitizer_params["pulse_width"], shift_value, digitizer_params["noise_level"], digitizer_params["noise_scale"]
            )
            pmt_result = np.array(pmt_pulse1275_noise) + np.array(pmt_pulse1275_noise_2)
            Pulse_1 = pmt_pulse1275_noise
            szenario = 4
        else:
            print("Keine Bedingung erfüllt")
            
        # Akima-Interpolation
        xs = np.linspace(min(digitizer_params["t"]), max(digitizer_params["t"]), num= digitizer_params["digitation_points"]*digitizer_params["render_points"])
        y_akima = Akima1DInterpolator(digitizer_params["t"], pmt_result)(xs)
        y_akima_GT = Akima1DInterpolator(digitizer_params["t"], Pulse_1)(xs)
        
        # 30 % der maximalen Amplitude für Pile Up 
        pmt_amplitude = np.max(y_akima)
        threshold = (digitizer_params["CF_level"]/100) * pmt_amplitude
        
        # 30 % der maximalen Amplitude für GT
        pmt_amplitude_GT = np.max(y_akima_GT)
        threshold_GT = (digitizer_params["CF_level"]/100) * pmt_amplitude_GT
        
        # Finde den ersten Index, bei dem das Signal den Schwellwert überschreitet
        index = np.where(y_akima >= threshold)[0][0]
        index_GT = np.where(y_akima_GT >= threshold_GT)[0][0]
        
        total_sum = np.sum(pmt_result)
        total_sum_GT = np.sum(Pulse_1)
           
   
        first_crossing, peak_to_rise, peak_to_down, difference_crossing,first_crossing_GT, peak_to_rise_GT ,peak_to_down_GT, difference_crossing_GT,GT_differnce = find_devitation_1_and_2_difference(xs, y_akima, y_akima_GT)         
        
        return {
            "differenceCrossing": difference_crossing,
            "Szenario": szenario,
            "amplitude_Value": pmt_amplitude,
            "differenceCrossing_GT": difference_crossing_GT,
            "amplitude_Value_GT": pmt_amplitude_GT,
            "Peak_to_rise":peak_to_rise,
            "peak_to_down":peak_to_down,
            "peak_to_rise_GT":peak_to_rise_GT,
            "peak_to_down_GT": peak_to_down_GT,
            "Ratio_Zero_crossing": peak_to_rise/peak_to_down,
            "Ratio_Zero_crossing_GT": peak_to_rise_GT/peak_to_down_GT,
            "GT_differnce":GT_differnce,
            "total_sum":total_sum,
            "total_sum_GT":total_sum_GT,
            "index":index,
            "index_GT":index_GT
            }
    except Exception as e:
        print(f"Fehler in Funktion für Index {i}: {e}")
        traceback.print_exc()  # Gibt den vollständigen Stacktrace aus
        return None

def generate_amplitudes(i, cdf, cdf_511) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate amplitude and amplitude_511 arrays based on given CDFs.

    Parameters:
        i (int): Number of values to generate.
        cdf (np.ndarray): CDF for the first distribution.
        cdf_511 (np.ndarray): CDF for the 511 keV distribution.

    Returns:
        tuple: Arrays for amplitude and amplitude_511.
    """
    batch_size = max(i, 100)  # Minimum batch size of 100

    # Generate amplitudes in batches
    amplitude = generate_random_values(cdf, 1400, batch_size)[:i]
    amplitude_511 = generate_random_values(cdf_511, 700, batch_size)[:i]

    return amplitude, amplitude_511

def generate_amplitude_doubleDetection(i,cdf, cdf_511,threshold)->Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Initialisiere leere Listen für gültige Werte
    valid_amplitude = []
    valid_amplitude_511 = []
    batch_size = max(i, 100)  # Batch-Größe: min. i oder 100 Werte auf einmal generieren
    max_1 = 500
    max_2 = 500
    while len(valid_amplitude) < i:  # Wiederhole, bis genug Werte gesammelt sind
        # Generiere in großen Batches
        amplitude = generate_random_values(cdf,max_1, batch_size)
        amplitude_511 = generate_random_values(cdf_511,max_2, batch_size)
        
        amplitude_True = generate_random_values(cdf,max_1, batch_size)
        amplitude_511_True = generate_random_values(cdf_511,max_2, batch_size)
        
        # Filtere gültige Werte
        valid_indices = (amplitude + amplitude_511 > threshold) & (amplitude_511> 10 )        
        valid_amplitude.extend(amplitude[valid_indices])
        valid_amplitude_511.extend(amplitude_511[valid_indices])
    
    # Begrenze die Ergebnisse auf genau i Elemente
    valid_amplitude = np.array(valid_amplitude[:i])
    valid_amplitude_511 = np.array(valid_amplitude_511[:i])
    
    return valid_amplitude, valid_amplitude_511, amplitude_True,amplitude_511_True

def linearLimt_filter_use(automation, area, amplitude, area_GT, amplitude_GT, Steigung, startValue,shift):
    i = 0
    i_GT = 0
    if automation:
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
        i = np.sum(np.array(area) > y_limits)

        y_limits_GT = model.predict(np.array(amplitude_GT).reshape(-1, 1)) + y_shift
        i_GT = np.sum(np.array(area_GT) > y_limits_GT)

    else:
        # Schwelle manuell berechnen
        steigung = Steigung
        start_value = startValue

        y_thresholds = steigung * np.array(amplitude) + start_value
        i = np.sum(np.array(area) > y_thresholds)

        y_thresholds_GT = steigung * np.array(amplitude_GT) + start_value
        i_GT = np.sum(np.array(area_GT) > y_thresholds_GT)

        # Berechnung der Werte für die Schwellenlinie
        x_value = np.linspace(min(amplitude), max(amplitude), 100)
        y_value = steigung * x_value + start_value

    return x_value.tolist(), y_value.tolist(), i, i_GT

def linearLimt_lower_filter_use(automation, area, amplitude, area_GT, amplitude_GT, crossing_line):
    i = 0
    i_GT = 0
    if automation:
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

    else:
        # Schwelle manuell berechnen
        y_thresholds = crossing_line
        x_value = np.linspace(min(amplitude), max(amplitude), 100)
        y_value = np.full_like(x_value, y_thresholds)  # NumPy-Array mit konstantem Wert

        i = np.sum(np.array(area) < y_thresholds)
        i_GT = np.sum(np.array(area_GT) < y_thresholds)


    return x_value.tolist(), y_value.tolist(), i, i_GT

def klein_nishina(E):
    """
    Klein-Nishina Formel für Compton-Streuung.
    E: Energie der Photonen (in MeV).
    Rückgabe: Dimensionslose Wahrscheinlichkeit.
    """
    m_e = 0.511  # Elektronenruhemasse in MeV
    epsilon = E / m_e  # Energieverhältnis
    term1 = (1 + epsilon) / epsilon**2
    term2 = (2 * (1 + epsilon) / (1 + 2 * epsilon)) - np.log(1 + 2 * epsilon) / epsilon
    term3 = np.log(1 + 2 * epsilon) / (2 * epsilon)
    return term1 * (term2 + term3)

def interaction_ratios(Z_eff, E):
    """
    Berechnet das Verhältnis von Photoeffekt und Compton-Streuung.
    Z_eff: Effektive Kernladungszahl des Materials.
    E: Photonenenergie (in MeV, numpy Array).
    Rückgabe: Verhältnis R(Z_eff, E), Wahrscheinlichkeiten für Photoeffekt und Compton.
    """
    n = 4.5  # Abhängigkeit von Z_eff
    kappa = 3  # Energieabhängigkeit des Photoeffekts
    sigma_photo = Z_eff**n * E**-kappa  # Photoeffekt-Wahrscheinlichkeit
    sigma_compton = Z_eff * klein_nishina(E)  # Compton-Streuung
    sigma_total = sigma_photo + sigma_compton
    P_photo = sigma_photo / sigma_total
    P_compton = sigma_compton / sigma_total
    return P_photo, P_compton

def broaden_spectrum(spectrum, resolution):
    """
    Breitenspektrum, um die Detektorauflösung zu simulieren.
    spectrum: Eingabespektrum.
    resolution: Relative Energieauflösung (FWHM/E).
    Rückgabe: Geglättetes Spektrum.
    """
    broadened = np.convolve(spectrum, norm.pdf(np.linspace(-3, 3, len(spectrum)), scale=resolution * len(spectrum)), mode='same')
    return broadened / broadened.sum()

def pulse_height_spectrum_with_photopeak(E, E_det, sigma_compton, sigma_photo, Z_eff, num_points=10000):
    """
    Erzeugt das Pulshöhenspektrum (PDF und CDF) unter Berücksichtigung von Photopeak und Comptoneffekt.

    Parameters:
        E: float
            Eingangsenergie des Photons (in MeV).
        E_det: np.ndarray
            Array der Detektorenergien.
        sigma: float
            Detektorauflösung (FWHM/E).
        Z_eff: float
            Effektive Kernladungszahl des Materials.
        num_points: int
            Anzahl der Winkelpunkte für die Integration.

    Returns:
        spectrum: np.ndarray
            Normiertes Pulshöhenspektrum.
        cdf: np.ndarray
            Kumulative Verteilungsfunktion (CDF).
    """
    thetas = np.linspace(0, np.pi, num_points)  # Winkel von 0 bis π
    d_theta = thetas[1] - thetas[0]  # Winkel-Schrittweite
    spectrum = np.zeros_like(E_det)

    # Berechne das Verhältnis von Photoeffekt zu Compton
    P_photo, P_compton = interaction_ratios(Z_eff, E)

    for i, E_d in enumerate(E_det):
        # Energieübertrag für Comptonstreuung
        E_e = energy_transfer(E, thetas)

        # Spektrum für den Comptoneffekt
        compton_spectrum = np.sum(
            angular_distribution(thetas) * detector_response(E_e, E_d, sigma_compton) * d_theta
        )

        # Spektrum für den Photopeak (Gauss)
        photopeak_spectrum = norm.pdf(E_d, loc=E, scale=sigma_photo * E)

        # Gesamt-Spektrum kombinieren
        spectrum[i] = P_compton * compton_spectrum + P_photo * photopeak_spectrum

    # Normieren und CDF berechnen
    pdf = spectrum / np.sum(spectrum)  # Normierte PDF
    cdf = np.cumsum(pdf)  # Kumulative Verteilungsfunktion
    cdf /= cdf[-1]  # Normiere die CDF

    return pdf, cdf

def generate_histogram_values(amplitude: np.ndarray, scale: int, bins: int) -> np.ndarray:
    """
    Generate a histogram array based on a cumulative distribution function (CDF).

    Parameters:
        cdf (np.ndarray): The cumulative distribution function (CDF), normalized to a maximum value of 1.
        scale (int): The range or scale of the values, e.g., 0 to scale.
        size (int): The number of random values to generate.
        bins (int): The number of bins for the histogram.

    Returns:
        np.ndarray: Array containing histogram bin counts.
    """
    # Compute histogram
    hist, _ = np.histogram(amplitude, bins=bins, range=(0, scale))

    return hist
