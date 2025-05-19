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
#**
#**This script uses the DLTPulseGenerator library developed by Dr. Danny Petschke 
#**for generating realistic PMT pulse shapes.
#**
#**DLTPulseGenerator is used under the terms of its license (BSD-3-Clause-licence).
#**Please refer to the original repository for more details:
#**https://github.com/dpscience/DLTPulseGenerator
#*************************************************************************************************


PATH_TO_LIBRARY = '/DLTPulseGenerator.dll'

import pyDLTPulseGenerator as dpg

import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import PHS_creation
from PHS_definition import PHS_parameter,filterParameter 
import Filter_to_use
from scipy.interpolate import Akima1DInterpolator
import InitList
import plotting
import h5py

numberOfPulses = 1000 # number of pulses to be generated ...

# set trigger levels for branches A & B [mV] ...
    
triggerA = -25 # [mV]
triggerB = -25 # [mV]



if __name__ == '__main__':
    start_time = time.time()
    
    spectrum, cdf = PHS_creation.pulse_height_spectrum(PHS_parameter.PhotonEnergyNr1, 
                                                       PHS_parameter.EnergyScaling, 
                                                       PHS_parameter.EnergyResolutionCompton)
    
    spectrum, cdf_511 = PHS_creation.pulse_height_spectrum(PHS_parameter.PhotonEnergyNr2, 
                                                               PHS_parameter.EnergyScaling, 
                                                               PHS_parameter.EnergyResolutionCompton*2)
    
    
    spectrum, cdf_cs = PHS_creation.pulse_height_spectrum_with_photopeak(PHS_parameter.PhotonEnergyNr1, 
                                                               PHS_parameter.EnergyScaling, 
                                                               PHS_parameter.EnergyResolutionCompton,
                                                               PHS_parameter.EnergyResolutionPhotoeffect, 
                                                               PHS_parameter.Z_eff)
    
    spectrum, cdf_511_cs = PHS_creation.pulse_height_spectrum_with_photopeak(PHS_parameter.PhotonEnergyNr2, 
                                                                   PHS_parameter.EnergyScaling, 
                                                                   PHS_parameter.EnergyResolutionCompton,
                                                                   PHS_parameter.EnergyResolutionPhotoeffect*5, 
                                                                   PHS_parameter.Z_eff)
    
    
    
    if (PHS_parameter.SolidScintillator == True):
        amplitude, amplitude_511 = PHS_creation.generate_amplitudes(numberOfPulses, 
                                                          cdf_cs, 
                                                          cdf_511_cs)
        
        PHS_value_1 = PHS_creation.generate_histogram_values(amplitude, 
                                                          PHS_parameter.NumberOfDigitizationPoints,
                                                          500)
        PHS_value_2 = PHS_creation.generate_histogram_values(amplitude_511, 
                                                          PHS_parameter.NumberOfDigitizationPoints,
                                                          500)
    else:
        amplitude, amplitude_511 = PHS_creation.generate_amplitudes(numberOfPulses, 
                                                          cdf, 
                                                          cdf_511)
        
        PHS_value_1 = PHS_creation.generate_histogram_values(amplitude, 
                                                          PHS_parameter.NumberOfDigitizationPoints,
                                                          500)
        PHS_value_2 = PHS_creation.generate_histogram_values(amplitude_511, 
                                                          PHS_parameter.NumberOfDigitizationPoints,
                                                          500)
    
    
    dpg.__information__() 
    dpg.__licence__()
    
    # define your simulation input ...
    
    lt          = dpg.DLTSimulationInput()
    setupInfo   = dpg.DLTSetup()
    pulseInfo   = dpg.DLTPulse()
    
    phs         = dpg.DLTPHS()


    stopPHS = PHS_value_2
    startPHS = PHS_value_1
    #stopPHS,startPHS = np.loadtxt('phs.txt', delimiter='\t', skiprows=2, unpack=True, dtype='float')
    
    # show data for verification ...
    
    fig,ax = plt.subplots()
    
    plt.plot(stopPHS/np.sum(stopPHS),'r-',label="BC422-Q | 22-Na | 511 keV")
    plt.plot(startPHS/np.sum(startPHS),'b-',label="BC422-Q | 22-Na | 1274 keV")
    plt.plot(((startPHS/np.sum(startPHS))+(stopPHS/np.sum(stopPHS))),'g-',label="superimposed PHS input")
    
    plt.legend(loc='best')
    
    ax.set_ylabel('pdf [a.u.]')
    ax.set_xlabel('amplitude distribution [a.u.]')
    
    plt.show()
    
    # modify PHS data structure in order to apply the loaded distributions (PHS) ...
    
    phs.m_useGaussianModels = False # indicate the library that we want to apply our own distribution ...
    
    phs.m_distributionStartA = stopPHS
    phs.m_distributionStopA = stopPHS
    phs.m_resolutionMilliVoltPerStepA = float(5/6)*np.abs(pulseInfo.m_amplitude_in_milliVolt)/len(startPHS) # fit PHS into the 5/6 of the max. amplitude ...
    phs.m_gridNumberA = 1024
   
    
   
    phs.m_distributionStartB = startPHS
    phs.m_distributionStopB = startPHS
    phs.m_resolutionMilliVoltPerStepB = float(5/6)*np.abs(pulseInfo.m_amplitude_in_milliVolt)/len(startPHS) # fit PHS into the 5/6 of the max. amplitude ...
    phs.m_gridNumberB = 1024
    
    maxAmplitude  = pulseInfo.m_amplitude_in_milliVolt
    sweep_in_ns   = setupInfo.m_sweep_in_nanoSeconds
    
    # initalize pulse generator ...
    
    pulseGen = dpg.DLTPulseGenerator(phs,
                                     lt,
                                     setupInfo,
                                     pulseInfo,
                                     PATH_TO_LIBRARY)
    
    # catch errors ...
    
    if not pulseGen.printErrorDescription():
        quit() # kill process on error ...

    pulseA = dpg.DLTPulseF() # pulse of detector A
    pulseB = dpg.DLTPulseF() # pulse of detector B
    
    pulsesShown = False
    numberPHSBins = 1024 # define a binning for the PHS (DDRS4PALS is using 1024)
    
    phsA = [0]*numberPHSBins
    phsB = [0]*numberPHSBins
    
    phsCorrupted = [0]*numberPHSBins
    
    n = 0
    corrupted_amplitude = []
    for i in tqdm(range(numberOfPulses), desc="Processing items"):

        while True:  # Wiederholen, bis die Bedingung erfüllt ist
            # Pulse generieren
            if not pulseGen.emitPulses(pulseA, pulseB, triggerA, triggerB):
                continue
           
            # Prüfen, ob die Bedingung des while erfüllt ist
            if pulseA.getMinimumVoltage() + pulseB.getMinimumVoltage() < -150 and \
               -500 < pulseA.getMinimumVoltage() < -20 and \
               -500 < pulseB.getMinimumVoltage() < -20:
                
                # Bedingung erfüllt, Schleife verlassen
                break
            
            if (abs( pulseA.getMinimumIndex() - pulseB.getMinimumIndex()) > abs(2.5 / 0.195)):
                print(abs( pulseA.getMinimumIndex() - pulseB.getMinimumIndex()))
                break
        x_A = pulseA.getTime()       
        y_A = pulseA.getVoltage()
        y_B = pulseB.getVoltage()
        y_corrupted = np.add(y_A, y_B)

        xs = np.linspace(min(x_A), max(x_A), num=PHS_parameter.NumberOfDigitizationPoints * filterParameter.renderPoints)
        yAkimaCorrupted = Akima1DInterpolator(x_A, y_corrupted)(xs)
        yAakimaGT = Akima1DInterpolator(x_A, y_B)(xs)
        min_value = np.min(yAkimaCorrupted)
        max_value = np.max(yAkimaCorrupted)
        min_value_GT = np.min(yAakimaGT)
        max_value_GT = np.max(yAakimaGT)
        
        Area = sum(yAkimaCorrupted)
        Area_GT = sum(yAakimaGT)

        first_10p, last_10p = Filter_to_use.find_10percent_crossings(yAkimaCorrupted)
        first_10p_GT, last_10p_GT = Filter_to_use.find_10percent_crossings(yAakimaGT)

        idx_min = np.argmin(yAkimaCorrupted)
        idx_min_GT = np.argmin(yAakimaGT)
        equidistance_time = dpg.DLTSetup.m_sweep_in_nanoSeconds/ (PHS_parameter.NumberOfDigitizationPoints * filterParameter.renderPoints) # in ns

        risetime_10p = (idx_min * equidistance_time)-(first_10p * equidistance_time)# in ns
        falltime_10p = (last_10p * equidistance_time)-(idx_min * equidistance_time)# in ns

        risetime_10p_GT = (idx_min_GT * equidistance_time)-(first_10p_GT * equidistance_time)# in ns
        falltime_10p_GT = (last_10p_GT * equidistance_time)-(idx_min_GT * equidistance_time)# in ns

        ratio_rt_ft = risetime_10p/falltime_10p
        ratio_rt_ft_GT = risetime_10p_GT/falltime_10p_GT

        pulslength_10p = falltime_10p - risetime_10p # in ns
        pulslength_10p_GT = falltime_10p_GT - risetime_10p_GT # in ns


        m_fC, m_PtR, m_PtD, m_dfCrossing,m_fC_GT, m_PtR_GT ,m_PtD_GT, m_dfCrossing_GT = Filter_to_use.find_devitation_1_and_2_difference(xs, yAkimaCorrupted, yAakimaGT)
        
        

        if Area<-1e9 or min_value>1 :
            Area = 0
           

        if Area_GT<-1e9 or min_value_GT>1  :
            Area_GT = 0
            
        
        if pulseInfo.m_amplitude_in_milliVolt > 0:
            phsCorruptedIndex = (max_value / maxAmplitude) * numberPHSBins
            InitList.filterList.maxAmplitude.append(max_value)
            InitList.filterList.maxAmplitude_GT.append(max_value_GT)
            InitList.filterList.Area.append(Area)
            InitList.filterList.Area_GT.append(Area_GT)
            InitList.filterList.PtD.append(m_PtD)
            InitList.filterList.PtD_GT.append(m_PtD_GT)
            InitList.filterList.PtR.append(m_PtR)
            InitList.filterList.PtR_GT.append(m_PtR_GT)
            InitList.filterList.RatioC.append(m_PtR/ m_PtD)
            InitList.filterList.RatioC_GT.append(m_PtR_GT/ m_PtD_GT)
            InitList.filterList.dfCrossing.append(m_dfCrossing)
            InitList.filterList.dfCrossing_GT.append(m_dfCrossing_GT)            
            
        else:
            phsCorruptedIndex = (min_value / maxAmplitude) * numberPHSBins
            InitList.filterList.maxAmplitude.append(min_value)
            InitList.filterList.maxAmplitude_GT.append(min_value_GT)
            InitList.filterList.Area.append(Area)
            InitList.filterList.Area_GT.append(Area_GT)
            InitList.filterList.PtD.append(m_PtD)
            InitList.filterList.PtD_GT.append(m_PtD_GT)
            InitList.filterList.PtR.append(m_PtR)
            InitList.filterList.PtR_GT.append(m_PtR_GT)
            InitList.filterList.RatioC.append(m_PtR/ m_PtD)
            InitList.filterList.RatioC_GT.append(m_PtR_GT/ m_PtD_GT)
            InitList.filterList.dfCrossing.append(m_dfCrossing)
            InitList.filterList.dfCrossing_GT.append(m_dfCrossing_GT)
            InitList.filterList.amplitude.append(min_value)
            InitList.filterList.amplitude_GT.append(min_value_GT)
            InitList.filterList.risetime.append(risetime_10p)
            InitList.filterList.risetime_GT.append(risetime_10p_GT)
            InitList.filterList.falltime.append(falltime_10p)
            InitList.filterList.falltime_GT.append(falltime_10p_GT)
            InitList.filterList.ratio_rt_fl.append(ratio_rt_ft)
            InitList.filterList.ratio_rt_fl_GT.append(ratio_rt_ft_GT)
            InitList.filterList.pulsewidth.append(pulslength_10p)
            InitList.filterList.pulsewidth_GT.append(pulslength_10p_GT)
            
        # Nach der while-Schleife: Bedingung erfüllt, verarbeite die Pulse
        if pulseInfo.m_amplitude_in_milliVolt > 0:
            phsAIndex = (pulseA.getMaximumVoltage() / maxAmplitude) * numberPHSBins
            phsBIndex = (pulseB.getMaximumVoltage() / maxAmplitude) * numberPHSBins
            
        else:
            phsAIndex = (pulseA.getMinimumVoltage() / maxAmplitude) * numberPHSBins
            phsBIndex = (pulseB.getMinimumVoltage() / maxAmplitude) * numberPHSBins


        if phsAIndex <= numberPHSBins - 1 and phsAIndex >= 0:
            phsA[int(phsAIndex)] += 1
    
        if phsBIndex <= numberPHSBins - 1 and phsBIndex >= 0:
            phsB[int(phsBIndex)] += 1
            
        if phsCorruptedIndex <= numberPHSBins - 1 and phsCorruptedIndex >= 0:
            phsCorrupted[int(phsCorruptedIndex)] += 1
    
        if not pulsesShown:  # Die ersten Pulse anzeigen
            pulsesShown = True
    
            fig, ax = plt.subplots()
    
            plt.plot(pulseA.getTime(), pulseA.getVoltage(), 'r-', label="pulse-A")
            plt.plot(pulseB.getTime(), pulseB.getVoltage(), 'b-', label="pulse-B")
    
            plt.legend(loc='best')
    
            ax.set_ylabel('amplitude [mV]')
            ax.set_xlabel('time [ns]')
    
            if pulseInfo.m_amplitude_in_milliVolt < 0.:
                ax.set_ylim([maxAmplitude, 50.])
            else:
                ax.set_ylim([-50., maxAmplitude])
    
            ax.set_xlim([0., sweep_in_ns])
    
            plt.show()
            

    # show the resulting phs ...
    end_time = time.time()

    print(f"Berechnung abgeschlossen in {end_time - start_time:.2f} Sekunden")
    print(" ")
    
    fig,ax = plt.subplots()
    
    voltOutA   = []
    voltOutB   = []
    voltOutCorrupted = []
    sumPHSOutA  = 0
    sumPHSOutB  = 0
    sumPHSOutCorrupted  = 0
    normPHSOutA = phsA/np.sum(phsA)
    normPHSOutB = phsB/np.sum(phsB)
    normPHSOutCorrupted = phsCorrupted/np.sum(phsCorrupted)
    
    for i in range(0,len(phsA)):
        voltOutA.append(i*np.abs(maxAmplitude/numberPHSBins))
        sumPHSOutA = sumPHSOutA + np.abs(maxAmplitude/numberPHSBins)*normPHSOutA[i]
    for i in range(0,len(phsB)):
        voltOutB.append(i*np.abs(maxAmplitude/numberPHSBins))
        sumPHSOutB = sumPHSOutB + np.abs(maxAmplitude/numberPHSBins)*normPHSOutB[i]
        
    for i in range(0,len(phsCorrupted)):
        voltOutCorrupted.append(i*np.abs(maxAmplitude/numberPHSBins))
        sumPHSOutCorrupted = sumPHSOutCorrupted + np.abs(maxAmplitude/numberPHSBins)*normPHSOutCorrupted[i]
    
    
    plt.plot(voltOutCorrupted,normPHSOutCorrupted/sumPHSOutCorrupted,'ro',label="PHS output Corrupted")
    plt.plot(voltOutA,normPHSOutA/sumPHSOutA,'rx',label="PHS output A")
    plt.plot(voltOutB,normPHSOutB/sumPHSOutB,'bx',label="PHS output B")
    
    
    
    voltIn    = []
    sumPHSIn  = 0
    normPHSIn = (startPHS+stopPHS)/(np.sum(startPHS) + np.sum(stopPHS))
    
    for i in range(0,len(startPHS)):
        voltIn.append(i*phs.m_resolutionMilliVoltPerStepA)
        sumPHSIn = sumPHSIn + normPHSIn[i]*phs.m_resolutionMilliVoltPerStepA
        
    #plt.plot(voltIn,normPHSIn/sumPHSIn,'g-',label="PHS input")
       
    plt.legend(loc='best')
     
    ax.set_ylabel('pdf [a.u.]')
    ax.set_xlabel('amplitude distribution [mV]')
            
    ax.set_xlim([0.,np.abs(pulseInfo.m_amplitude_in_milliVolt)])
    
    plt.show()
    
    all_list = [InitList.filterList.amplitude, InitList.filterList.risetime, InitList.filterList.falltime, InitList.filterList.ratio_rt_fl, InitList.filterList.pulsewidth, InitList.filterList.Area,InitList.filterList.dfCrossing,InitList.filterList.RatioC]
    # Column Stack -> jede Zeile enthält die Elemente desselben Index aus allen Listen
    Feature_list = np.column_stack(all_list)

    all_list_GT = [InitList.filterList.amplitude_GT, InitList.filterList.risetime_GT, InitList.filterList.falltime_GT, InitList.filterList.ratio_rt_fl_GT, InitList.filterList.pulsewidth_GT, InitList.filterList.Area_GT,InitList.filterList.dfCrossing_GT,InitList.filterList.RatioC_GT]
    # Column Stack -> jede Zeile enthält die Elemente desselben Index aus allen Listen
    Feature_list_GT = np.column_stack(all_list_GT)

    columns = ["Amplitude", "Risetime 10% to 100%", "Falltime 100% to 10%", "Ratio Risetime/Falltime", "Pulsewidth 10% to 10%", "Area", "Difference Zeros f''", "ratio zeros f' and f''"]
    
   
    # 1. Datei zum Schreiben öffnen
    with h5py.File("Feature-List.h5", "w") as f:
        # 2. Dataset erstellen und Daten speichern
        dset = f.create_dataset("my_data", data=Feature_list)
        
        # 3. Spaltennamen als Attribut ablegen
        # HDF5 speichert Strings/Listen am besten als numpy-array (Typ "object" oder via ASCII-Codierung).
        dset.attrs["columns"] = np.array(columns, dtype=object)
    print("HDF5-Datei 'example.h5' wurde erstellt.")
    with h5py.File("Feature-List_GT.h5", "w") as f:
        # 2. Dataset erstellen und Daten speichern
        dset = f.create_dataset("my_data", data=Feature_list_GT)
        
        # 3. Spaltennamen als Attribut ablegen
        # HDF5 speichert Strings/Listen am besten als numpy-array (Typ "object" oder via ASCII-Codierung).
        dset.attrs["columns"] = np.array(columns, dtype=object)

    
    print("HDF5-Datei 'example.h5' wurde erstellt.")

