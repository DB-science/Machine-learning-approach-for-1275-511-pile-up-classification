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

import matplotlib.pyplot as plt

def plotting(amplitude, Area,
             amplitude_GT,Area_GT, 
             amplitude_upperlimit,area_upperlimit, 
             differenceCrossing, differenceCrossing_GT,
             amplitude_upperlimit_crossong, crossing_difference_upperlimit, 
             Ratio_Zero_crossing,Ratio_Zero_crossing_GT,
             amplitude_downlimit, crossingratio_list):
    

    
    
    plt.figure(figsize=(10,6))
    # Originalsignal
    plt.plot(amplitude, Area, marker='o', linestyle='None', label="real", color="blue")
    plt.plot(amplitude_GT,Area_GT, marker='x', linestyle='None', label="GT", color="red")
    plt.plot(amplitude_upperlimit,area_upperlimit, marker='_', linestyle='None', label="limit", color="black")
    plt.xlabel("Amplitude")
    plt.ylabel("Area ")
    plt.title("Area-Filter")
    plt.legend()
    plt.grid()
    plt.show()
    
   
    plt.figure(figsize=(10,6))
    # Originalsignal
    plt.plot(amplitude, differenceCrossing, marker='o', linestyle='None', label="Crossing-difference", color="blue")
    plt.plot(amplitude_GT, differenceCrossing_GT, marker='x', linestyle='None', label="Crossing-difference-GroundTruth", color="red")
    plt.plot(amplitude_upperlimit_crossong, crossing_difference_upperlimit, marker='_', linestyle='None', label="Limit", color="black")   
    plt.xlabel("amplitude [mV]")
    plt.ylabel("crossing-difference[s]")
    plt.title("Crossing-difference")
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure(figsize=(10,6))
    # Originalsignal
    plt.plot(amplitude, Ratio_Zero_crossing, marker='o', linestyle='None', label="Ratio_crossing", color="green")
    plt.plot(amplitude_GT, Ratio_Zero_crossing_GT, marker='x', linestyle='None', label="Ratio_crossing_GT", color="red")
    plt.plot(amplitude_downlimit, crossingratio_list, marker='_', linestyle='None', label="Limit", color="black")
    plt.xlabel("amplitude [mV]")
    plt.ylabel("Ratio crossing")
    plt.tight_layout()  # Automatisches Layout
    #plt.yscale("log")
    plt.title("Ratio")
    plt.legend()
    plt.grid()
    plt.show()

    
    
