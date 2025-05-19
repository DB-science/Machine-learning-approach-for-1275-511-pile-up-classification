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

class PHS_parameter:
    SolidScintillator = False
    
    #if CrystalScintillator which effective atomic number
    Z_eff = 25 #the effective atomic number of the scintillator
    # Parameter
    PhotonEnergyNr1 = 1275 # in keV, should be the highest enrgy of the photons
    PhotonEnergyNr2 = 511  # in keV
    
    EnergyResolutionCompton = 50  # Energieauflösung des Detektors in keV
    EnergyResolutionPhotoeffect = 0.01 # Energieauflösung des Detektors in keV
    

    NumberOfDigitizationPoints = 1024 #for Drs4 it is 1024
    
    # Energieachse für das Spektrum
    EnergyScaling = np.linspace(0, PhotonEnergyNr1*1.5, NumberOfDigitizationPoints)
    
class filterParameter:
    #Akime-spline render points
    renderPoints = 20
