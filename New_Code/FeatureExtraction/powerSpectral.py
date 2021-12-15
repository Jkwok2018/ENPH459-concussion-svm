# Author: Dylan Patrick Whitney
# Date: March 25, 2018
# Purpose: Returns a 7x27 feature matrix given a set of cleaned and
# interpolated EEG Data in .Mat format.
# Params: eegMat: EEG Data in .Mat format
# featureMatrix: 7x27 feature matrix. The 7 features in order are: mean
# theta power, mean alpha power, mean beta power, mean gamma power, alphaPower/thetaPower, betaPower/alphaPower, gammaPower/betaPower.

import numpy as np
from . import meanPower
import matplotlib.pyplot as plt
# import .meanPower as mp
    
def powerSpectral(eegMat = None): 
    samplingFrequency = 250
    
    numRows = eegMat.shape[0]
    # print('numRows', numRows)
    frequencyIncrement = samplingFrequency / numRows
    featureMatrix = np.zeros((7, 27))
    fig, axs = plt.subplots(27)
    # For each of the 27 channels, compute the discrete fourier transform, find
# the frequency bands that correspond to brain waves (i.e Alpha, Beta,
# Gamma, Theta), and compute the mean absolute powers
    for i in range(27):
        y = (np.abs(np.fft.fft(eegMat[:,i])) ** 2) / numRows

        frequency = np.linspace(0, samplingFrequency/2, len(y))

        axs[i].plot(frequency, y)
        # print(len(y))

        
        #f = (0:length(y)-1)*(250/length(y));
#h=plot(f(1:15000),y(1:15000))
#xlabel('Frequency (Hz)')
#ylabel('Power')
#title('Power Spectrum vs. Frequency')
# saveas(h,'filename.jpg')
        currentFrequency = frequencyIncrement
        #theta = 4-8Hz
        theta = []
        for k in range(numRows):
            currentFrequency = currentFrequency + frequencyIncrement
            if currentFrequency < 4:
                continue
            theta.append(y[k])
            if currentFrequency >= 8:
                break

        theta = list(filter(lambda x: x != 0, theta))
        thetaPower = (sum(theta)) / len(theta)

        featureMatrix[0,i] = thetaPower
        alphaPower = meanPower.meanPower(y,frequencyIncrement,numRows,8,12)
        featureMatrix[1,i] = alphaPower
        betaPower = meanPower.meanPower(y,frequencyIncrement,numRows,12,30)
        featureMatrix[2,i] = betaPower
        gammaPower = meanPower.meanPower(y,frequencyIncrement,numRows,30,50)
        featureMatrix[3,i] = gammaPower
        # Compute the ratios of absolute power in adjacent frequency bands
        featureMatrix[4,i] = alphaPower / thetaPower
        featureMatrix[5,i] = betaPower / alphaPower
        featureMatrix[6,i] = gammaPower / betaPower
    plt.show()
    return featureMatrix