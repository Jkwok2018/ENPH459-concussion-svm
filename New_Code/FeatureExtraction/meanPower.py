# Author: Dylan Patrick Whitney
# Date: March 25, 2018
# Purpose: To compute the mean absolute power given a frequency band
# Params: Inputs: fourierTransform: discrete fourier transform of time series data
#                                   frequencyIncrement: the change in frequency with each
#                                   change of array index
#                 numRows: number of rows in the feature matrix
#                 startFrequency: starting frequency of the frequency band
#                 endFrequency: ending frequency of the frequency band
#         Output: power: the mean absolute power for the given frequency
#                        band

import numpy as np
    
def meanPower(fourierTransform = None,frequencyIncrement = None,numRows = None,startFrequency = None,endFrequency = None): 
    currentFrequency = frequencyIncrement
    frequencyBand = []
    for k in range(numRows):
        currentFrequency = currentFrequency + frequencyIncrement
        if currentFrequency < startFrequency:
            continue
        frequencyBand.append(fourierTransform[k])
        if currentFrequency > endFrequency:
            break
    
    frequencyBand = list(filter(lambda x: x != 0, frequencyBand))
    power = (sum(frequencyBand)) / len(frequencyBand)
    return power
    