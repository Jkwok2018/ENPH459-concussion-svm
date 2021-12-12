import numpy as np
from tabulate import tabulate
    
def getShannonEntropy(eegMatrix = None,decimalPlace = None): 
    #GETSHANNONENTROPY Takes in matrix of EEG data and returns a 1x27 matrix
# with Shannon Entropy of each channel.
#   Input: eegMatrix - a nx27 matrix of the eeg data for each channel
#          decimalPlace - how many digits to the right to round the signals
#          to, without rounding there are too many distinct signals
#   Output: shannonEntropyMatrix - a 1x27 matrix of the shannon entropy of
#   each channel. This can then be added to the feature matrix.
    
    rowSize,colSize = eegMatrix.shape
    shannonEntropyMatrix = np.zeros((1,colSize))
    for col in range(colSize):
        roundedCol = np.round(eegMatrix[:,col],decimalPlace)
        unique_ele, counts_ele = np.unique(roundedCol, return_counts= True)
        counts_ele = (counts_ele/len(roundedCol))

        for i in range(len(unique_ele)):
            shannonEntropyMatrix[0,col] += (-counts_ele[i]) * (np.log(counts_ele[i]))
    return shannonEntropyMatrix
    