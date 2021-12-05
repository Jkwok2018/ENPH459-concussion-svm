import numpy as np
    
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
    for col in np.arange(1,colSize+1).reshape(-1):
        roundedCol = np.round(eegMatrix(:,col),decimalPlace)
        freqTable = tabulate(roundedCol)
        freqTable[:,3] = freqTable(:,3) / 100
        freqRow,freqCol = freqTable.shape
        for i in np.arange(1,freqRow+1).reshape(-1):
            shannonEntropyMatrix[1,col] = shannonEntropyMatrix(1,col) + (- freqTable(i,3)) * (np.log(freqTable(i,3)))
    
    return shannonEntropyMatrix
    
    return shannonEntropyMatrix